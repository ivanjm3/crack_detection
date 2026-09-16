#!/usr/bin/env python3
"""Serve the live CrackNet overlay as MJPEG over HTTP.

Open http://<jetson-ip>:8080/ from any machine that can reach the Jetson.
Stdlib only - no Flask, no extra pip installs.

Endpoints:
  /            small HTML viewer page
  /stream.mjpg raw multipart/x-mixed-replace MJPEG stream
  /snapshot.jpg single most recent frame
  /status.json  current coverage / fps / inference time

The capture+inference loop runs in one background thread and publishes the
latest encoded JPEG; HTTP handlers only ever read that, so any number of
viewers (or none) cannot slow inference down.
"""
import argparse, json, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from camera_ctl import AutoExposure
from live import open_c920, center_square, clean_mask
from trt_infer import CrackNetTRT

ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE,
             180: cv2.ROTATE_180,
             270: cv2.ROTATE_90_COUNTERCLOCKWISE}

PAGE = b"""<!doctype html>
<title>CrackNet live</title>
<style>
  body{margin:0;background:#111;color:#eee;font:14px system-ui,sans-serif;
       display:flex;flex-direction:column;align-items:center;gap:12px;padding:16px}
  img{max-width:min(100%,720px);width:100%;height:auto;border-radius:8px;background:#000}
  #s{font-variant-numeric:tabular-nums;opacity:.85}
  b.crack{color:#ff5252}b.clear{color:#4caf50}
</style>
<h3>CrackNet &mdash; live</h3>
<img src="/stream.mjpg" alt="live overlay">
<div id="s">connecting&hellip;</div>
<script>
setInterval(async()=>{
  try{
    const r=await fetch('/status.json',{cache:'no-store'}), d=await r.json();
    document.getElementById('s').innerHTML=
      `<b class="${d.crack?'crack':'clear'}">${d.crack?'CRACK':'clear'}</b>`+
      ` &nbsp; coverage ${(100*d.coverage).toFixed(2)}%`+
      ` &nbsp; inference ${d.infer_ms.toFixed(1)} ms`+
      ` &nbsp; ${d.fps.toFixed(1)} fps`;
  }catch(e){}
},500);
</script>
"""


class Pipeline(threading.Thread):
    """Camera -> TensorRT -> annotated JPEG, published for HTTP handlers."""

    daemon = True

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.lock = threading.Condition()
        self.jpeg = None
        self.seq = 0
        self.status = {"crack": False, "coverage": 0.0, "infer_ms": 0.0, "fps": 0.0}
        self.stop_flag = threading.Event()

    def run(self):
        # pycuda.autoinit created the context on the MAIN thread, and a CUDA
        # context belongs to one thread at a time. Push it onto this thread
        # before touching TensorRT, or every cuda.* call raises
        # "invalid device context - no currently active context?".
        import pycuda.autoinit
        cuda_ctx = pycuda.autoinit.context
        cuda_ctx.push()
        try:
            self._loop()
        finally:
            cuda_ctx.pop()

    def _loop(self):
        a = self.args
        net = CrackNetTRT(a.engine)
        S = net.size
        cap = open_c920(a.cam)
        print(f"pipeline: model input {S}x{S}, engine {a.engine}", flush=True)

        ae = None
        if not a.no_auto_exposure:
            ae = AutoExposure(cap, target=a.ae_target, gain=a.gain)
            for _ in range(12):
                ok, f0 = cap.read()
                if ok:
                    ae.update(f0, force=True)
            print("auto-exposure settled:", ae.label(), flush=True)

        t_prev, fps, smooth = time.time(), 0.0, 0.0
        while not self.stop_flag.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            if a.rotate:
                frame = cv2.rotate(frame, ROTATIONS[a.rotate])
            crop, _ = center_square(frame)
            if ae:
                ae.update(crop)
            rgb = cv2.cvtColor(cv2.resize(crop, (S, S), interpolation=cv2.INTER_AREA),
                               cv2.COLOR_BGR2RGB)

            t0 = time.time()
            prob = net.infer(CrackNetTRT.preprocess(rgb))
            t_inf = (time.time() - t0) * 1000

            mask = clean_mask((prob > a.thresh).astype(np.uint8) * 255, a.min_area,
                              max_area_frac=a.max_area_frac,
                              max_halfwidth=a.max_halfwidth,
                              max_solidity=a.max_solidity,
                              shape_filter=not a.no_shape_filter)
            frac = float((mask > 0).mean())
            smooth = 0.7 * smooth + 0.3 * frac
            is_crack = smooth > a.alert_frac

            big = cv2.resize(mask, (crop.shape[1], crop.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
            vis = crop.copy()
            vis[big > 0] = (0.4 * vis[big > 0] +
                            0.6 * np.array([0, 0, 255])).astype(np.uint8)
            contours, _ = cv2.findContours(big, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (0, 255, 255), 1)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now
            label = f"{'CRACK' if is_crack else 'clear'}  cov={100*smooth:.2f}%  " \
                    f"inf={t_inf:.1f}ms  fps={fps:.1f}"
            cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255) if is_crack else (0, 200, 0), 2)

            if a.width and vis.shape[1] != a.width:
                h = int(vis.shape[0] * a.width / vis.shape[1])
                vis = cv2.resize(vis, (a.width, h), interpolation=cv2.INTER_AREA)

            ok, buf = cv2.imencode(".jpg", vis,
                                   [cv2.IMWRITE_JPEG_QUALITY, a.quality])
            if not ok:
                continue
            with self.lock:
                self.jpeg = buf.tobytes()
                self.seq += 1
                self.status = {"crack": bool(is_crack), "coverage": smooth,
                               "infer_ms": t_inf, "fps": fps,
                               "exposure": ae.stats["exposure"] if ae else None,
                               "gain": ae.stats["gain"] if ae else None,
                               "median": ae.stats["median"] if ae else None,
                               "clipped": ae.stats["clipped"] if ae else None}
                self.lock.notify_all()

        cap.release()

    def latest(self, last_seq, timeout=5.0):
        """Block until a frame newer than last_seq exists; return (jpeg, seq)."""
        with self.lock:
            if self.seq == last_seq:
                self.lock.wait(timeout)
            return self.jpeg, self.seq


PIPE = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass                                    # keep the console quiet

    def handle_one_request(self):
        # A viewer closing its tab resets the connection mid-read, which
        # socketserver otherwise reports as an unhandled traceback per tab.
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError):
            self.close_connection = True

    def _send(self, code, ctype, body, extra=()):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in extra:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", PAGE)
        elif path == "/status.json":
            self._send(200, "application/json",
                       json.dumps(PIPE.status).encode())
        elif path == "/snapshot.jpg":
            jpeg, _ = PIPE.latest(-1)
            if jpeg is None:
                self._send(503, "text/plain", b"no frame yet")
            else:
                self._send(200, "image/jpeg", jpeg)
        elif path == "/stream.mjpg":
            self.stream()
        else:
            self._send(404, "text/plain", b"not found")

    def stream(self):
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        seq = -1
        try:
            while True:
                jpeg, seq = PIPE.latest(seq)
                if jpeg is None:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                 b"Content-Length: " +
                                 str(len(jpeg)).encode() + b"\r\n\r\n")
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass                                # viewer closed the tab


class Server(ThreadingHTTPServer):   # already threaded; no extra mixin
    daemon_threads = True
    allow_reuse_address = True


def main():
    global PIPE
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="cracknet_fp16.engine")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--thresh", type=float, default=0.55)
    ap.add_argument("--min-area", type=int, default=300)
    ap.add_argument("--alert-frac", type=float, default=0.01)
    ap.add_argument("--max-area-frac", type=float, default=0.20)
    ap.add_argument("--max-halfwidth", type=float, default=12.0)
    ap.add_argument("--max-solidity", type=float, default=0.80)
    ap.add_argument("--no-shape-filter", action="store_true")
    ap.add_argument("--no-auto-exposure", action="store_true")
    ap.add_argument("--ae-target", type=float, default=120.0)
    ap.add_argument("--gain", type=int, default=0)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--width", type=int, default=720,
                    help="downscale the streamed frame (0 = native 720)")
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                    help="rotate frames for a sideways-mounted camera")
    args = ap.parse_args()

    PIPE = Pipeline(args)
    PIPE.start()
    for _ in range(100):                        # wait for the first frame
        if PIPE.jpeg is not None:
            break
        time.sleep(0.1)

    srv = Server((args.host, args.port), Handler)
    print(f"serving on http://{args.host}:{args.port}/  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        PIPE.stop_flag.set()
        srv.server_close()


if __name__ == "__main__":
    main()
