#!/usr/bin/env python3
"""Stream a USB camera over HTTP as MJPEG. Standard library + OpenCV only.

    python3 stream_server.py                  camera 0, port 8080
    python3 stream_server.py --cam 1 --port 8000
    python3 stream_server.py --test-pattern   no camera needed (checks the network)

Open  http://<jetson-address>:8080/   in any browser on the same network.

    /               page with the live video + a status line
    /stream.mjpg    the raw MJPEG stream (use in <img>, VLC, OpenCV)
    /snapshot.jpg   one frame
    /status.json    {"fps": ..., "frames": ..., "uptime_s": ..., "camera": ...}

Put your own processing in `process()`; whatever it returns is what is
streamed and what `status` can report on.
"""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np


def process(frame):
    """Your code goes here: detection, overlays, measurements ...

    Takes a BGR image, returns (image_to_stream, dict_of_extra_status).
    """
    return frame, {}


class Camera(threading.Thread):
    """Reads frames as fast as the camera delivers and keeps only the newest.

    A viewer that is slow must never make the camera fall behind, and a viewer
    that connects late must see *now*, not the backlog - so there is no queue.
    """

    def __init__(self, args):
        super().__init__(daemon=True)
        self.args = args
        self.cv = threading.Condition()
        self.jpeg = None
        self.seq = 0
        self.extra = {}
        self.fps = 0.0
        self.t0 = time.time()
        self.error = None

    def open(self):
        a = self.args
        cap = cv2.VideoCapture(a.cam, cv2.CAP_V4L2)
        # MJPG, not the default YUYV: raw YUYV at 720p+ over USB 2.0 tops out
        # at a few fps and, with several cameras, simply fails.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
        cap.set(cv2.CAP_PROP_FPS, a.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError("cannot open camera %s - is it plugged in, and "
                               "is another program using it?" % a.cam)
        return cap

    def read(self, cap, n):
        if cap is None:                          # --test-pattern
            img = np.full((self.args.height, self.args.width, 3), 40, np.uint8)
            x = (n * 8) % self.args.width
            cv2.rectangle(img, (x, 0), (x + 40, self.args.height), (0, 200, 255), -1)
            cv2.putText(img, time.strftime("%H:%M:%S"), (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
            time.sleep(1.0 / self.args.fps)
            return img
        ok, img = cap.read()
        return img if ok else None

    def run(self):
        try:
            cap = None if self.args.test_pattern else self.open()
        except RuntimeError as exc:
            self.error = str(exc)
            print("ERROR:", exc)
            return
        n, t_fps, c_fps = 0, time.time(), 0
        while True:
            img = self.read(cap, n)
            if img is None:
                time.sleep(0.05)
                continue
            img, extra = process(img)
            ok, buf = cv2.imencode(".jpg", img,
                                   [cv2.IMWRITE_JPEG_QUALITY, self.args.quality])
            if not ok:
                continue
            n += 1
            c_fps += 1
            now = time.time()
            if now - t_fps >= 1.0:
                self.fps = c_fps / (now - t_fps)
                t_fps, c_fps = now, 0
            with self.cv:
                self.jpeg, self.extra, self.seq = buf.tobytes(), extra, n
                self.cv.notify_all()

    def wait(self, last, timeout=5.0):
        with self.cv:
            if self.seq == last:
                self.cv.wait(timeout)
            return self.jpeg, self.seq

    def status(self):
        d = {"camera": self.args.cam, "fps": round(self.fps, 1),
             "frames": self.seq, "uptime_s": int(time.time() - self.t0),
             "error": self.error}
        d.update(self.extra)
        return d


PAGE = b"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Jetson camera</title>
<style>body{margin:0;background:#111;color:#eee;font:16px system-ui}
img{width:100%;max-width:960px;display:block;margin:auto;background:#000}
p{text-align:center;opacity:.8}</style></head><body>
<img src="stream.mjpg" alt="live"><p id=s>connecting...</p>
<script>
async function t(){try{const r=await fetch('status.json',{cache:'no-store'});
const d=await r.json();document.getElementById('s').textContent=
d.error?('camera error: '+d.error):(d.fps+' fps - '+d.frames+' frames - up '+d.uptime_s+' s');
}catch(e){document.getElementById('s').textContent='NO LINK to the Jetson'}}
setInterval(t,1000);t();
</script></body></html>"""

CAM = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError):
            self.close_connection = True       # a viewer closing its tab

    def send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self.send(200, "text/html; charset=utf-8", PAGE)
        elif path == "/status.json":
            self.send(200, "application/json", json.dumps(CAM.status()).encode())
        elif path == "/snapshot.jpg":
            jpg, _ = CAM.wait(-1)
            if jpg:
                self.send(200, "image/jpeg", jpg)
            else:
                self.send(503, "text/plain", b"no frame yet")
        elif path == "/stream.mjpg":
            self.stream()
        else:
            self.send(404, "text/plain", b"not found")

    def stream(self):
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        seq = -1
        try:
            while True:
                jpg, seq = CAM.wait(seq)
                if jpg is None:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                 b"Content-Length: " + str(len(jpg)).encode() +
                                 b"\r\n\r\n" + jpg + b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    global CAM
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cam", type=int, default=0, help="/dev/videoN index")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--quality", type=int, default=80, help="JPEG 1-100")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--test-pattern", action="store_true",
                    help="stream a moving test card instead of a camera")
    args = ap.parse_args()

    CAM = Camera(args)
    CAM.start()
    srv = Server((args.host, args.port), Handler)
    print("serving on http://%s:%d/   (Ctrl-C to stop)" % (args.host, args.port))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
