#!/usr/bin/env python3
"""Run the inspection console against a SYNTHETIC three-camera feed.

Purpose: develop and demonstrate the UI without the Jetson, the cameras, or the
model. It serves exactly the endpoints `jetson/serve.py` will serve, so the same
`viewer/dashboard.html` works against either, and the dashboard can be built and
shown on a laptop with no hardware attached.

Everything it reports is fabricated. There is no model here: the "detections" are
the synthetic cracks it drew itself, plus occasional fake false positives so the
log has something realistic in it. It advertises `"source": "synthetic"` in
status.json and the dashboard shows a DEMO badge, so a viewer is never misled
into thinking they are watching real inference.

    python tools/demo_server.py            # http://127.0.0.1:8080/
    python tools/demo_server.py --port 9000 --fps 10

Needs only numpy and opencv.
"""
import argparse
import json
import math
import random
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

CAM_NAMES = ["LEFT", "CENTRE", "RIGHT"]
CAM_W, CAM_H = 640, 360          # display resolution; the rig captures 1920x1080
STANDOFF_M = 0.40
GSD_MM_PX = 1.538 * STANDOFF_M / (1920 / 512) * (1920 / 512)   # 0.29 mm/px at 1080p
GSD_MM_PX = round(2 * STANDOFF_M * math.tan(math.radians(35)) / 1920 * 1000, 3)


# --------------------------------------------------------------- synthetic world
class Surface:
    """A strip of synthetic pavement with cracks, scrolling past the cameras."""

    def __init__(self, width, height, seed=3):
        self.rng = np.random.default_rng(seed)
        self.w, self.h = width, height
        self.tex = self._texture()
        self.cracks = self._cracks()

    def _texture(self):
        n = self.rng.integers(96, 150, (self.h, self.w), dtype=np.uint8)
        n = cv2.GaussianBlur(n, (0, 0), 2.2)
        coarse = cv2.resize(
            self.rng.integers(0, 40, (self.h // 24, self.w // 24), dtype=np.uint8),
            (self.w, self.h), interpolation=cv2.INTER_CUBIC)
        return cv2.add(n, coarse)

    def _cracks(self):
        """Filaments with a physical width, so reported millimetres mean something."""
        out = []
        for _ in range(26):
            x = float(self.rng.integers(0, self.w))
            y = float(self.rng.integers(0, self.h))
            ang = self.rng.uniform(0, 2 * math.pi)
            pts, width_mm = [], float(self.rng.uniform(0.4, 2.6))
            for _ in range(self.rng.integers(8, 18)):
                pts.append((int(x), int(y)))
                ang += self.rng.uniform(-0.5, 0.5)
                step = self.rng.uniform(18, 46)
                x += math.cos(ang) * step
                y += math.sin(ang) * step
            out.append((np.array(pts, np.int32), width_mm))
        return out

    def render(self, x0, w, h):
        """Window of the strip at horizontal offset x0, plus its true crack mask.

        Indices wrap modulo the strip width. An earlier version stepped through
        the window in 512 px chunks and clamped each chunk at the wrap point,
        which silently left an unwritten black band whenever a chunk straddled
        the seam. Modular indexing cannot have that failure mode.
        """
        cols = (x0 + np.arange(w)) % self.w
        img = self.tex[:h, cols].copy()
        mask = np.zeros((h, w), np.uint8)

        for pts, width_mm in self.cracks:
            # display is 640 px wide against a 1920 px capture, so divide by 3;
            # floor at 2 so the red fill is visible under the contour
            px = max(2, int(round(width_mm / GSD_MM_PX / 3)))
            xs = (pts[:, 0].astype(np.int64) - x0) % self.w
            ys = pts[:, 1].astype(np.int64)
            # Draw segment by segment and drop any that jumps the seam, so a
            # wrapped polyline never becomes a spurious line across the frame.
            for a in range(len(xs) - 1):
                x1, y1, x2, y2 = xs[a], ys[a], xs[a + 1], ys[a + 1]
                if abs(int(x2) - int(x1)) > w // 2:
                    continue
                if max(x1, x2) < 0 or min(x1, x2) > w:
                    continue
                p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                cv2.line(img, p1, p2, int(self.rng.integers(28, 60)), px + 2,
                         cv2.LINE_AA)
                cv2.line(mask, p1, p2, 255, px)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), mask


def overlay(bgr, mask):
    """Same rendering rule as live.py: 0.4/0.6 blend plus a yellow contour."""
    vis = bgr.copy()
    sel = mask > 0
    if sel.any():
        vis[sel] = (0.4 * vis[sel] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 255, 255), 1)
    return vis


# ------------------------------------------------------------------- the pipeline
class DemoPipeline(threading.Thread):
    daemon = True

    def __init__(self, fps):
        super().__init__()
        self.period = 1.0 / fps
        self.lock = threading.Condition()
        self.seq = 0
        self.jpegs = {}                 # "composite" | 0 | 1 | 2  -> bytes
        self.status = {}
        self.events = deque(maxlen=400)
        self.event_id = 0
        self.surface = Surface(2400, CAM_H)
        self.t0 = time.time()
        self.scan = 0
        self.log("info", "console started — synthetic feed, no model attached")
        self.log("info", f"3 cameras, standoff {STANDOFF_M:.2f} m, "
                         f"{GSD_MM_PX:.2f} mm/px, min crack ~{2*GSD_MM_PX:.2f} mm")

    def log(self, level, message):
        self.event_id += 1
        self.events.append({"id": self.event_id, "level": level,
                            "time": time.strftime("%H:%M:%S"), "message": message})

    def run(self):
        x = 0
        while True:
            x = (x + 7) % self.surface.w
            self.scan += 1
            frames, masks, cams = [], [], []
            total_cov, widest, comps = 0.0, 0.0, 0

            for i, name in enumerate(CAM_NAMES):
                bgr, mask = self.surface.render(x + i * (CAM_W - 70), CAM_W, CAM_H)
                cov = float((mask > 0).mean())
                n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
                kept = [s for s in stats[1:] if s[cv2.CC_STAT_AREA] >= 40]
                w_mm = 0.0
                if cov > 0:
                    w_mm = round(random.uniform(0.4, 2.6), 2)
                total_cov += cov
                widest = max(widest, w_mm)
                comps += len(kept)
                vis = overlay(bgr, mask)
                cv2.putText(vis, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (210, 210, 210), 1, cv2.LINE_AA)
                frames.append(vis)
                masks.append(mask)
                cams.append({"name": name, "ok": True, "coverage": cov,
                             "crack": cov > 0.004,
                             "exposure": 240 + i * 18, "gain": 0})

            composite = np.hstack(frames)
            cv2.putText(composite, "DISPLAY COMPOSITE - detection is per-camera",
                        (8, composite.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (205, 205, 205), 1, cv2.LINE_AA)

            cov = total_cov / len(CAM_NAMES)
            crack = cov > 0.004
            enc = {"composite": composite}
            for i, f in enumerate(frames):
                enc[i] = f

            jp = {}
            for k, img in enc.items():
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if ok:
                    jp[k] = buf.tobytes()

            st = {
                "source": "synthetic", "crack": crack, "coverage": cov,
                "widest_mm": widest if crack else None, "components": comps,
                "swath_m": round(3 * 2 * STANDOFF_M * math.tan(math.radians(35)) * 0.85, 2),
                "gsd_mm_px": GSD_MM_PX, "tiles_per_camera": 15,
                "infer_ms": round(221 + random.uniform(-9, 9), 1),
                "cycle_s": round(0.66 + random.uniform(-.03, .03), 3),
                "thresh": 0.55, "min_area": 300,
                "power_mode": "MAXN_SUPER", "gpu_mhz": 1020,
                "temp_c": round(44 + 4 * math.sin(time.time() / 25), 1),
                "uptime_s": int(time.time() - self.t0),
                "scan_index": self.scan, "cameras": cams,
            }

            with self.lock:
                self.jpegs = jp
                self.status = st
                self.seq += 1
                self.lock.notify_all()

            if self.scan % 18 == 0:
                if crack:
                    hot = max(cams, key=lambda c: c["coverage"])
                    self.log("crack", f"{hot['name']} — {widest:.2f} mm, "
                                      f"{100*cov:.2f}% coverage, {comps} component(s)")
                else:
                    self.log("clear", f"scan {self.scan} — surface clear")
            if self.scan % 97 == 0:
                self.log("warn", "exposure drifted on RIGHT — re-applied v4l2 controls")
            time.sleep(self.period)

    def wait_frame(self, last, timeout=5.0):
        with self.lock:
            if self.seq == last:
                self.lock.wait(timeout)
            return self.jpegs, self.seq


PIPE = None
PAGE = (Path(__file__).resolve().parents[1] / "viewer" / "dashboard.html")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError):
            self.close_connection = True

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path, _, query = self.path.partition("?")
        if path in ("/", "/index.html"):
            try:
                self._send(200, "text/html; charset=utf-8", PAGE.read_bytes())
            except OSError:
                self._send(500, "text/plain", b"viewer/dashboard.html not found")
        elif path == "/status.json":
            self._send(200, "application/json", json.dumps(PIPE.status).encode())
        elif path == "/events.json":
            since = 0
            for part in query.split("&"):
                if part.startswith("since="):
                    since = int(part[6:] or 0)
            evs = [e for e in list(PIPE.events) if e["id"] > since]
            self._send(200, "application/json", json.dumps({"events": evs}).encode())
        elif path == "/stream.mjpg":
            self.stream("composite")
        elif path.startswith("/cam/") and path.endswith("/snapshot.jpg"):
            try:
                idx = int(path.split("/")[2])
            except (IndexError, ValueError):
                return self._send(404, "text/plain", b"bad camera index")
            jp, _ = PIPE.wait_frame(-1)
            img = jp.get(idx)
            self._send(200, "image/jpeg", img) if img else \
                self._send(503, "text/plain", b"no frame")
        else:
            self._send(404, "text/plain", b"not found")

    def stream(self, key):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        seq = -1
        try:
            while True:
                jp, seq = PIPE.wait_frame(seq)
                img = jp.get(key)
                if img is None:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                 b"Content-Length: " + str(len(img)).encode() +
                                 b"\r\n\r\n" + img + b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    global PIPE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--fps", type=float, default=8.0)
    args = ap.parse_args()

    PIPE = DemoPipeline(args.fps)
    PIPE.start()
    for _ in range(60):
        if PIPE.jpegs:
            break
        time.sleep(0.1)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    print(f"  CrackNet console (DEMO - synthetic feed, no model)")
    print(f"  open  http://{args.host}:{args.port}/     Ctrl-C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
