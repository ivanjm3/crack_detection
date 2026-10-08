#!/usr/bin/env python3
"""Serve the three-camera inspection console.

Same contract as tools/demo_server.py, which was built against this exact set of
endpoints so the dashboard could be developed with no hardware attached. The
page itself is unchanged - viewer/dashboard.html runs against either server, and
the only difference it can see is status.source, which reads "synthetic" there
and "jetson" here. That is the whole point of having built the demo first.

  /                   viewer/dashboard.html
  /stream.mjpg        composite of the three cameras, MJPEG
  /cam/<i>/snapshot.jpg  one camera's overlaid frame
  /status.json        verdict, measurements, pipeline and system state
  /events.json?since= append-only event log

The scan loop runs in one background thread and publishes finished JPEGs;
handlers only ever read what it published, so no number of viewers can slow a
scan down, and a scan is never half-published.

    python3 serve_scan.py                       # port 8080
    python3 serve_scan.py --standoff 0.35
"""
import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

import scanner
from scanner import Scanner, composite, overlay

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE_CANDIDATES = (
    os.path.join(HERE, "dashboard.html"),            # uploaded next to this file
    os.path.join(HERE, "..", "viewer", "dashboard.html"),   # in the repo
)


def load_page():
    for path in PAGE_CANDIDATES:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return f.read()
    return (b"<!doctype html><title>dashboard missing</title>"
            b"<p>dashboard.html was not found next to serve_scan.py or in "
            b"../viewer/. Endpoints still work: /status.json, /stream.mjpg")


def soc_temp_c():
    """Hottest thermal zone, or None. Not every zone exists on every L4T build,
    and a missing sensor is not a reason to fail a scan."""
    best = None
    for zone in range(12):
        path = "/sys/class/thermal/thermal_zone%d/temp" % zone
        try:
            with open(path) as f:
                milli = int(f.read().strip())
        except (OSError, ValueError):
            continue
        c = milli / 1000.0
        if 0 < c < 150 and (best is None or c > best):
            best = c
    return best


def gpu_mhz():
    base = "/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/cur_freq"
    try:
        with open(base) as f:
            return int(int(f.read().strip()) / 1e6)
    except (OSError, ValueError):
        return None


def power_mode():
    try:
        with open("/etc/nvpmodel.conf") as f:
            conf = f.read()
        with open("/var/lib/nvpmodel/status") as f:
            # The file reads "pmode:0002" - zero-padded, so the id has to be
            # parsed as a number. Matching the raw string against "ID=2" in
            # nvpmodel.conf fails and falls through to a bare mode number.
            cur = int(f.read().strip().split(":")[-1])
        for line in conf.splitlines():
            line = line.strip()
            if line.startswith("< POWER_MODEL") and "ID=%d " % cur in line:
                return line.split("NAME=")[-1].rstrip(" >").strip()
        return "mode %d" % cur
    except (OSError, IndexError, ValueError):
        return None


class ScanLoop(threading.Thread):
    daemon = True

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.lock = threading.Condition()
        self.jpegs = {}
        self.status = {"source": "jetson", "scan_index": 0}
        self.events = []
        self.event_id = 0
        self.seq = 0
        self.t0 = time.time()
        self.stop_flag = threading.Event()

    def log(self, level, message):
        self.event_id += 1
        self.events.append({"id": self.event_id, "level": level,
                            "time": time.strftime("%H:%M:%S"),
                            "message": message})
        del self.events[:-400]          # the page only ever shows the tail

    def run(self):
        # pycuda.autoinit built the CUDA context on the MAIN thread and a
        # context is owned by one thread at a time, so it has to be pushed here
        # before anything touches TensorRT. Without this every cuda.* call
        # raises "invalid device context".
        import pycuda.autoinit
        ctx = pycuda.autoinit.context
        ctx.push()
        try:
            self._loop()
        finally:
            ctx.pop()

    def _loop(self):
        a = self.args
        try:
            sc = Scanner(a)
        except Exception as exc:
            self.log("warn", "scanner failed to start: %s" % exc)
            raise
        self.log("info", sc.describe())
        self.log("info", "min crack resolvable ~%.2f mm (2 px)" % (2 * sc.gsd))

        last_verdict = None
        while not self.stop_flag.is_set():
            try:
                cams, tot = sc.scan()
            except Exception as exc:
                self.log("warn", "scan failed: %s" % exc)
                time.sleep(2.0)
                continue

            jp = {}
            ok, buf = cv2.imencode(".jpg", composite(cams, a.composite_width),
                                   [cv2.IMWRITE_JPEG_QUALITY, a.quality])
            if ok:
                jp["composite"] = buf.tobytes()
            for i, c in enumerate(cams):
                vis = overlay(c["frame"], c["mask"])
                h = int(vis.shape[0] * a.thumb_width / vis.shape[1])
                vis = cv2.resize(vis, (a.thumb_width, h),
                                 interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", vis,
                                       [cv2.IMWRITE_JPEG_QUALITY, a.quality])
                if ok:
                    jp[i] = buf.tobytes()

            st = {
                "source": "jetson",
                "crack": tot["crack"],
                "coverage": tot["coverage"],
                "widest_mm": tot["widest_mm"],
                "components": tot["components"],
                "swath_m": round(3 * sc.gsd * a.width / 1000.0, 2),
                "gsd_mm_px": round(sc.gsd, 3),
                "tiles_per_camera": len(sc.tiles),
                "infer_ms": tot["infer_s"] * 1000.0 / max(len(cams), 1),
                "cycle_s": tot["cycle_s"],
                "capture_s": tot["capture_s"],
                "thresh": a.thresh,
                "min_area": a.min_area,
                "power_mode": power_mode(),
                "gpu_mhz": gpu_mhz(),
                "temp_c": soc_temp_c(),
                "uptime_s": int(time.time() - self.t0),
                "scan_index": tot["scan_index"],
                # The page wants cameras without the pixel data in them.
                "cameras": [{k: v for k, v in c.items()
                             if k not in ("frame", "mask")} for c in cams],
            }

            # Log on CHANGE, plus a periodic heartbeat. Logging every scan would
            # push a line every ~6 s and bury the one line that matters; logging
            # only on change would leave the panel looking frozen during a long
            # clear run, which is indistinguishable from a crashed loop.
            if tot["crack"] != last_verdict:
                if tot["crack"]:
                    hot = max(cams, key=lambda c: c["coverage"])
                    self.log("crack", "%s - %s, %.2f %% coverage, %d component(s)"
                             % (hot["name"],
                                ("%.2f mm widest" % tot["widest_mm"])
                                if tot["widest_mm"] else "width unresolved",
                                100 * tot["coverage"], tot["components"]))
                else:
                    self.log("clear", "surface clear")
                last_verdict = tot["crack"]
            elif tot["scan_index"] % 10 == 0:
                self.log("info", "scan %d - %s, %.2f %% coverage, cycle %.2f s"
                         % (tot["scan_index"],
                            "crack" if tot["crack"] else "clear",
                            100 * tot["coverage"], tot["cycle_s"]))

            for c in cams:
                if not c["converged"]:
                    self.log("warn", "%s exposure did not converge "
                                     "(exp %d gain %d, luma %.0f)"
                             % (c["name"], c["exposure"], c["gain"],
                                c["median_luma"]))

            with self.lock:
                self.jpegs = jp
                self.status = st
                self.seq += 1
                self.lock.notify_all()

            if a.interval:
                time.sleep(a.interval)
        sc.close()

    def wait_frame(self, last, timeout=20.0):
        """Block until a scan newer than `last` is published.

        The timeout is generous because a scan position takes ~6 s - the
        single-camera server's 5 s default would time out mid-scan and the
        stream would stutter on every cycle.
        """
        with self.lock:
            if self.seq == last:
                self.lock.wait(timeout)
            return self.jpegs, self.seq


PIPE = None
PAGE = None


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
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", PAGE)
        elif path == "/status.json":
            self._send(200, "application/json", json.dumps(PIPE.status).encode())
        elif path == "/events.json":
            since = 0
            if "since=" in self.path:
                try:
                    since = int(self.path.split("since=")[1].split("&")[0])
                except ValueError:
                    since = 0
            evs = [e for e in PIPE.events if e["id"] > since]
            self._send(200, "application/json",
                       json.dumps({"events": evs}).encode())
        elif path == "/stream.mjpg":
            self.stream()
        elif path.startswith("/cam/") and path.endswith("/snapshot.jpg"):
            try:
                idx = int(path.split("/")[2])
            except (IndexError, ValueError):
                return self._send(404, "text/plain", b"bad camera index")
            jp, _ = PIPE.wait_frame(-1)
            img = jp.get(idx)
            if img:
                self._send(200, "image/jpeg", img)
            else:
                self._send(503, "text/plain", b"no frame")
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
                jp, seq = PIPE.wait_frame(seq)
                img = jp.get("composite")
                if img is None:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                 b"Content-Length: " +
                                 str(len(img)).encode() + b"\r\n\r\n")
                self.wfile.write(img)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    global PIPE, PAGE
    ap = scanner.add_arguments(argparse.ArgumentParser())
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--interval", type=float, default=0.0,
                    help="extra pause between scan positions")
    ap.add_argument("--composite-width", type=int, default=1440)
    ap.add_argument("--thumb-width", type=int, default=420)
    ap.add_argument("--quality", type=int, default=82)
    args = ap.parse_args()

    PAGE = load_page()
    PIPE = ScanLoop(args)
    PIPE.start()

    srv = Server((args.host, args.port), Handler)
    print("serving on http://%s:%d/  (Ctrl-C to stop)" % (args.host, args.port))
    print("first scan takes longer: the exposure prime runs once per camera")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        PIPE.stop_flag.set()
        srv.server_close()


if __name__ == "__main__":
    main()
