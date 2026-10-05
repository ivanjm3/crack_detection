#!/usr/bin/env python3
"""Phase 0 feasibility harness for the three-camera rig.

Answers one question: how many USB cameras will actually stream at once on this
board, and at what frame rate?

On this Orin Nano every Type-A port sits behind a single Realtek USB 2.0 hub on
Bus 01 (480 Mbit/s), and the C920 is a USB 2.0 device, so all cameras contend for
one bus. UVC reserves isochronous bandwidth from the figure the camera firmware
declares rather than what it actually sends, so the third camera typically fails
to open with ENOSPC - "No space left on device" - even though the real data rate
would fit easily.

This script opens cameras one at a time, cumulatively, so the failure point is
identified exactly rather than as a blanket "it doesn't work". Then it measures
sustained frame rate with every camera streaming together, which is the number
that decides whether the design is viable.

Usage
  python3 multicam_probe.py                       # 1280x720 @ 30, MJPG
  python3 multicam_probe.py --width 640 --height 480 --fps 15
  python3 multicam_probe.py --seconds 60          # longer soak

If it fails, retry with the uvcvideo bandwidth quirk:
  sudo rmmod uvcvideo; sudo modprobe uvcvideo quirks=128
  (make permanent: echo 'options uvcvideo quirks=128' | sudo tee /etc/modprobe.d/uvcvideo.conf)
"""
import argparse
import glob
import os
import subprocess
import time

import cv2


def capture_nodes():
    """/dev/video* nodes that actually do video capture.

    The C920 exposes two nodes - the second is a metadata node that opens but
    never yields a frame, so filtering by capability avoids counting one camera
    twice.
    """
    nodes = []
    for dev in sorted(glob.glob("/dev/video*"),
                      key=lambda p: int("".join(c for c in p if c.isdigit()) or 0)):
        try:
            out = subprocess.run(["v4l2-ctl", "-d", dev, "--all"],
                                 capture_output=True, text=True, timeout=5).stdout
        except Exception:
            continue
        if "Video Capture" in out and "Format Video Capture" in out:
            name = "?"
            for line in out.splitlines():
                if "Card type" in line:
                    name = line.split(":", 1)[1].strip()
                    break
            nodes.append((dev, name))
    return nodes


def usb_topology():
    try:
        return subprocess.run(["lsusb", "-t"], capture_output=True,
                              text=True, timeout=5).stdout.rstrip()
    except Exception:
        return "(lsusb unavailable)"


def dmesg_bandwidth_errors():
    """The kernel says why an open failed; the OpenCV error does not."""
    try:
        out = subprocess.run(["dmesg", "--since", "-2min"], capture_output=True,
                             text=True, timeout=5).stdout
    except Exception:
        return []
    keys = ("No space left", "not enough bandwidth", "uvcvideo", "usb")
    return [l for l in out.splitlines()
            if any(k.lower() in l.lower() for k in keys)
            and ("bandwidth" in l.lower() or "No space" in l)]


def open_camera(dev, w, h, fps):
    """Open one camera as MJPG. Returns (cap, error_string)."""
    idx = int("".join(c for c in dev if c.isdigit()))
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None, "VideoCapture refused to open the device"
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # before size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Opening succeeds lazily; bandwidth is only reserved when streaming starts,
    # so a camera is not proven until it has actually delivered a frame.
    for _ in range(15):
        ok, frame = cap.read()
        if ok and frame is not None:
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            got = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))
            return cap, None if got == "MJPG" else f"negotiated {got}, not MJPG"
        time.sleep(0.1)
    cap.release()
    return None, "opened but produced no frames (bandwidth reservation refused?)"


def measure(caps, seconds):
    """Sustained fps per camera with all of them streaming together."""
    counts = {d: 0 for d, _ in caps}
    fails = {d: 0 for d, _ in caps}
    t0 = time.time()
    while time.time() - t0 < seconds:
        for dev, cap in caps:
            ok, frame = cap.read()
            if ok and frame is not None:
                counts[dev] += 1
            else:
                fails[dev] += 1
    elapsed = time.time() - t0
    return {d: (counts[d] / elapsed, fails[d]) for d, _ in caps}


def cpu_percent(sample=1.0):
    def snap():
        with open("/proc/stat") as f:
            v = [int(x) for x in f.readline().split()[1:]]
        return sum(v), v[3] + v[4]          # total, idle
    t1, i1 = snap()
    time.sleep(sample)
    t2, i2 = snap()
    dt, di = t2 - t1, i2 - i1
    return 100.0 * (1 - di / dt) if dt else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()

    print("=" * 68)
    print("USB topology  (all Type-A ports share Bus 01 at 480 Mbit/s)")
    print("=" * 68)
    print(usb_topology())

    quirks = "/sys/module/uvcvideo/parameters/quirks"
    print("\nuvcvideo quirks:",
          open(quirks).read().strip() if os.path.exists(quirks)
          else "(module not loaded - no camera attached yet)")

    nodes = capture_nodes()
    print(f"\ncapture-capable nodes: {len(nodes)}")
    for dev, name in nodes:
        print(f"  {dev:<14} {name}")
    if not nodes:
        print("\nNo cameras found. Plug them into Type-A ports directly, not a hub.")
        return

    print("\n" + "=" * 68)
    print(f"Opening cameras cumulatively at {args.width}x{args.height} MJPG @ {args.fps}")
    print("=" * 68)

    caps, first_failure = [], None
    for n, (dev, name) in enumerate(nodes, 1):
        cap, err = open_camera(dev, args.width, args.height, args.fps)
        if err:
            print(f"  camera {n} ({dev}): FAILED - {err}")
            if first_failure is None:
                first_failure = n
            for line in dmesg_bandwidth_errors()[-3:]:
                print(f"      kernel: {line.strip()[:110]}")
            break
        caps.append((dev, cap))
        print(f"  camera {n} ({dev}): streaming  [{n} open simultaneously]")

    if not caps:
        print("\nNot a single camera streamed. Check cabling before anything else.")
        return

    print("\n" + "=" * 68)
    print(f"Sustained rate with {len(caps)} camera(s) streaming, {args.seconds:.0f}s")
    print("=" * 68)
    before = cpu_percent()
    rates = measure(caps, args.seconds)
    after = cpu_percent()
    for dev, _ in caps:
        fps, bad = rates[dev]
        flag = "" if fps >= 12 else "   <-- below the 12 fps gate"
        print(f"  {dev:<14} {fps:6.1f} fps   {bad:4d} failed reads{flag}")
    print(f"\n  aggregate {sum(r[0] for r in rates.values()):.1f} fps"
          f"   CPU {before:.0f}% -> {after:.0f}%")

    for _, cap in caps:
        cap.release()

    print("\n" + "=" * 68)
    print("VERDICT")
    print("=" * 68)
    worst = min(r[0] for r in rates.values())
    if len(caps) >= 3 and worst >= 12:
        print(f"  PASS - {len(caps)} cameras, slowest {worst:.1f} fps.")
        print("  Phase 0 gate met. Proceed to Phase 1.")
    elif len(caps) >= 3:
        print(f"  MARGINAL - {len(caps)} cameras open but slowest is {worst:.1f} fps.")
        print("  Retry at --fps 15, and move JPEG decode to nvjpegdec.")
    else:
        print(f"  FAIL - only {len(caps)} of {len(nodes)} cameras streamed"
              f"{f' (camera {first_failure} refused)' if first_failure else ''}.")
        print("  This is the expected USB 2.0 bandwidth wall. In order:")
        print("    1. sudo rmmod uvcvideo && sudo modprobe uvcvideo quirks=128")
        print("    2. retry with --width 640 --height 480 --fps 15")
        print("    3. fall back to 2x USB + 1x CSI camera (see docs/multicam-plan.md §3.1)")


if __name__ == "__main__":
    main()
