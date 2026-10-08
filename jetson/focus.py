#!/usr/bin/env python3
"""Find the focus setting that is actually sharp at the working distance.

camera_setup.sh pins focus_absolute=30 with autofocus off. On the C920 that
control runs 0 (far) to 255 (close), so 30 is very nearly infinity - correct for
a camera looking down a corridor and wrong for one held 20 cm from a surface.

This matters more here than it would almost anywhere else. The model responds to
thin dark lines, and defocus blur is a low-pass filter: it spreads a 3 px crack
across 15 px and raises its minimum towards the surrounding grey. The crack does
not get harder to see, it stops being present in the image. No threshold, filter
or training fixes that, and every accuracy measurement taken out of focus is
measuring the blur instead of the detector.

Sharpness is the variance of the Laplacian over the centre of the frame - the
standard autofocus metric. It rewards high spatial frequency, which is what
focus restores, and the centre crop keeps a bright window or a dark edge at the
frame boundary from dominating the number.

    python3 focus.py sweep                 # every camera, full range
    python3 focus.py sweep --camera 1
    python3 focus.py apply                 # sweep, then save focus.json
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from capture import Camera, discover, load_rig

FOCUS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "focus.json")


def sharpness(frame, crop=0.5):
    """Variance of the Laplacian over the central `crop` fraction."""
    h, w = frame.shape[:2]
    ch, cw = int(h * crop), int(w * crop)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    gray = cv2.cvtColor(frame[y0:y0 + ch, x0:x0 + cw], cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def sweep_one(cam, lo=0, hi=255, step=15, settle=0.45):
    """Sharpness against focus_absolute. Returns [(focus, sharpness), ...].

    The lens is a physical mechanism and takes time to travel, so each setting
    gets a settle period and the frames from during the move are discarded. A
    sweep without that measures where the lens was, not where it was sent.
    """
    out = []
    for value in range(lo, hi + 1, step):
        cam.cap.set(cv2.CAP_PROP_FOCUS, value)
        t0 = time.time()
        while time.time() - t0 < settle:
            cam.cap.read()
        best = 0.0
        for _ in range(3):
            ok, frame = cam.cap.read()
            if ok and frame is not None:
                best = max(best, sharpness(frame))
        out.append((value, best))
    return out


def refine(cam, coarse, step=5, settle=0.45):
    """Re-sweep finely around the coarse peak.

    The coarse step is large enough to straddle the true peak, and focus error
    is not symmetric in its effect - being slightly close is not the same as
    slightly far - so the peak is located rather than interpolated.
    """
    peak = max(coarse, key=lambda r: r[1])[0]
    lo = max(0, peak - 20)
    hi = min(255, peak + 20)
    return sweep_one(cam, lo, hi, step, settle)


def run(role, info, args):
    cam = Camera(role, info, args.width, args.height)
    cam.open()
    try:
        # Autofocus must be off or the camera will quietly overrule every value
        # written here, and the sweep would measure the camera's opinion rather
        # than the setting.
        cam.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        time.sleep(0.3)
        coarse = sweep_one(cam, step=args.step, settle=args.settle)
        fine = refine(cam, coarse, settle=args.settle) if args.refine else []
        allr = {v: s for v, s in coarse}
        allr.update({v: s for v, s in fine})
        rows = sorted(allr.items())
        best_focus, best_sharp = max(rows, key=lambda r: r[1])

        print("\n%s  (%s)" % (role, info["dev"]))
        print("  %-8s %12s" % ("focus", "sharpness"))
        for v, s in rows:
            bar = "#" * int(40 * s / max(best_sharp, 1e-9))
            mark = "  <- best" if v == best_focus else ""
            print("  %-8d %12.1f  %s%s" % (v, s, bar, mark))
        current = sharpness_at(cam, 30, args.settle)
        print("  best focus %d -> %.1f sharpness" % (best_focus, best_sharp))
        print("  current setting 30 -> %.1f  (%.1fx worse)"
              % (current, best_sharp / max(current, 1e-9)))
        return best_focus, best_sharp, current
    finally:
        cam.close()


def sharpness_at(cam, value, settle=0.45):
    cam.cap.set(cv2.CAP_PROP_FOCUS, value)
    t0 = time.time()
    while time.time() - t0 < settle:
        cam.cap.read()
    best = 0.0
    for _ in range(3):
        ok, frame = cam.cap.read()
        if ok and frame is not None:
            best = max(best, sharpness(frame))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("sweep", "apply"))
    ap.add_argument("--camera", type=int, default=None,
                    help="index into the discovered list; default all")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--step", type=int, default=15)
    ap.add_argument("--settle", type=float, default=0.45)
    ap.add_argument("--no-refine", dest="refine", action="store_false")
    args = ap.parse_args()

    try:
        rig = load_rig()
    except RuntimeError as exc:
        print(exc)
        return
    items = sorted(rig.items())
    if args.camera is not None:
        cams = discover()
        items = [("camera%d" % args.camera, cams[args.camera])]

    result = {}
    for role, info in items:
        best, sharp, current = run(role, info, args)
        result[role] = {"focus": best, "sharpness": sharp,
                        "sharpness_at_30": current, "port": info["port"]}

    if args.mode == "apply":
        with open(FOCUS_JSON, "w") as f:
            json.dump(result, f, indent=2)
        print("\nwrote %s" % FOCUS_JSON)
        print("capture.py applies these on open; re-run after moving the rig, "
              "because focus is a property of the DISTANCE, not the camera.")


if __name__ == "__main__":
    main()
