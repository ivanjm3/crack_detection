#!/usr/bin/env python3
"""Does the C920 show the same field of view at 720p and 1080p?

Every resolution figure in docs/multicam-plan.md §4A.4 assumes it does - that 1080p
reads the whole sensor rather than cropping a window out of it. If that assumption is
wrong, the 1080p ground-sampling-distance numbers are wrong with it, and the standoff
chosen from them is wrong too.

No ruler and no human judgement needed. Capture the same static scene at both
resolutions, then search for the crop factor f that best aligns them:

    take the central f of the 1080p frame -> resize to 720p size -> compare

  f ~ 1.00  both resolutions see the same scene       -> §4A.4 holds
  f <  1.00  1080p is cropped; it sees 1/f LESS width -> rescale the GSD table

Comparison is zero-mean normalised correlation, so the two captures do not need
matching exposure.

Usage:   python3 check_fov_parity.py [--cam 0] [--frames 8]
Point the camera at something static and textured - a wall with marks, printed
text, anything non-uniform. A blank wall carries no signal to align on.
"""
import argparse
import time

import cv2
import numpy as np


def grab(dev, w, h, n_avg, settle=12):
    """Open at one resolution and return an averaged grayscale frame."""
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open /dev/video{dev}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # before size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    for _ in range(settle):          # let auto-exposure and the stream settle
        cap.read()
        time.sleep(0.03)

    acc, got = None, 0
    for _ in range(n_avg * 3):
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        acc = g if acc is None else acc + g
        got += 1
        if got >= n_avg:
            break
    actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    if not got:
        raise RuntimeError(f"no frames at {w}x{h}")
    return acc / got, actual


def zncc(a, b):
    """Zero-mean normalised cross-correlation: immune to exposure differences."""
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--frames", type=int, default=8)
    args = ap.parse_args()

    print("Capturing the same scene at both resolutions - keep everything still.\n")
    lo, lo_actual = grab(args.cam, 1280, 720, args.frames)
    hi, hi_actual = grab(args.cam, 1920, 1080, args.frames)
    print(f"  720p  negotiated {lo_actual[0]}x{lo_actual[1]}")
    print(f"  1080p negotiated {hi_actual[0]}x{hi_actual[1]}")
    if hi_actual == lo_actual:
        print("\n  1080p did not negotiate - the camera stayed at 720p. "
              "Cannot compare; check --list-formats-ext.")
        return

    H, W = lo.shape
    # Mild blur so the search tracks structure rather than sensor noise.
    lo_b = cv2.GaussianBlur(lo, (5, 5), 0)

    best = (None, -2.0)
    print(f"\n{'crop f':>8} {'1080p sees':>12} {'ZNCC':>8}")
    results = []
    for f in np.arange(0.60, 1.001, 0.02):
        ch, cw = int(hi.shape[0] * f), int(hi.shape[1] * f)
        y0, x0 = (hi.shape[0] - ch) // 2, (hi.shape[1] - cw) // 2
        sub = hi[y0:y0 + ch, x0:x0 + cw]
        sub = cv2.resize(sub, (W, H), interpolation=cv2.INTER_AREA)
        s = zncc(lo_b, cv2.GaussianBlur(sub, (5, 5), 0))
        results.append((f, s))
        if s > best[1]:
            best = (f, s)
    for f, s in results:
        if abs(f - best[0]) < 0.07 or abs(f - 1.0) < 0.001:
            bar = "#" * int(max(0, s) * 40)
            print(f"{f:>8.2f} {1/f:>11.2f}x {s:>8.3f}  {bar}")

    f, score = best
    print(f"\nbest crop factor f = {f:.2f}   (ZNCC {score:.3f})")

    print("\n" + "=" * 60)
    if score < 0.35:
        print("INCONCLUSIVE - the scene has too little structure to align on,")
        print("or something moved. Point at textured detail and re-run.")
    elif f >= 0.96:
        print("SAME FIELD OF VIEW.")
        print("1080p reads the full sensor. docs/multicam-plan.md §4A.4 holds as")
        print("written: at 0.4 m standoff, 0.29 mm/px and ~0.58 mm minimum crack.")
    else:
        print(f"1080p IS CROPPED - it sees {1/f:.2f}x LESS than 720p.")
        print("The §4A.4 table must be rescaled: the 1080p swath shrinks by this")
        print(f"factor, while GSD improves by {1/f:.2f}x beyond what is tabulated.")
        print("Net: finer detection, narrower coverage. Re-pick the standoff.")
    print("=" * 60)


if __name__ == "__main__":
    main()
