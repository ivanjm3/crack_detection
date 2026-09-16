#!/usr/bin/env python3
"""Grab N frames from the C920 as parity/tuning inputs (setup.md 5).

Frames are centre-cropped square, the same geometry live.py feeds the model.
"""
import argparse, os, time
import cv2
from live import open_c920, center_square

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="val_samples")
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--cam", type=int, default=0)
ap.add_argument("--interval", type=float, default=0.3)
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
cap = open_c920(args.cam)
for _ in range(10):        # let exposure settle
    cap.read()

n = 0
while n < args.n:
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("no frame from camera")
    crop, _ = center_square(frame)
    path = os.path.join(args.out, f"cam_{n:03d}.jpg")
    cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    n += 1
    time.sleep(args.interval)

cap.release()
print(f"wrote {n} frames to {args.out}/ ({crop.shape[1]}x{crop.shape[0]})")
