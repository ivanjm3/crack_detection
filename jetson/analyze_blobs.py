#!/usr/bin/env python3
"""Measure blob geometry to justify the shape-filter thresholds.

Runs the engine over a directory of frames and prints, per component, the
metrics clean_mask() filters on - plus a synthetic thin-crack control, so we
can see the filter keeps filaments while rejecting fat blobs.
"""
import argparse, glob, os
import cv2, numpy as np
from trt_infer import CrackNetTRT
from live import clean_mask

ap = argparse.ArgumentParser()
ap.add_argument("--engine", default="cracknet_fp16.engine")
ap.add_argument("--dir", default="val_samples")
ap.add_argument("--thresh", type=float, default=0.55)
ap.add_argument("--min-area", type=int, default=300)
args = ap.parse_args()

net = CrackNetTRT(args.engine)
S = net.size


def metrics(mask):
    """Per-component (area, mean half-width, solidity, area fraction)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < args.min_area:
            continue
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        sub = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        perim = sum(cv2.arcLength(c, True) for c in cnts)
        hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
        ha = cv2.contourArea(hull)
        out.append((area, area / max(perim, 1.0),
                    area / ha if ha > 0 else 1.0, area / mask.size))
    return out


print("=== synthetic control: a thin crack-like filament ===")
synth = np.zeros((S, S), np.uint8)
pts = np.array([[60, 40], [180, 150], [150, 300], [260, 420], [400, 470]])
cv2.polylines(synth, [pts], False, 255, 3)
cv2.polylines(synth, [np.array([[180, 150], [300, 180], [380, 140]])], False, 255, 2)
for a, hw, sol, fr in metrics(synth):
    print(f"  area={a:7d}  half-width={hw:6.2f}px  solidity={sol:5.3f}  frac={100*fr:6.3f}%")
kept = clean_mask(synth, args.min_area)
print(f"  -> survives clean_mask: {(kept>0).sum()}/{(synth>0).sum()} px "
      f"({'KEPT' if (kept>0).sum() > 0 else 'REJECTED - filter is too strict!'})")

print("\n=== real frames from %s/ ===" % args.dir)
paths = sorted(glob.glob(os.path.join(args.dir, "*.jpg")))
allm, cov_before, cov_after = [], [], []
for p in paths:
    img = cv2.imread(p)
    if img is None:
        continue
    rgb = cv2.cvtColor(cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2RGB)
    prob = net.infer(CrackNetTRT.preprocess(rgb))
    raw = (prob > args.thresh).astype(np.uint8) * 255
    old = clean_mask(raw, args.min_area, shape_filter=False)
    new = clean_mask(raw, args.min_area, shape_filter=True)
    allm += metrics(raw)
    cov_before.append((old > 0).mean())
    cov_after.append((new > 0).mean())

if allm:
    a = np.array(allm)
    print(f"  {len(paths)} frames, {len(allm)} components above min-area")
    for name, col in (("area (px)", 0), ("half-width (px)", 1),
                      ("solidity", 2), ("frame fraction", 3)):
        v = a[:, col]
        print(f"    {name:18s} min={v.min():9.3f}  median={np.median(v):9.3f}  max={v.max():9.3f}")

print(f"\n  coverage area-filter only : mean {100*np.mean(cov_before):6.3f}%  max {100*np.max(cov_before):6.3f}%")
print(f"  coverage + shape filter   : mean {100*np.mean(cov_after):6.3f}%  max {100*np.max(cov_after):6.3f}%")
red = 1 - np.mean(cov_after) / max(np.mean(cov_before), 1e-9)
print(f"  false-positive coverage removed: {100*red:.1f}%")
