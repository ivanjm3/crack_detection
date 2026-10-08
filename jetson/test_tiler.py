#!/usr/bin/env python3
"""Tests for tiler.py. No camera, no GPU, no model - pure geometry."""
import sys

import cv2
import numpy as np

from tiler import plan_tiles, extract, stitch_masks, coverage_map, describe

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")


print("coverage: every pixel must be seen by at least one tile")
for w, h in ((1280, 720), (1920, 1080), (640, 512), (1024, 768)):
    tiles = plan_tiles(w, h)
    cov = coverage_map(tiles, w, h)
    check(f"{w}x{h} fully covered", cov.min() >= 1,
          f"{len(tiles)} tiles, min coverage {cov.min()}, max {cov.max()}")

print("\nbounds: no tile may hang off the frame")
for w, h in ((1280, 720), (1920, 1080), (1023, 513)):
    tiles = plan_tiles(w, h)
    ok = all(0 <= x <= w - 512 and 0 <= y <= h - 512 for x, y in tiles)
    check(f"{w}x{h} all tiles inside", ok)

print("\nno resampling: extracted tiles are exact sub-arrays of the frame")
rng = np.random.default_rng(0)
frame = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
tiles = plan_tiles(1280, 720)
crops = extract(frame, tiles)
exact = all(c.shape == (512, 512, 3) and
            np.array_equal(c, frame[y:y + 512, x:x + 512])
            for c, (x, y) in zip(crops, tiles))
check("tiles are byte-identical sub-arrays", exact, f"{len(crops)} tiles")

print("\nround trip: a crack drawn across the frame survives tile -> stitch")
truth = np.zeros((720, 1280), np.uint8)
cv2.line(truth, (40, 60), (1240, 700), 255, 3)          # spans the whole frame
cv2.line(truth, (900, 80), (300, 690), 255, 2)          # crosses the first
tiles = plan_tiles(1280, 720)
per_tile = extract(truth, tiles)                        # stand in for model output
rebuilt = stitch_masks(per_tile, tiles, 1280, 720)
check("stitched mask equals the original", np.array_equal(rebuilt, truth),
      f"{int((truth > 0).sum())} px in, {int((rebuilt > 0).sum())} px out")

print("\nunion on overlap: a detection in one tile is not erased by its neighbour")
blank = [np.zeros((512, 512), np.uint8) for _ in tiles]
blank[0] = np.full((512, 512), 255, np.uint8)           # only the first tile fires
merged = stitch_masks(blank, tiles, 1280, 720)
x0, y0 = tiles[0]
check("lone detection survives the merge",
      merged[y0:y0 + 512, x0:x0 + 512].min() == 255)

print("\nconnectivity: a crack crossing tile seams stays ONE component")
n, _, stats, _ = cv2.connectedComponentsWithStats(rebuilt, connectivity=8)
truth_n, _, _, _ = cv2.connectedComponentsWithStats(truth, connectivity=8)
check("component count matches the original", n == truth_n,
      f"{n - 1} component(s) found, {truth_n - 1} expected")

print("\nerrors: bad input is rejected rather than silently mishandled")
try:
    plan_tiles(400, 300)
    check("frame smaller than tile raises", False)
except ValueError:
    check("frame smaller than tile raises", True)
try:
    stitch_masks([np.zeros((512, 512), np.uint8)], tiles, 1280, 720)
    check("mask/tile count mismatch raises", False)
except ValueError:
    check("mask/tile count mismatch raises", True)

print("\ncost of each plan")
for w, h in ((1280, 720), (1920, 1080)):
    print("  " + describe(w, h))

print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} passed")
sys.exit(1 if FAIL else 0)
