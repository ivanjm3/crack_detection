#!/usr/bin/env python3
"""Tests for tiler.py. No camera, no GPU, no model - pure geometry."""
import sys

import cv2
import numpy as np

from tiler import (plan_tiles, extract, stitch_masks, coverage_map,
                   describe, tile_window, blend_logits)

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

print("\nblending: overlapping tiles are averaged, not unioned")
win = tile_window(64, taper=0.25)
check("window carries full weight at the centre", abs(win[32, 32] - 1.0) < 1e-6)
check("window is strongly suppressed at the edge",
      0 < win[32, 0] < 0.05 * win[32, 32],
      f"edge weight {win[32, 0]:.4f}")
check("window rises monotonically from edge to centre",
      bool(np.all(np.diff(win[32, :32]) >= -1e-7)))

bad_taper = []
for bad in (-0.1, 0.5, 1.0):
    try:
        tile_window(64, taper=bad)
        bad_taper.append(bad)
    except ValueError:
        pass
check("an impossible taper is rejected", not bad_taper, f"accepted {bad_taper}")

btiles = plan_tiles(300, 200, tile=128, overlap=0.2)
flat = [np.full((128, 128), 2.5, np.float32) for _ in btiles]
out, wsum = blend_logits(flat, btiles, 300, 200, tile=128)
check("a constant survives the weighted average everywhere",
      bool(np.allclose(out, 2.5, atol=1e-4)),
      f"range {out.min():.4f}..{out.max():.4f}")
check("every pixel gets some weight", bool(np.all(wsum > 0)))

# The reason this replaced the union: one tile fires hard at its own edge - a
# border artefact - while the neighbour sees the same ground in its interior and
# says nothing. The union keeps the artefact because something fired.
etiles = plan_tiles(300, 128, tile=128, overlap=0.4)
spike = [np.full((128, 128), -4.0, np.float32) for _ in etiles]
spike[0][:, -3:] = 8.0
blended, _ = blend_logits(spike, etiles, 300, 128, tile=128)
unioned = stitch_masks([(lg > 0).astype(np.uint8) * 255 for lg in spike],
                       etiles, 300, 128, tile=128)
col = etiles[0][0] + 128 - 2
check("a tile-edge spike does not survive the blend", blended[64, col] < 0.0,
      f"blended logit {blended[64, col]:.2f} at the seam")
check("the union DOES keep it, which is the bug being fixed",
      unioned[64, col] > 0)

agree = [np.full((128, 128), -4.0, np.float32) for _ in etiles]
for lg in agree:
    lg[60:68, :] = 6.0
blended2, _ = blend_logits(agree, etiles, 300, 128, tile=128)
check("a detection every tile agrees on is kept", blended2[64, 150] > 0.0,
      f"blended logit {blended2[64, 150]:.2f}")

try:
    blend_logits([np.zeros((128, 128), np.float32)], btiles, 300, 200, tile=128)
    check("blend rejects a logit/tile count mismatch", False)
except ValueError:
    check("blend rejects a logit/tile count mismatch", True)


print("\ncost of each plan")
for w, h in ((1280, 720), (1920, 1080)):
    print("  " + describe(w, h))

print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} passed")
sys.exit(1 if FAIL else 0)

