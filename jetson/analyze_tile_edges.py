#!/usr/bin/env python3
"""Does the model fire more near tile borders than in tile interiors?

A convolutional network has no information beyond the edge of its input, so the
first and last rows of a tile are predicted from padded context that does not
exist in the scene. The standard symptom is a response that tracks tile
boundaries rather than the surface, and the standard mitigation is to average
overlapping predictions instead of taking their union - the union keeps the
strongest artefact from every tile, which is the opposite of suppressing it.

tiler.py currently thresholds each tile and ORs the overlaps. That choice was
made to favour recall on cracks truncated at a seam. This measures what it costs,
by comparing detection density against distance from the nearest tile border.

A flat profile means the concern does not apply here and the union is fine. A
spike in the first few pixels means tile borders are generating detections, and
every one of them is a false positive by construction.

    python3 analyze_tile_edges.py --frames scan/*.png
    python3 analyze_tile_edges.py --capture      # grab fresh frames first
"""
import argparse
import glob

import cv2
import numpy as np

from live import clean_mask
import verify
from tiler import plan_tiles, extract, stitch_masks, blend_logits
from trt_infer import CrackNetTRT, logit


def border_distance(tiles, w, h, tile):
    """For each pixel, distance to the nearest border of any tile covering it.

    Taken as the MAXIMUM over covering tiles: a pixel in the overlap sits at the
    edge of one tile and comfortably inside its neighbour, and it is the
    interior view that the union keeps if either fires. Scoring it by its worst
    tile would blame the overlap for an artefact the union may have already
    rejected.
    """
    best = np.zeros((h, w), np.int32)
    for x, y in tiles:
        ys, xs = np.mgrid[0:tile, 0:tile]
        d = np.minimum.reduce([xs, ys, tile - 1 - xs, tile - 1 - ys])
        region = best[y:y + tile, x:x + tile]
        np.maximum(region, d, out=region)
    return best


def profile(mask, dist, edges):
    """Detection rate in each band of border distance."""
    out = []
    for lo, hi in zip(edges, edges[1:]):
        sel = (dist >= lo) & (dist < hi)
        n = int(sel.sum())
        if n == 0:
            continue
        out.append((lo, hi, n, float((mask[sel] > 0).mean())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="cracknet_fp16.engine")
    ap.add_argument("--frames", nargs="*", default=None)
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--thresh", type=float, default=0.55)
    ap.add_argument("--min-area", type=int, default=600)
    ap.add_argument("--overlap", type=float, default=0.15)
    ap.add_argument("--blend", action="store_true",
                    help="use the averaged blend instead of the union")
    args = ap.parse_args()

    paths = args.frames or sorted(glob.glob("scan/*.png"))
    if args.capture or not paths:
        from capture import SequentialRig
        rig = SequentialRig()
        res = rig.capture()
        rig.close()
        paths = []
        for role, (frame, _s) in res.items():
            p = "scan/edge_%s.png" % role
            cv2.imwrite(p, frame)
            paths.append(p)
        print("captured %d frame(s)\n" % len(paths))

    net = CrackNetTRT(args.engine)
    S, thr = net.size, logit(args.thresh)

    agg_raw, agg_clean, agg_n = {}, {}, {}
    edges = [0, 2, 4, 8, 16, 32, 64, 128, 256]

    for path in paths:
        frame = cv2.imread(path)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        tiles = plan_tiles(w, h, tile=S, overlap=args.overlap)
        logits = []
        for crop in extract(frame, tiles, S):
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            logits.append(net.infer_logits(CrackNetTRT.preprocess(rgb)))

        union = stitch_masks([(lg > thr).astype(np.uint8) * 255 for lg in logits],
                             tiles, w, h, S)
        blended, _ = blend_logits(logits, tiles, w, h, S)
        raw = (blended > thr).astype(np.uint8) * 255 if args.blend else union
        cleaned = clean_mask(raw, args.min_area)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        verified, dropped = verify.apply(cleaned, gray)
        dist = border_distance(tiles, w, h, S)

        print("%s  %d tiles  union %.3f %%  blended %.3f %%  "
              "cleaned %.3f %%  verified %.3f %%  (dropped %d straight, %d flat)"
              % (path.split("/")[-1], len(tiles),
                 100 * (union > 0).mean(), 100 * ((blended > thr)).mean(),
                 100 * (cleaned > 0).mean(), 100 * (verified > 0).mean(),
                 dropped["straight"], dropped["flat"]))
        cleaned = verified
        for lo, hi, n, rate in profile(raw, dist, edges):
            agg_raw[(lo, hi)] = agg_raw.get((lo, hi), 0.0) + rate * n
            agg_n[(lo, hi)] = agg_n.get((lo, hi), 0) + n
        for lo, hi, n, rate in profile(cleaned, dist, edges):
            agg_clean[(lo, hi)] = agg_clean.get((lo, hi), 0.0) + rate * n

    print("\ndetection rate by distance from the nearest tile border")
    print("%-14s %12s %10s %10s" % ("band (px)", "pixels", "raw %", "final %"))
    base_raw = base_clean = None
    for key in sorted(agg_n):
        lo, hi = key
        n = agg_n[key]
        r = 100 * agg_raw[key] / n
        c = 100 * agg_clean.get(key, 0.0) / n
        if lo >= 128:
            base_raw, base_clean = r, c
        print("%-14s %12d %10.3f %10.3f" % ("%d-%d" % (lo, hi), n, r, c))

    if base_raw:
        first = sorted(agg_n)[0]
        r0 = 100 * agg_raw[first] / agg_n[first]
        c0 = 100 * agg_clean.get(first, 0.0) / agg_n[first]
        print("\nedge band vs interior:  raw %.2fx   cleaned %.2fx"
              % (r0 / max(base_raw, 1e-9), c0 / max(base_clean, 1e-9)))
        print("A ratio near 1.0 means tile borders are not generating "
              "detections.\nA ratio well above 1.0 means they are, and every "
              "one is a false positive.")


if __name__ == "__main__":
    main()
