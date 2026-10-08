#!/usr/bin/env python3
"""Why was each detection kept or dropped?

The verification filters were validated two ways: against synthetic cracks, and
against real false positives. Neither proves the thing that matters most, which
is that a REAL crack survives them. A filter that silently removes true
detections does its damage in the one place you would look to check it - the
output - so it has to be checked against a known crack, on purpose.

This prints every component that reaches the filters with the numbers each
filter used, and what happened to it. On a real crack the right outcome is not
just "kept" but "kept with margin": a crack scoring 2.1 against a valley
threshold of 2.0 passed by luck and will fail on the next surface.

It also writes an annotated image, because a table cannot show which mark on the
wall each row refers to.

    python3 explain.py --capture --out explain/
    python3 explain.py --frames scan/*.png
"""
import argparse
import glob
import os

import cv2
import numpy as np

import verify
from live import clean_mask
from measure import components as measure_components, gsd_mm_px
from tiler import plan_tiles, extract, blend_logits
from trt_infer import CrackNetTRT, logit

KEEP = (90, 220, 90)
DROP_STRAIGHT = (60, 170, 255)
DROP_FLAT = (70, 70, 255)


def annotate(frame, lab, rows):
    vis = frame.copy()
    for r in rows:
        colour = (KEEP if r["verdict"] == "kept" else
                  DROP_STRAIGHT if r["verdict"] == "too straight" else DROP_FLAT)
        sel = lab == r["label"]
        vis[sel] = (0.35 * vis[sel] + 0.65 * np.array(colour)).astype(np.uint8)
        cv2.rectangle(vis, (r["x"] - 4, r["y"] - 4),
                      (r["x"] + r["w"] + 4, r["y"] + r["h"] + 4), colour, 2)
        cv2.putText(vis, "#%d %s" % (r["idx"], r["verdict"]),
                    (r["x"], max(18, r["y"] - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, colour, 2, cv2.LINE_AA)
    legend = ["green = kept", "orange = too straight", "red = not a valley"]
    for i, text in enumerate(legend):
        cv2.putText(vis, text, (12, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (235, 235, 235), 2, cv2.LINE_AA)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="cracknet_fp16.engine")
    ap.add_argument("--frames", nargs="*")
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--out", default="explain")
    ap.add_argument("--thresh", type=float, default=0.55)
    ap.add_argument("--min-area", type=int, default=600)
    ap.add_argument("--overlap", type=float, default=0.15)
    ap.add_argument("--taper", type=float, default=0.25)
    ap.add_argument("--standoff", type=float, default=0.40)
    ap.add_argument("--mm-per-px", type=float, default=0.0)
    ap.add_argument("--hfov", type=float, default=70.42)
    ap.add_argument("--max-width-mm", type=float, default=15.0)
    ap.add_argument("--max-straightness", type=float, default=1.12)
    ap.add_argument("--min-straight-len-mm", type=float, default=40.0)
    ap.add_argument("--min-valleyness", type=float, default=2.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    paths = args.frames or sorted(glob.glob("scan/*.png"))
    if args.capture:
        from capture import SequentialRig
        rig = SequentialRig()
        res = rig.capture()
        rig.close()
        paths = []
        for role, (frame, _s) in res.items():
            p = os.path.join(args.out, "raw_%s.png" % role)
            cv2.imwrite(p, frame)
            paths.append(p)

    net = CrackNetTRT(args.engine)
    S, thr = net.size, logit(args.thresh)
    gsd = args.mm_per_px if args.mm_per_px > 0 else None

    for path in paths:
        frame = cv2.imread(path)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        g = gsd if gsd else gsd_mm_px(args.standoff, w, args.hfov)
        tiles = plan_tiles(w, h, tile=S, overlap=args.overlap)
        logits = [net.infer_logits(CrackNetTRT.preprocess(
            cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in extract(frame, tiles, S)]
        blended, _ = blend_logits(logits, tiles, w, h, S, args.taper)
        raw = (blended > thr).astype(np.uint8) * 255
        cleaned = clean_mask(raw, args.min_area,
                             max_halfwidth=(args.max_width_mm / 2.0) / g)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        comps = verify.report(cleaned, gray,
                              min_len_px=args.min_straight_len_mm / g)
        geo = {c["label"]: c for c in comps}
        sizes = measure_components(cleaned, g, min_area=1)

        n, lab, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        rows = []
        for idx, c in enumerate(sorted(comps, key=lambda r: -r["area_px"]), 1):
            straight = c["straightness"]
            if (c["long_enough"] and straight is not None
                    and straight < args.max_straightness):
                vd = "too straight"
            elif c["valleyness"] < args.min_valleyness:
                vd = "not a valley"
            else:
                vd = "kept"
            rows.append(dict(c, idx=idx, verdict=vd))

        name = os.path.basename(path)
        print("\n%s   %d tiles, %.3f mm/px, raw %.2f %% -> cleaned %.2f %%"
              % (name, len(tiles), g, 100 * (raw > 0).mean(),
                 100 * (cleaned > 0).mean()))
        if not rows:
            print("  nothing reached the filters")
            continue
        print("  %-3s %9s %9s %9s %9s %10s   %s"
              % ("#", "area", "len mm", "width mm", "straight", "valley", "verdict"))
        for r in rows:
            print("  %-3d %9d %9.1f %9.2f %9s %10.2f   %s"
                  % (r["idx"], r["area_px"], r["length_px"] * g,
                     r["width_px"] * g,
                     "-" if r["straightness"] is None else "%.2f" % r["straightness"],
                     r["valleyness"], r["verdict"]))

        kept = [r for r in rows if r["verdict"] == "kept"]
        if kept:
            vmin = min(r["valleyness"] for r in kept)
            print("  kept %d of %d; tightest valley margin %.2f against a "
                  "threshold of %.2f" % (len(kept), len(rows), vmin,
                                         args.min_valleyness))
        else:
            print("  kept 0 of %d" % len(rows))

        out = os.path.join(args.out, "explain_" + name.rsplit(".", 1)[0] + ".jpg")
        cv2.imwrite(out, annotate(frame, lab, rows))
        print("  wrote %s" % out)


if __name__ == "__main__":
    main()
