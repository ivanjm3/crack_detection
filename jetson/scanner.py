#!/usr/bin/env python3
"""One scan position, end to end: three cameras -> tiles -> masks -> millimetres.

This is the piece that joins the parts already built and tested separately:

    capture.py   sequential 1080p capture from three cameras   (USB bandwidth)
    tiler.py     native-resolution tiling and mask reassembly  (resolution)
    trt_infer.py the FP16 engine                               (inference)
    live.py      clean_mask()                                  (false positives)
    measure.py   component geometry in millimetres             (reporting)

Each camera is processed INDEPENDENTLY and completely: tiled, inferred,
stitched, cleaned and measured on its own frame. Nothing is stitched across
cameras before inference, for the reason in docs/multicam-plan.md §1.2 - a
blended seam is a long thin linear feature, which is precisely what the shape
filter is built to keep, so a cross-camera mosaic would manufacture the exact
artefact the detector is most likely to call a crack. The three-camera composite
this produces is for human eyes only, and says so on the image itself.

    python3 scanner.py --once            one scan position, print the numbers
    python3 scanner.py --once --save out/
"""
import argparse
import os
import time

import cv2
import numpy as np

import measure
from capture import SequentialRig
from live import clean_mask
import verify
from tiler import plan_tiles, extract, blend_logits
from trt_infer import CrackNetTRT, logit

# min_area is in PIXELS, and native tiling changed what a pixel is.
#
# The single-camera pipeline centre-cropped 720x720 and downscaled it to 512, so
# its 300 px threshold was measured on pixels 1/0.711 larger than native ones.
# The same physical speck covers (1/0.711)^2 = 1.98x more pixels at native
# resolution, so carrying 300 across unchanged would reject roughly half the
# physical area it used to - silently tightening the filter while appearing to
# leave it alone.
#
# This is a first-order correction, not a tuning. §7 of the plan still wants
# these thresholds set against real pavement.
MIN_AREA_NATIVE = 600


class Scanner:
    """Owns the cameras and the engine; produces one scan position per call."""

    def __init__(self, args):
        self.a = args
        self.net = CrackNetTRT(args.engine)
        self.size = self.net.size
        # Threshold in LOGIT space so the sigmoid never has to run. It is ~2.5 ms
        # a call, which was noise for one frame a time and is 22 % of inference
        # once a scan runs 15 tiles per camera x 3 cameras.
        self.thr = logit(args.thresh)
        self.rig = SequentialRig(args.width, args.height, args.average,
                                 verbose=args.verbose)
        self.tiles = plan_tiles(args.width, args.height, tile=self.size,
                                overlap=args.overlap)
        # mm/px is the scale factor on EVERY width this system reports, so it
        # is linear in the standoff: assume 0.40 m when the camera is really at
        # 0.15 m and every crack is reported 2.7x too wide. The field-of-view
        # calculation is only ever as good as the distance it is given, so a
        # directly measured figure overrides it.
        self.gsd = (args.mm_per_px if args.mm_per_px > 0 else
                    measure.gsd_mm_px(args.standoff, args.width, args.hfov))
        # The shape filter's thickness limit becomes PHYSICAL here, which is the
        # payoff docs/multicam-plan.md §2.2 promised: once a pixel has a known
        # size on the ground, "too thick to be a crack" can be stated in
        # millimetres instead of pixels.
        #
        # It also fixes a bug this file would otherwise have introduced. Native
        # tiling changed what a pixel is, and max_halfwidth is in pixels. The
        # old pipeline worked on model pixels of ~0.62 mm, so its 12 px limit
        # rejected anything wider than ~15 mm; carried across to 0.294 mm native
        # pixels, the same 12 would reject anything wider than 7 mm - silently
        # halving the widest crack the system can report, while looking like an
        # unchanged setting. min_area above has the same hazard and the same
        # cause.
        self.max_halfwidth_px = (args.max_width_mm / 2.0) / self.gsd
        self.scan_index = 0

    def describe(self):
        return ("%d camera(s), %dx%d, %d tiles/camera, %.3f mm/px at %.2f m, "
                "reject wider than %.1f mm (%.0f px)%s"
                % (len(self.rig.cams), self.a.width, self.a.height,
                   len(self.tiles), self.gsd, self.a.standoff,
                   self.a.max_width_mm, 2 * self.max_halfwidth_px,
                   "" if self.a.no_verify else
                   ", straightness < %.2f, valley < %.1f"
                   % (self.a.max_straightness, self.a.min_valleyness)))

    def infer_frame(self, frame):
        """Full-frame mask for one camera: tile, infer, blend, clean, verify."""
        crops = extract(frame, self.tiles, self.size)
        logits = []
        t0 = time.perf_counter()
        for crop in crops:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            logits.append(self.net.infer_logits(CrackNetTRT.preprocess(rgb)))
        infer_s = time.perf_counter() - t0

        # Overlapping tiles are AVERAGED, then thresholded once - not
        # thresholded per tile and unioned. The union marked a pixel if any tile
        # fired on it, so it kept the strongest border artefact from every tile
        # covering that pixel; measured on this rig, detections within 2 px of a
        # tile border ran 2.2x the interior rate raw and 11.3x after shape
        # filtering (analyze_tile_edges.py). Averaging lets a tile that can see
        # the context outvote one that is guessing from padding.
        blended, _w = blend_logits(logits, self.tiles, frame.shape[1],
                                   frame.shape[0], self.size, self.a.taper)
        raw = (blended > self.thr).astype(np.uint8) * 255

        # Shape filtering runs ONCE on the reassembled mask, never per tile. A
        # crack crossing a tile boundary is truncated in both tiles, and a
        # truncated fragment is short, stubby and solid - it would be thrown out
        # by the very filter meant to protect against blobs.
        clean = clean_mask(raw, self.a.min_area,
                           max_area_frac=self.a.max_area_frac,
                           max_halfwidth=self.max_halfwidth_px,
                           max_solidity=self.a.max_solidity,
                           shape_filter=not self.a.no_shape_filter)

        dropped = {}
        if not self.a.no_verify:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clean, dropped = verify.apply(
                clean, gray,
                max_straightness=self.a.max_straightness,
                min_valleyness=self.a.min_valleyness,
                min_len_px=self.a.min_straight_len_mm / self.gsd)
        return clean, infer_s, dropped

    def scan(self):
        """Capture and analyse one scan position. Returns (cameras, totals)."""
        self.scan_index += 1
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        captured = self.rig.capture()
        capture_s = time.perf_counter() - t0

        cams, infer_total = [], 0.0
        for role in ("left", "top", "right"):
            if role not in captured:
                continue
            frame, cstats = captured[role]
            mask, infer_s, dropped = self.infer_frame(frame)
            infer_total += infer_s
            m = measure.summarise(mask, self.gsd, min_area=1)
            cams.append({
                "name": role.upper(),
                "ok": True,
                "frame": frame,
                "mask": mask,
                "coverage": m["coverage"],
                "components": m["components"],
                "widest_mm": m["widest_mm"],
                "longest_mm": m["longest_mm"],
                "crack": m["coverage"] > self.a.alert_frac,
                "exposure": cstats["exposure"],
                "gain": cstats["gain"],
                "median_luma": cstats["median_luma"],
                "converged": cstats["converged"],
                "infer_ms": infer_s * 1000.0,
                "dropped_straight": dropped.get("straight", 0),
                "dropped_flat": dropped.get("flat", 0),
            })

        widths = [c["widest_mm"] for c in cams if c["widest_mm"] is not None]
        totals = {
            # Coverage is averaged over cameras, not pooled over pixels. The
            # three cameras see three different patches of ground, so a pooled
            # figure would let one busy camera be diluted by two empty ones in a
            # way that depends on frame size rather than on the surface.
            "coverage": float(np.mean([c["coverage"] for c in cams])) if cams else 0.0,
            "components": sum(c["components"] for c in cams),
            "widest_mm": max(widths) if widths else None,
            "dropped_straight": sum(c["dropped_straight"] for c in cams),
            "dropped_flat": sum(c["dropped_flat"] for c in cams),
            "capture_s": capture_s,
            "infer_s": infer_total,
            "cycle_s": time.perf_counter() - t_start,
            "scan_index": self.scan_index,
        }
        totals["crack"] = any(c["crack"] for c in cams)
        return cams, totals

    def close(self):
        self.rig.close()


def overlay(bgr, mask):
    """Red fill plus a yellow outline, the same treatment live.py uses."""
    vis = bgr.copy()
    hit = mask > 0
    if hit.any():
        vis[hit] = (0.4 * vis[hit] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 255, 255), 1)
    return vis


def composite(cams, width=1440):
    """Side-by-side view of the three cameras, labelled as display-only.

    The label is not decoration. Someone looking at three frames butted together
    will reasonably assume they were stitched and fed to the model as one image,
    which is exactly the design this project rejected and documented at length.
    """
    if not cams:
        return np.zeros((240, width, 3), np.uint8)
    tiles = []
    for c in cams:
        vis = overlay(c["frame"], c["mask"])
        cv2.putText(vis, c["name"], (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (235, 235, 235), 2, cv2.LINE_AA)
        tiles.append(vis)
    strip = np.hstack(tiles)
    h = max(1, int(strip.shape[0] * width / strip.shape[1]))
    strip = cv2.resize(strip, (width, h), interpolation=cv2.INTER_AREA)
    cv2.putText(strip, "DISPLAY COMPOSITE - detection is per-camera",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (205, 205, 205), 1, cv2.LINE_AA)
    return strip


def add_arguments(ap):
    ap.add_argument("--engine", default="cracknet_fp16.engine")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--average", type=int, default=4)
    ap.add_argument("--overlap", type=float, default=0.15)
    ap.add_argument("--thresh", type=float, default=0.55)
    ap.add_argument("--min-area", type=int, default=MIN_AREA_NATIVE)
    ap.add_argument("--alert-frac", type=float, default=0.01)
    ap.add_argument("--max-area-frac", type=float, default=0.20)
    ap.add_argument("--max-width-mm", type=float, default=15.0,
                    help="components wider than this are blobs, not cracks; "
                         "15 mm matches what the old 12 px limit meant at the "
                         "single-camera pixel scale")
    ap.add_argument("--max-solidity", type=float, default=0.80)
    ap.add_argument("--no-shape-filter", action="store_true")
    ap.add_argument("--taper", type=float, default=0.25,
                    help="fraction of each tile edge that fades out when "
                         "overlapping tiles are averaged")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the straightness and valley tests")
    ap.add_argument("--max-straightness", type=float, default=1.12,
                    help="reject long components straighter than this; 1.0 is a "
                         "ruled line. Seams and folds are manufactured straight, "
                         "fractures are not")
    ap.add_argument("--min-straight-len-mm", type=float, default=40.0,
                    help="only test straightness above this length - a short "
                         "crack has no room to wander")
    ap.add_argument("--min-valleyness", type=float, default=2.0,
                    help="grey levels a component must be darker than BOTH its "
                         "sides. A step edge scores 0 at any contrast")
    ap.add_argument("--mm-per-px", type=float, default=0.0,
                    help="measured ground sample distance; overrides --standoff. "
                         "Photograph something of known width and divide.")
    ap.add_argument("--standoff", type=float, default=0.40,
                    help="camera-to-surface distance in metres; sets mm/px")
    ap.add_argument("--hfov", type=float, default=70.42,
                    help="horizontal FOV in degrees (C920 16:9; NOT the 78 deg "
                         "diagonal figure from the datasheet)")
    ap.add_argument("--verbose", action="store_true")
    return ap


def main():
    ap = add_arguments(argparse.ArgumentParser())
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    sc = Scanner(args)
    print(sc.describe(), flush=True)
    try:
        for _ in range(1 if args.once else args.repeat):
            cams, tot = sc.scan()
            print("\nscan %d   %s   cycle %.2f s "
                  "(capture %.2f, infer %.2f)"
                  % (tot["scan_index"],
                     "CRACK" if tot["crack"] else "clear",
                     tot["cycle_s"], tot["capture_s"], tot["infer_s"]))
            print("  dropped: %d too straight, %d not a valley"
                  % (tot["dropped_straight"], tot["dropped_flat"]))
            for c in cams:
                print("  %-6s cov %6.3f %%  %3d comp  widest %s  "
                      "exp %4d gain %3d  infer %5.0f ms"
                      % (c["name"], 100 * c["coverage"], c["components"],
                         ("%.2f mm" % c["widest_mm"]) if c["widest_mm"]
                         else "   -   ",
                         c["exposure"], c["gain"], c["infer_ms"]))
            if args.save:
                os.makedirs(args.save, exist_ok=True)
                stamp = time.strftime("%Y%m%d-%H%M%S")
                cv2.imwrite(os.path.join(args.save, "%s_composite.jpg" % stamp),
                            composite(cams))
                for c in cams:
                    cv2.imwrite(os.path.join(
                        args.save, "%s_%s.jpg" % (stamp, c["name"].lower())),
                        overlay(c["frame"], c["mask"]))
                print("  saved to %s" % args.save)
    finally:
        sc.close()


if __name__ == "__main__":
    main()
