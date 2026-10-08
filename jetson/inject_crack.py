#!/usr/bin/env python3
"""Draw a physically plausible crack into a real captured frame.

There is no labelled crack image on this board and no cracked surface in front
of the cameras, so the one question the verification filters have not been asked
is the one that matters: does a REAL crack survive them?

A synthetic mask on a synthetic background cannot answer it, because the valley
test reads the IMAGE, not the mask - it would be measuring a background that was
never photographed. So the crack is drawn into a frame this rig actually
captured, keeping that frame's texture, lighting, vignetting and sensor noise,
and only the crack is known by construction.

What makes it plausible rather than merely dark:

  it meanders       a fracture follows the weakest path; a ruled line would be
                    rejected by the straightness test and would deserve to be
  it tapers         cracks are wider in the middle of a span than at their ends
  it is a VALLEY    darkened relative to the local surface on both sides, which
                    is what a gap in a surface does to the light
  it is soft-edged  a lens has a point spread function; a hard-edged line is a
                    drawing, not a photograph
  it is multiplicative, not a constant subtraction: a crack reflects a FRACTION
                    of what the surface would, so its depth follows the local
                    illumination instead of punching to the same grey in bright
                    and shadowed areas alike

    python3 inject_crack.py --frame explain/raw_top.png --width-px 4
"""
import argparse
import os

import cv2
import numpy as np


def crack_path(h, w, rng, wander=0.06, steps=400):
    """A meandering top-to-bottom path, as a random walk with momentum.

    Momentum is what makes it look like a fracture rather than noise: a crack
    keeps going in roughly the direction it was going, and turns gradually.
    """
    x = w * 0.5
    vx = 0.0
    pts = []
    for i in range(steps):
        y = h * i / float(steps - 1)
        vx = 0.85 * vx + rng.normal(0, wander * w / steps * 12)
        x = float(np.clip(x + vx, w * 0.12, w * 0.88))
        pts.append((x, y))
    return pts


def inject(frame, width_px=4.0, depth=0.45, seed=3, blur=1.2):
    """Return (image_with_crack, ground_truth_mask)."""
    rng = np.random.default_rng(seed)
    h, w = frame.shape[:2]
    pts = crack_path(h, w, rng)

    # Build the crack profile at float precision on its own canvas, so the
    # taper and the soft edge are not quantised twice.
    canvas = np.zeros((h, w), np.float32)
    n = len(pts)
    for i, (x, y) in enumerate(pts):
        t = i / float(n - 1)
        # Taper: full width across the middle, narrowing to nothing at the ends.
        taper = np.sin(np.pi * t) ** 0.6
        rad = max(width_px * taper / 2.0, 0.0)
        if rad <= 0.05:
            continue
        cv2.circle(canvas, (int(round(x)), int(round(y))),
                   max(int(round(rad)), 1), 1.0, -1)
    canvas = cv2.GaussianBlur(canvas, (0, 0), blur)
    canvas = np.clip(canvas, 0.0, 1.0)

    # Multiplicative darkening: the crack returns (1 - depth) of the light the
    # surface would have returned at that point.
    out = frame.astype(np.float32)
    atten = 1.0 - depth * canvas[..., None]
    out *= atten
    # Photon noise does not vanish because a region is dark; re-adding a little
    # keeps the crack from being unnaturally smooth compared with its surround.
    out += rng.normal(0, 1.5, out.shape) * canvas[..., None]
    out = np.clip(out, 0, 255).astype(np.uint8)

    truth = (canvas > 0.5).astype(np.uint8) * 255
    return out, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True)
    ap.add_argument("--out", default="injected")
    ap.add_argument("--width-px", type=float, default=4.0)
    ap.add_argument("--depth", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    frame = cv2.imread(args.frame)
    if frame is None:
        raise SystemExit("cannot read %s" % args.frame)
    os.makedirs(args.out, exist_ok=True)

    for width in (2.0, 4.0, 8.0):
        img, truth = inject(frame, width, args.depth, args.seed)
        base = "crack_w%d" % int(width)
        cv2.imwrite(os.path.join(args.out, base + ".png"), img)
        cv2.imwrite(os.path.join(args.out, base + "_truth.png"), truth)
        print("%s  %d truth px, mean depth %.1f grey levels"
              % (base, int((truth > 0).sum()),
                 float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
                       - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[truth > 0].mean())))


if __name__ == "__main__":
    main()
