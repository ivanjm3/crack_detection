#!/usr/bin/env python3
"""Does the width estimator return the width? Checked against known geometry.

Crack width is the headline number this system reports, and it is the one number
nobody can sanity-check by eye - a 3 mm crack and a 6 mm crack look identical on
a screen without a scale bar. So it is verified against masks whose true width is
known by construction, before it is trusted on a real surface.

Four shapes, each the cause of a specific error:

    straight     the easy case; any estimator should pass
    diagonal     rasterisation makes a diagonal stroke's perimeter longer than
                 its true outline, which biases area/perimeter low
    branched     a fork has far more perimeter per unit area than one stroke,
                 the same bias but much larger
    tapered      wide at one end, hairline at the other; the mean describes
                 neither end, and it is the wide end that gets graded

Run:  python3 test_measure.py
"""
import sys

import cv2
import numpy as np

import measure

GSD = 1.0          # 1 mm per pixel, so millimetres and pixels are the same
                   # number here and an error is read directly as pixels


def _stroke(size, p0, p1, w_lo, w_hi=None):
    """Analytic stroke: every pixel within w/2 of the segment, w tapering.

    Built from a distance field rather than cv2.line because cv2.line's drawn
    width is not the thickness asked for - thickness=3 puts down 5 rows - so a
    test using it would be checking the estimator against a width that was never
    on the image. Here the width is exact by construction.
    """
    w_hi = w_lo if w_hi is None else w_hi
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    seg2 = dx * dx + dy * dy
    t = np.clip(((xs - x0) * dx + (ys - y0) * dy) / seg2, 0.0, 1.0)
    px, py = x0 + t * dx, y0 + t * dy
    dist = np.hypot(xs - px, ys - py)
    half = (w_lo + t * (w_hi - w_lo)) / 2.0
    return ((dist <= half) * 255).astype(np.uint8)


def true_width(mask, axis=0):
    """The width actually present in the image, counted along `axis`.

    The generator is analytic, but rasterising a width of w still yields a
    pixel count that can differ from w by one, so the check is against what is
    on the image rather than what was requested.
    """
    counts = (mask > 0).sum(axis=axis)
    counts = counts[counts > 0]
    return float(np.median(counts)) if counts.size else 0.0


def straight(width, size=420):
    return _stroke(size, (60, size // 2), (360, size // 2), width)


def diagonal(width, size=420):
    return _stroke(size, (70, 70), (330, 330), width)


def branched(width, size=420):
    m = _stroke(size, (40, 210), (240, 210), width)
    m = np.maximum(m, _stroke(size, (240, 210), (380, 110), width))
    return np.maximum(m, _stroke(size, (240, 210), (380, 310), width))


def tapered(w_lo, w_hi, size=420):
    return _stroke(size, (50, size // 2), (370, size // 2), w_lo, w_hi)


def widths(mask):
    c = measure.components(mask, GSD, min_area=1)
    if not c:
        return None
    c = c[0]
    return (c["width_mm"], c["width_median_mm"], c["width_max_mm"],
            c["width_mean_mm"])


def main():
    print("all figures in pixels (gsd = 1.0 mm/px)\n")
    print("%-28s %8s %8s %8s %8s %8s"
          % ("case", "true", "WIDTH", "median", "max", "area/per"))

    cases = []
    for w in (3, 5, 9, 15):
        m = straight(w)
        cases.append(("straight w=%d" % w, true_width(m), m))
    for w in (3, 5, 9):
        # Measured perpendicular to a 45 degree stroke: a vertical pixel count
        # across a diagonal overstates the width by sqrt(2).
        m = diagonal(w)
        cases.append(("diagonal w=%d" % w, true_width(m) / np.sqrt(2.0), m))
    for w in (3, 5, 9):
        m = branched(w)
        cases.append(("branched w=%d" % w, true_width(m), m))
    # For a taper the quantity that matters for grading is the WIDE end, so that
    # is what the "true" column holds; the ridge median is expected to sit near
    # the middle of the range and p95/max near the wide end.
    cases.append(("tapered 2->10 (wide end)", 10, tapered(2, 10)))
    cases.append(("tapered 3->15 (wide end)", 15, tapered(3, 15)))
    cases.append(("tapered 1->6  (wide end)", 6, tapered(1, 6)))

    worst_ridge = 0.0
    fails = []
    for name, true_w, mask in cases:
        got = widths(mask)
        if got is None:
            fails.append("%s: no component found" % name)
            continue
        width, med, mx, mean = got
        print("%-28s %8.2f %8.2f %8.2f %8.2f %8.2f"
              % (name, true_w, width, med, mx, mean))

        err = abs(width - true_w)
        worst_ridge = max(worst_ridge, err)
        if name.startswith("tapered"):
            # The reported width must find the WIDE end: that is the end that
            # gets graded, and understating it is the failure that matters.
            if err > 1.5:
                fails.append("%s: width %.2f, expected ~%.1f"
                             % (name, width, true_w))
            if med >= true_w:
                fails.append("%s: median %.2f should sit below the wide end %.1f"
                             % (name, med, true_w))
        elif err > 0.75:
            fails.append("%s: width %.2f, true %.2f (err %.2f px)"
                         % (name, width, true_w, err))

    print("\nworst error on the reported width: %.2f px" % worst_ridge)
    if fails:
        print("\nFAIL")
        for f in fails:
            print("  " + f)
        return 1
    print("PASS - width recovered within 0.75 px on uniform and branched strokes, "
          "and the taper's wide end within 1.5 px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
