"""Turn a cleaned mask into physical measurements.

The single-camera pipeline reports coverage - the fraction of pixels the model
called crack. That is a useful alarm and a useless measurement: it conflates one
wide crack with a dozen hairlines, and it changes if the camera moves closer
without the surface changing at all. A survey needs the answer in millimetres.

Two things make millimetres available here that were not available before:

  * native-resolution tiling, so a pixel is a known size on the ground rather
    than a size that depends on an arbitrary downscale (see tiler.py), and
  * a fixed standoff, so that size is constant across the frame. On a stationary
    rig looking at a flat surface this is a scale factor, not a homography.

Crack WIDTH is the quantity that matters - standards grade cracks by width, not
by area - and it is recovered from the component's own geometry:

    mean half-width = area / perimeter

A stroke of width w and length L has area ~ wL and perimeter ~ 2L (the two long
sides dominate once L >> w), so area/perimeter ~ w/2. This is the same identity
clean_mask() already uses to reject components that are too thick; here it is
read as a measurement rather than a test.

Two things that look like alternatives and are not:

  cv2.contourArea collapses to ~0 for a 1-px-wide line, so area must come from
  the connected-component pixel count. Using contourArea would make every
  hairline measure zero width - the exact cracks that matter most.

  The bounding box is not a width. A diagonal crack's box is as wide as the
  crack is long, and a branching crack's box covers the whole branch structure.
"""
import cv2
import numpy as np


def gsd_mm_px(standoff_m, width_px, hfov_deg=70.42):
    """Ground sample distance: millimetres of surface per pixel.

    hfov_deg defaults to the C920's horizontal field of view at 16:9. The
    camera is specified by its 78 degree DIAGONAL field, which is not the number
    needed here; using the diagonal would overstate the swath by about 11 % and
    every width measured would inherit that error.
    """
    swath_mm = 2.0 * standoff_m * np.tan(np.radians(hfov_deg) / 2.0) * 1000.0
    return swath_mm / float(width_px)


def _width_from_distance(sub):
    """Width in pixels from the distance transform, as (median, p95, max).

    area/perimeter gives the MEAN width over the whole component, which is the
    right answer for a clean straight stroke and the wrong one in two cases that
    occur constantly:

      Branching. A forked crack has far more perimeter per unit area than a
      single stroke, so the mean width comes out too small - exactly where a
      crack is most structurally interesting.

      Taper. A crack that is 4 mm at one end and hairline at the other has a
      mean that describes neither end, and the widest point is the one that
      matters for grading.

    The distance transform gives, for every interior pixel, the distance to the
    nearest background pixel - which is the LOCAL half-width at that point. The
    ridge running down the middle of the stroke therefore carries the local
    width profile, and a percentile over the ridge is a width that survives both
    branching and taper.

    Only ridge pixels are counted. Every component tapers to zero at its ends,
    so averaging the distance transform over all pixels would drag any width
    toward zero regardless of the shape - the ridge is where the width actually
    lives. The ridge is taken as pixels within half a pixel of the local
    maximum, which is cheap and needs no skeletonisation.
    """
    # The component arrives as its own tight bounding box, so a stroke can fill
    # the crop edge to edge and leave NO background pixel inside it. The
    # distance transform then has nothing to measure from and returns distances
    # of order the crop size: a 2 px line 300 px long measured as 302 px wide.
    # One pixel of zero border gives every stroke an outside to be measured
    # against, and costs nothing.
    sub = cv2.copyMakeBorder(sub, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(sub, cv2.DIST_L2, 5)
    peak = cv2.dilate(dist, np.ones((3, 3), np.uint8))
    ridge = dist[(dist > 0) & (dist >= peak - 0.5)]
    if ridge.size == 0:
        return 0.0, 0.0, 0.0

    # Width is 2d - 1, not 2d. The transform measures to the nearest ZERO
    # PIXEL, whose centre lies one full pixel beyond the last foreground pixel,
    # while the stroke's actual edge is only half a pixel beyond it. So the
    # ridge value overstates the half-width by exactly 0.5 px, and 2d would
    # report a 3 px crack as 4 px - at 0.29 mm/px that is a 0.29 mm error on a
    # measurement whose whole point is sub-millimetre resolution.
    def to_width(d):
        return 2.0 * d - 1.0

    return (to_width(float(np.median(ridge))),
            to_width(float(np.percentile(ridge, 95))),
            to_width(float(ridge.max())))


def components(mask, gsd, min_area=1):
    """Per-component measurements for a cleaned mask.

    Returns a list of dicts, widest first:
        area_px, length_mm, width_mm, x, y, w, h

    Length is estimated as area / width, i.e. perimeter / 2, rather than from
    the bounding box diagonal - a meandering crack is longer than its box.
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        w, h = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        sub = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        perim = sum(cv2.arcLength(c, True) for c in cnts)
        if perim <= 0:
            continue
        mean_px = 2.0 * area / perim
        med_px, p95_px, max_px = _width_from_distance(sub)
        # A single isolated pixel has area 1 and perimeter 4, giving a width of
        # half a pixel. Physically a detected pixel is at least one pixel wide,
        # and reporting sub-pixel widths would imply a precision the sensor does
        # not have, so the floor is one pixel.
        mean_px = max(mean_px, 1.0)
        med_px = max(med_px, 1.0)
        out.append({
            "area_px": area,
            # The headline width is the 95th percentile of the ridge, chosen on
            # measured evidence rather than taste (test_measure.py):
            #
            #   uniform strokes  p95 is exact - 3, 5, 9, 15 px recovered as
            #                    3.00, 5.00, 9.00, 15.00
            #   branched         p95 is exact where the ridge MEDIAN is not:
            #                    the junction contributes a cluster of short
            #                    distances that drags the median to 1.80 on a
            #                    3 px fork, a 40 % underestimate
            #   tapered          p95 lands on the wide end, which is the end
            #                    that gets graded; the median describes neither
            #                    end of a taper
            #
            # max is not used as the headline because a single ragged pixel at a
            # junction sets it - on the 9 px fork max reads 10.59 against a true
            # 9.00. p95 discards that tail without discarding the wide end.
            "width_mm": max(p95_px, 1.0) * gsd,
            "width_max_mm": max(max_px, 1.0) * gsd,
            "width_median_mm": med_px * gsd,
            "width_mean_mm": mean_px * gsd,
            # Length from area/mean-width: a meandering crack is longer than its
            # bounding box diagonal, so the box cannot be used here.
            "length_mm": (area / mean_px) * gsd,
            "x": x, "y": y, "w": w, "h": h,
        })
    out.sort(key=lambda c: c["width_mm"], reverse=True)
    return out


def summarise(mask, gsd, min_area=1):
    """Coverage, component count and widest crack for one camera."""
    comps = components(mask, gsd, min_area)
    return {
        "coverage": float((mask > 0).mean()),
        "components": len(comps),
        "widest_mm": comps[0]["width_mm"] if comps else None,
        "widest_max_mm": max((c["width_max_mm"] for c in comps), default=None),
        "longest_mm": max((c["length_mm"] for c in comps), default=None),
    }
