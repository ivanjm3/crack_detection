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
        width_px = 2.0 * area / perim
        # A single isolated pixel has area 1 and perimeter 4, giving a width of
        # half a pixel. Physically a detected pixel is at least one pixel wide,
        # and reporting sub-pixel widths would imply a precision the sensor does
        # not have, so the floor is one pixel.
        width_px = max(width_px, 1.0)
        out.append({
            "area_px": area,
            "width_mm": width_px * gsd,
            "length_mm": (area / width_px) * gsd,
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
        "longest_mm": max((c["length_mm"] for c in comps), default=None),
    }
