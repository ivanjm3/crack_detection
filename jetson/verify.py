"""Two geometric tests that ask whether a detection is physically a crack.

clean_mask() rejects components by size and thickness. That removes fat compact
blobs - people, cars, curtains - and it measurably removes most of the model's
out-of-domain response. What it cannot remove is anything long and thin, because
long and thin is the definition it keeps. Every false positive still standing on
this rig is long and thin: panel seams, vent louvres, the join between a wall and
a ceiling, the folded corner of a cardboard box.

So these two tests ask different questions, each targeting a property that a
crack has and a man-made edge does not.

STRAIGHTNESS - is it suspiciously perfect?

    A crack is a fracture. It follows whatever path through the material is
    weakest, which is never a straight line at the scale of its own width. A
    panel seam, a door frame or a box fold is manufactured, and is straight to
    within a pixel over hundreds of pixels.

    Measured as the RMS perpendicular distance of the component's pixels from
    its own principal axis, in units of what that spread would be if the
    component were a perfectly straight stroke of the same width. A uniform
    straight stroke of width w has an RMS perpendicular spread of w/sqrt(12), so
    the ratio is ~1.0 for a straight stroke and grows as the component wanders.

    Only applied to components long enough to have demonstrated a wander. A
    short crack is straight for the same reason a short arc of any curve is:
    there was no room to bend.

VALLEYNESS - is it actually a dark line, or just a boundary?

    A crack is a valley in the intensity surface. It is darker than the surface
    on BOTH sides, because it is a gap that light does not return from.

    A shadow line, a fold, or the edge of an object is a STEP. It is darker on
    one side and lighter on the other, and the intensity at the boundary itself
    is midway between. The second derivative across a valley is strongly
    positive at its centre; across a step it passes through zero there.

    That difference is what the Hessian measures, and it is the standard tool in
    the concrete-crack literature (Frangi-type ridge filters). The largest
    eigenvalue of the Hessian, where positive, is the valley strength across the
    darkest direction. It is evaluated at several scales because the response
    peaks when the scale matches the feature width, and crack widths vary.

Both produce a per-component number. Neither is a hard physical constant, so
both thresholds are arguments and both default to values that reject only
clear-cut cases - a filter that silently removes real cracks is worse than no
filter, because its damage is invisible in exactly the output you would use to
check it.
"""
import cv2
import numpy as np


def valleyness(gray, scales=(2.0, 3.5, 6.0)):
    """Per-pixel valley DEPTH in grey levels: how much darker than both sides.

    Returns a float32 map. A crack 30 grey levels deep reads ~30; a step edge of
    any contrast reads ~0.

    Curvature alone does not work, and the first version of this function used
    it and failed its own test. Across a blurred step the second derivative is
    positive on the dark side and negative on the bright side, so a mask lying
    on the boundary picks up the positive lobe: a high-contrast step scored 29.9
    against 26.4 for a shallow crack - ranked the wrong way round.

    What actually distinguishes them is not how sharply the intensity bends but
    WHICH SIDES ARE BRIGHTER. A crack is a gap light does not return from, so
    the surface is brighter on both sides. A step has brighter on one side and
    darker on the other, by definition. So:

        depth = min(I(p + d*n), I(p - d*n)) - I(p)

    taking n across the feature and d about its half-width. For a valley both
    samples sit on the surface and depth is the valley's true depth. For a step
    one sample sits on the dark side, which is darker than the boundary itself,
    so the minimum goes NEGATIVE and the feature is rejected no matter how
    strong its contrast. The min is doing the work: it is an AND over the two
    sides, and a step can never satisfy both.

    The Hessian is still used, but only to find the direction n across the
    feature - not to score it.
    """
    g = gray.astype(np.float32)
    h, w = g.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    best = None
    for d in scales:
        sigma = max(d / 2.0, 0.8)
        k = int(2 * round(3 * sigma) + 1)
        sm = cv2.GaussianBlur(g, (k, k), sigma)
        gxx = cv2.Sobel(sm, cv2.CV_32F, 2, 0, ksize=3)
        gyy = cv2.Sobel(sm, cv2.CV_32F, 0, 2, ksize=3)
        gxy = cv2.Sobel(sm, cv2.CV_32F, 1, 1, ksize=3)

        # Principal direction of the Hessian: the direction in which intensity
        # curves most, which for a line feature is across it. The half-angle is
        # not optional - the Hessian's orientation field has a period of pi, so
        # atan2 on the doubled angle is what makes it single-valued.
        theta = 0.5 * np.arctan2(2.0 * gxy, gxx - gyy)
        nx, ny = np.cos(theta), np.sin(theta)

        a = cv2.remap(sm, xs + d * nx, ys + d * ny, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE)
        b = cv2.remap(sm, xs - d * nx, ys - d * ny, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE)
        depth = np.minimum(a, b) - sm
        best = depth if best is None else np.maximum(best, depth)
    # Several scales are tried because the response is strongest when d matches
    # the feature's half-width, and crack widths vary across one frame.
    return np.maximum(best, 0.0)


def _straightness(ys, xs, width_px):
    """RMS perpendicular spread about the principal axis, in straight-stroke units.

    ~1.0 means as straight as a ruled line of that width. Larger means it wanders.
    """
    if ys.size < 8:
        return None
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], 1)
    pts -= pts.mean(0)
    # Principal axis from the 2x2 covariance; the minor eigenvector is the
    # direction across the stroke, so the minor eigenvalue IS the mean squared
    # perpendicular distance. No projection loop needed.
    cov = np.cov(pts, rowvar=False)
    evals = np.linalg.eigvalsh(cov)             # ascending
    rms_perp = float(np.sqrt(max(evals[0], 0.0)))
    expected = max(width_px, 1.0) / np.sqrt(12.0)
    return rms_perp / expected


def _local_valleyness(gray, x, y, w, h, scales, margin):
    """Valley map for one component's neighbourhood instead of the whole frame.

    Components are sparse - seventeen of them on a 1920x1080 frame - so
    computing the map everywhere spends 1.1 s per scan position to read a few
    thousand pixels. The margin is wide enough that the Gaussians at the largest
    scale see real pixels rather than a replicated crop edge; without it the
    border of every crop would look like a step and the test would reject
    components for an artefact of its own cropping.
    """
    h_img, w_img = gray.shape[:2]
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(w_img, x + w + margin), min(h_img, y + h + margin)
    return valleyness(gray[y0:y1, x0:x1], scales), x0, y0


def report(mask, gray, comps=None, valley_map=None, min_len_px=120,
           scales=(2.0, 3.5, 6.0)):
    """Per-component straightness and valleyness for an already-cleaned mask.

    Returns a list of dicts; the mask is not modified. Separated from apply() so
    the numbers can be inspected on real surfaces before any threshold is chosen
    from them.
    """
    margin = int(round(4 * max(scales)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        w, h = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        sub = lab[y:y + h, x:x + w] == i
        ys, xs = np.nonzero(sub)
        cnts, _ = cv2.findContours(sub.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
        perim = sum(cv2.arcLength(c, True) for c in cnts) if cnts else 0.0
        width_px = max(2.0 * area / perim, 1.0) if perim > 0 else 1.0
        length_px = area / width_px

        # Valleyness is read as the MEDIAN over the component, not the mean: a
        # component that clips the end of a genuine crack would otherwise be
        # carried by a few very strong pixels.
        if valley_map is not None:
            vals = valley_map[y:y + h, x:x + w][sub]
        else:
            vmap, vx, vy = _local_valleyness(gray, x, y, w, h, scales, margin)
            vals = vmap[y - vy:y - vy + h, x - vx:x - vx + w][sub]
        out.append({
            "label": i, "area_px": area,
            "x": x, "y": y, "w": w, "h": h,
            "width_px": width_px, "length_px": length_px,
            "straightness": _straightness(ys + y, xs + x, width_px),
            "valleyness": float(np.median(vals)) if vals.size else 0.0,
            "long_enough": length_px >= min_len_px,
        })
    return out


def apply(mask, gray, max_straightness=1.12, min_valleyness=0.0,
          min_len_px=120, valley_map=None):
    """Drop components that are too straight, or not a dark line at all.

    max_straightness defaults to 1.12 - within 12 % of a ruled line - and is
    only applied to components longer than min_len_px. Both guards exist because
    the cost of this filter is asymmetric: a rejected real crack disappears
    without trace, while a surviving false positive is visible and can be
    filtered later.

    min_valleyness defaults to 0 (off). The valley response is in intensity
    units and its useful threshold depends on surface contrast, so it is left
    for a value measured on the real surface rather than guessed here. report()
    exists to produce that measurement.
    """
    comps = report(mask, gray, valley_map=valley_map, min_len_px=min_len_px)
    n, lab, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    dropped = {"straight": 0, "flat": 0}
    for c in comps:
        if (c["long_enough"] and c["straightness"] is not None
                and c["straightness"] < max_straightness):
            dropped["straight"] += 1
            continue
        if min_valleyness > 0 and c["valleyness"] < min_valleyness:
            dropped["flat"] += 1
            continue
        keep[lab == c["label"]] = 255
    return keep, dropped
