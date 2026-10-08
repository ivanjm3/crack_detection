#!/usr/bin/env python3
"""Do the two verification tests separate what they claim to separate?

Both are built on an assertion about physics, and an assertion about physics can
be checked on images where the physics is known by construction:

    a crack is a VALLEY  - darker than the surface on both sides
    an edge is a STEP    - darker on one side, lighter on the other
    a crack WANDERS      - it follows the weakest path through the material
    a seam is STRAIGHT   - it was manufactured

If the valley test cannot tell a dark line from a step of the same contrast,
it is measuring contrast rather than shape and will reject nothing useful.

Run:  python3 test_verify.py
"""
import sys

import cv2
import numpy as np

import verify

SIZE = 400
RNG = np.random.default_rng(7)


def noisy(img, sigma=2.0):
    """Sensor noise, so nothing passes by being unrealistically clean."""
    return np.clip(img + RNG.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def valley_image(width=3, depth=60, base=150, blur=1.0):
    """Uniform surface with a dark line down the middle - a crack."""
    img = np.full((SIZE, SIZE), float(base))
    c = SIZE // 2
    img[:, c - width // 2: c - width // 2 + width] -= depth
    img = cv2.GaussianBlur(img, (0, 0), blur)
    return noisy(img), c


def step_image(depth=60, base=150, blur=1.0):
    """Half dark, half light - a shadow, a fold, or the edge of an object."""
    img = np.full((SIZE, SIZE), float(base))
    c = SIZE // 2
    img[:, c:] -= depth
    img = cv2.GaussianBlur(img, (0, 0), blur)
    return noisy(img), c


def strip_mask(col, width=3):
    """The mask the model would produce along a vertical feature."""
    m = np.zeros((SIZE, SIZE), np.uint8)
    m[20:SIZE - 20, col - width // 2: col - width // 2 + width] = 255
    return m


def wavy_mask(amplitude, width=3, periods=3.0):
    """A stroke that wanders by +-amplitude px about a vertical line."""
    m = np.zeros((SIZE, SIZE), np.uint8)
    for y in range(20, SIZE - 20):
        t = (y - 20) / float(SIZE - 40)
        x = int(round(SIZE // 2 + amplitude * np.sin(2 * np.pi * periods * t)))
        m[y, x - width // 2: x - width // 2 + width] = 255
    return m


def main():
    fails = []

    print("VALLEY vs STEP  (median Hessian valley response over the mask)\n")
    print("%-34s %12s" % ("image", "valleyness"))
    results = {}
    for name, (img, col) in (("crack: 3 px dark line", valley_image(3)),
                             ("crack: 6 px dark line", valley_image(6)),
                             ("crack: shallow (depth 25)", valley_image(3, 25)),
                             ("edge: step, same contrast", step_image(60)),
                             ("edge: step, high contrast", step_image(110))):
        mask = strip_mask(col)
        rep = verify.report(mask, img, min_len_px=10)
        v = rep[0]["valleyness"] if rep else 0.0
        results[name] = v
        print("%-34s %12.2f" % (name, v))

    weakest_crack = min(v for k, v in results.items() if k.startswith("crack"))
    strongest_edge = max(v for k, v in results.items() if k.startswith("edge"))
    print("\nweakest crack %.2f   strongest edge %.2f   separation %.1fx"
          % (weakest_crack, strongest_edge,
             weakest_crack / max(strongest_edge, 1e-6)))
    if weakest_crack <= strongest_edge:
        fails.append("valleyness does not separate cracks from steps: "
                     "weakest crack %.2f <= strongest edge %.2f"
                     % (weakest_crack, strongest_edge))

    print("\n\nSTRAIGHTNESS  (1.0 = as straight as a ruled line of that width)\n")
    print("%-34s %12s" % ("mask", "straightness"))
    img, col = valley_image(3)
    straight = verify.report(strip_mask(col), img, min_len_px=10)[0]["straightness"]
    print("%-34s %12.3f" % ("ruled straight line", straight))
    if abs(straight - 1.0) > 0.25:
        fails.append("a straight line scored %.3f, expected ~1.0" % straight)

    prev = straight
    for amp in (2, 4, 8, 16):
        m = wavy_mask(amp)
        s = verify.report(m, img, min_len_px=10)[0]["straightness"]
        print("%-34s %12.3f" % ("wanders +-%d px" % amp, s))
        if s <= prev:
            fails.append("wander +-%d px scored %.3f, not above the previous "
                         "%.3f - the measure is not monotonic" % (amp, s, prev))
        prev = s

    # The default threshold has to keep every wandering stroke and drop the
    # ruled one; if it cannot, the default is wrong whatever the measure does.
    thr = 1.12
    m = wavy_mask(2)
    s2 = verify.report(m, img, min_len_px=10)[0]["straightness"]
    print("\ndefault max_straightness = %.2f" % thr)
    print("  ruled line        %.3f  -> %s" % (straight,
                                               "dropped" if straight < thr else "KEPT"))
    print("  gentlest wander   %.3f  -> %s" % (s2,
                                               "dropped" if s2 < thr else "kept"))
    if straight >= thr:
        fails.append("the ruled line survives the default threshold")
    if s2 < thr:
        fails.append("the gentlest wander (+-2 px) is dropped by the default "
                     "threshold - too aggressive")

    if fails:
        print("\nFAIL")
        for f in fails:
            print("  " + f)
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
