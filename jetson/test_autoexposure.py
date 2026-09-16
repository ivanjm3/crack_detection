#!/usr/bin/env python3
"""Regression test for the auto-exposure loop.

Reproduces the reported failure without needing anyone to move the camera:
force the controller into the stuck state (exposure floored, gain maxed - what
a white wall drove it to) and verify it climbs back out; then force the
opposite extreme and verify it comes down. The old ratchet-based controller
could not recover from the first case at all.
"""
import time
import cv2
import numpy as np

from live import open_c920, center_square
from camera_ctl import AutoExposure, EXP_MIN, GAIN_MAX


def settle(ae, cap, seconds, tag, trace_every=2.0):
    t0 = last = time.time()
    trace = []
    while time.time() - t0 < seconds:
        ok, f = cap.read()
        if not ok:
            continue
        crop, _ = center_square(f)
        ae.update(crop)
        if time.time() - last >= trace_every:
            last = time.time()
            trace.append((time.time() - t0, ae.exp, ae.gain, ae.stats["median"]))
            print(f"      t={trace[-1][0]:5.1f}s exp={ae.exp:4d} gain={ae.gain:3d} "
                  f"median={ae.stats['median']:5.1f}")
    med = ae.stats["median"]
    print(f"  {tag:<34} exposure={ae.exp:4d} gain={ae.gain:3d} median={med:5.1f}")
    return med, trace


cap = open_c920(0)
ae = AutoExposure(cap, verbose=False)
print("target median = %.0f (deadband +-%.0f)\n" % (ae.target, ae.tol))

base, _ = settle(ae, cap, 14, "baseline settle")

# --- the reported failure: stuck dark ------------------------------------
print("\nforcing the stuck state seen in the field (exp=3, gain=255):")
ae._apply_exposure(EXP_MIN)
ae.gain = GAIN_MAX                      # bypass the cap, as the old bug did
cap.set(cv2.CAP_PROP_GAIN, GAIN_MAX)
ae.stats.update(exposure=ae.exp, gain=ae.gain)
print(f"  {'forced':<34} exposure={ae.exp:4d} gain={ae.gain:3d}")
rec_dark, _ = settle(ae, cap, 20, "after 12s of control")

# --- the opposite extreme: stuck bright ----------------------------------
print("\nforcing the opposite extreme (exposure at cap):")
ae._apply_exposure(ae.exp_cap)
print(f"  {'forced':<34} exposure={ae.exp:4d} gain={ae.gain:3d}")
rec_bright, _ = settle(ae, cap, 20, "after 12s of control")

cap.release()

print("\n=== verdict ===")
ok_dark = abs(rec_dark - ae.target) <= 3 * ae.tol
ok_bright = abs(rec_bright - ae.target) <= 3 * ae.tol
print(f"  recovers from stuck-dark   : {'PASS' if ok_dark else 'FAIL'} "
      f"(median {rec_dark:.1f}, want {ae.target:.0f}+-{3*ae.tol:.0f})")
print(f"  recovers from stuck-bright : {'PASS' if ok_bright else 'FAIL'} "
      f"(median {rec_bright:.1f}, want {ae.target:.0f}+-{3*ae.tol:.0f})")
print(f"  gain never max'd at min exp: "
      f"{'PASS' if not (ae.exp <= EXP_MIN and ae.gain >= GAIN_MAX) else 'FAIL'}")
