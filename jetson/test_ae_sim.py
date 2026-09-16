#!/usr/bin/env python3
"""Deterministic test of the auto-exposure control law.

The live room cannot validate this: with someone moving in frame, the median
swings 19 <-> 178 at a FIXED exposure, so any pass/fail is noise. This drives
the same controller against a simulated sensor whose response is known, and
which reproduces the two behaviours that broke earlier versions:

  * actuation lag - a change takes `lag` frames to appear
  * transient garbage - the C920 emits black/blown frames right after a change

It checks convergence from both extremes, including the exact stuck state the
field failure produced (exposure floored, gain maxed).
"""
import sys
import numpy as np

from camera_ctl import AutoExposure, EXP_MIN, EXP_MAX, GAIN_MIN, GAIN_MAX

CAP_PROP_EXPOSURE, CAP_PROP_GAIN = 15, 14      # cv2 constants


class FakeCap:
    """Sensor model: luma proportional to exposure * gain, then clipped."""

    def __init__(self, reflectance=0.5, lag=3, garbage=2, seed=0):
        self.reflectance = reflectance
        self.lag = lag
        self.garbage = garbage
        self.exp, self.gain = 156, 0
        self.pending = []
        self.since_change = 99
        self.rng = np.random.default_rng(seed)

    def set(self, prop, value):
        self.pending.append((self.lag, prop, int(value)))
        self.since_change = 0
        return True

    def _tick(self):
        nxt = []
        for n, prop, value in self.pending:
            if n <= 0:
                if prop == CAP_PROP_EXPOSURE:
                    self.exp = value
                else:
                    self.gain = value
            else:
                nxt.append((n - 1, prop, value))
        self.pending = nxt
        self.since_change += 1

    def grab(self):
        self._tick()
        return True

    def read(self):
        self._tick()
        # transient garbage right after a control change
        if self.since_change <= self.garbage:
            v = 0 if self.rng.random() < 0.5 else 255
            return True, np.full((64, 64, 3), v, np.uint8)
        luma = 255.0 * self.reflectance * (self.exp / 250.0) * (1 + self.gain / 48.0)
        luma = float(np.clip(luma, 0, 255))
        frame = np.full((64, 64, 3), int(round(luma)), np.uint8)
        return True, frame

    def true_median(self):
        luma = 255.0 * self.reflectance * (self.exp / 250.0) * (1 + self.gain / 48.0)
        return float(np.clip(luma, 0, 255))


def run(name, reflectance, start_exp, start_gain, steps=400, target=120.0):
    cap = FakeCap(reflectance=reflectance)
    ae = AutoExposure(cap, target=target, exposure=start_exp, gain=start_gain)
    cap.exp, cap.gain, cap.pending = start_exp, start_gain, []
    ae.exp, ae.gain = start_exp, start_gain
    ae._last = -1e9

    for _ in range(steps):
        ok, frame = cap.read()
        ae._last = -1e9            # bypass wall-clock throttling in the sim
        ae._blind_until = 0.0
        ae.update(frame)

    final = cap.true_median()
    ok = abs(final - target) <= 3 * ae.tol
    # A scene can be too dark to reach the target within the exposure and gain
    # caps. Maxing out is the correct response, not a failure.
    maxed = cap.exp >= ae.exp_cap and cap.gain >= ae.gain_cap and final < target
    # The state the field failure produced: darkest exposure, noisiest gain.
    bad = cap.exp <= EXP_MIN and cap.gain >= GAIN_MAX
    passed = (ok or maxed) and not bad
    note = "  (at caps, scene too dark)" if maxed and not ok else ""
    print(f"  {name:<32} exp={cap.exp:4d} gain={cap.gain:3d} "
          f"luma={final:6.1f}  {'PASS' if passed else 'FAIL'}{note}")
    return passed


print("simulated sensor, target median 120 (+-24 accepted)\n")
results = [
    run("mid-grey, from default",      0.50, 156,   0),
    run("STUCK DARK (exp=3,gain=255)", 0.50, EXP_MIN, GAIN_MAX),
    run("stuck bright (exp at cap)",   0.50, 500,   0),
    run("white wall (high reflect)",   0.95, 156,   0),
    run("white wall, from stuck dark", 0.95, EXP_MIN, GAIN_MAX),
    run("dark asphalt (low reflect)",  0.12, 156,   0),
    run("very dark scene",             0.04, 156,   0),
]
print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
