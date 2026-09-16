"""Closed-loop exposure control for the C920 (fix #1).

Why this exists: setup.md 4.2 pins exposure to a single value tuned for one
scene. Point the camera at a white wall and the sensor saturates - large areas
sit at 255 and stay there. A hairline crack is a low-contrast dark line, so
once its surroundings clip, the contrast carrying the crack is destroyed at the
sensor. No downstream contrast processing can recover it: the information is
gone before the JPEG is even encoded.

Design (a damped proportional loop onto a target luminance, the standard
machine-vision approach):

  * The control metric is the MEDIAN luma, driven to a mid-grey target. The
    median is always reachable, which matters: an earlier version targeted the
    99th percentile at 235, unreachable on a bimodal scene (dark room + bright
    window), so the loop wound up until the highlights blew out, cut hard, and
    limit-cycled. A reachable target needs no ratchet to stay stable.

  * Exposure and gain form a two-stage ladder with a strict ordering:
    going brighter raises exposure first and only touches gain once exposure
    is at its MAXIMUM; going darker drops gain first and only then exposure.
    An earlier version could raise gain while exposure sat at its MINIMUM,
    producing exposure=3 with gain=255 - the darkest, noisiest image possible,
    with no way back. The ladder makes that state unreachable.

  * Exposure is capped well below the device maximum. exposure_time_absolute
    is in 100 us units, so the device limit of 2047 is a 205 ms shutter: both
    heavy motion blur and a ~5 fps cap. Gain covers what is left.

  * Gain is kept as low as possible. Gain amplifies sensor noise into thin
    bright streaks - exactly the filaments the model hunts for. This camera was
    found sitting at gain=255 (its maximum, default 0), left there by the
    firmware's own auto-exposure before manual mode was set.

Both controls go through cv2.VideoCapture.set(), which maps to the same V4L2
controls v4l2-ctl writes - verified on this device.

Note on the C920 specifically: its firmware drops out of manual mode during
stream initialisation and ignores exposure set before streaming starts, so
open_c920() re-applies the controls after the first frames. This matches
reports on the linux-media list.
"""
import time

import cv2
import numpy as np

# device limits, from `v4l2-ctl -d /dev/video0 --list-ctrls`
EXP_MIN, EXP_MAX = 3, 2047
GAIN_MIN, GAIN_MAX = 0, 255


class AutoExposure:
    def __init__(self, cap, target=120.0, tol=8.0, interval=0.4,
                 exposure=156, gain=0, damp=0.6, exp_cap=500,
                 gain_cap=160, gain_step=16, clip_guard=0.05,
                 settle=0.5, flush=5, verbose=False):
        self.cap = cap
        self.target = float(target)
        self.tol = float(tol)
        self.interval = float(interval)
        self.damp = float(damp)
        self.exp_cap = int(min(exp_cap, EXP_MAX))
        self.gain_cap = int(min(gain_cap, GAIN_MAX))
        self.gain_step = int(gain_step)
        self.clip_guard = float(clip_guard)
        self.settle = float(settle)
        self.flush = int(flush)
        self.verbose = verbose

        self.exp = int(exposure)
        self.gain = int(gain)
        self._last = 0.0
        # The sensor takes several frames to honour a new exposure, and the
        # C920 emits black/garbage frames in between. Measured at a FIXED
        # exposure=500 gain=0, consecutive medians came back as 4 and 241.
        # Acting on those transients is what made the loop thrash. So after
        # every change: drop the in-flight frames, stay blind briefly, and
        # smooth what is left.
        self._blind_until = 0.0
        self.med_ema = None
        self.stats = {"median": 0.0, "clipped": 0.0,
                      "exposure": self.exp, "gain": self.gain}
        self._apply_gain(self.gain)
        self._apply_exposure(self.exp)

    def _apply_exposure(self, value):
        self.exp = int(np.clip(round(value), EXP_MIN, self.exp_cap))
        self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exp)

    def _apply_gain(self, value):
        self.gain = int(np.clip(round(value), GAIN_MIN, self.gain_cap))
        self.cap.set(cv2.CAP_PROP_GAIN, self.gain)

    def update(self, frame_bgr, force=False):
        """Call once per frame; self-throttles to `interval` seconds."""
        now = time.time()
        if not force and (now - self._last < self.interval
                          or now < self._blind_until):
            return False
        self._last = now

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        med_raw = float(np.median(gray))
        clipped = float((gray >= 254).mean())

        if self.med_ema is None:
            self.med_ema = med_raw
        else:
            self.med_ema = 0.5 * self.med_ema + 0.5 * med_raw
        med = self.med_ema
        self.stats.update(median=med, clipped=clipped,
                          exposure=self.exp, gain=self.gain)

        # Effective target: if a lot of the frame is blown out we are losing
        # crack contrast right now, so aim lower until the highlights recover.
        target = self.target * (0.8 if clipped > self.clip_guard else 1.0)
        err = med - target

        if abs(err) <= self.tol:
            # Converged on brightness - but the ladder can converge at a noisy
            # operating point (recovering from the stuck state lands on
            # exposure=71 gain=111, correct luma, needless noise). Gain costs
            # image quality and exposure does not, up to the motion-blur cap,
            # so trade a little gain for exposure and let the loop re-settle.
            if self.gain > GAIN_MIN and self.exp < self.exp_cap:
                self._apply_gain(self.gain - min(8, self.gain))
                self._apply_exposure(self.exp * 1.2)
                for _ in range(self.flush):
                    self.cap.grab()
                self._blind_until = time.time() + self.settle
                return True
            return False                        # deadband: don't hunt

        # Luma is ~linear in exposure, so target/med is a Newton step; damp it
        # and clamp the per-step ratio so a bad frame can't cause a big jump.
        ratio = float(np.clip(target / max(med, 1.0), 0.4, 2.5)) ** self.damp
        # Scale the gain step with the error too, so unwinding a maxed-out gain
        # takes a few steps rather than a few dozen.
        gstep = int(np.clip(abs(err) / max(target, 1.0) * 96, self.gain_step, 96))

        if err < 0:                             # too dark -> brighten
            if self.exp < self.exp_cap:
                self._apply_exposure(self.exp * ratio)
            elif self.gain < self.gain_cap:
                self._apply_gain(self.gain + gstep)
            else:
                return False                    # at both caps; nothing to do
        else:                                   # too bright -> darken
            if self.gain > GAIN_MIN:
                self._apply_gain(self.gain - gstep)
            else:
                self._apply_exposure(self.exp * ratio)

        # Drop the frames already in flight so the next measurement sees the
        # new setting rather than the old one (or a transient black frame).
        for _ in range(self.flush):
            self.cap.grab()
        self._blind_until = time.time() + self.settle

        if self.verbose:
            print(f"    [ae] median={med:5.1f} clip={100*clipped:5.2f}% "
                  f"-> exposure={self.exp:4d} gain={self.gain:3d}")
        return True

    def label(self):
        s = self.stats
        return (f"exp={s['exposure']} gain={s['gain']} "
                f"med={s['median']:.0f} clip={100*s['clipped']:.1f}%")
