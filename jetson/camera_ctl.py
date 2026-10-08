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

# The exposure ladder this camera ACTUALLY has while streaming (fix #4).
#
# exposure_time_absolute advertises a range of 3..2047 in steps of 1, and when
# no stream is running the driver accepts all 747 of those values. While
# streaming at 1080p it does not: the camera snaps the shutter to a x2 ladder
# and silently clamps everything else to the nearest rung. Measured on this
# board by writing every value 5..899 with a stream open and reading back what
# the camera settled on - the only values that ever came back were these.
#
# This is the root cause of a failure that looked like bad loop tuning: luma
# could only ever be 99 or 148 on the test scene, so a target of 120 sat in a
# gap between rungs and the loop oscillated between them forever. One stop of
# exposure resolution cannot hit an arbitrary luma. Gain can, so gain - not
# exposure - has to be the fine actuator, which inverts part of the ladder
# below.
#
# (156 and 312 being adjacent rungs also explains the very first exposure bug,
# where the value "drifted 156 -> 312" on stream start. It was not drifting by
# a little; it was being pushed up exactly one rung.)
EXP_RUNGS = (19, 38, 77, 156, 312, 624, 1250, 2047)


def rung_index(value, rungs=EXP_RUNGS):
    """Index of the rung nearest `value`, in log space.

    Log space because the rungs are geometric: between 156 and 312, linear
    distance would call 230 "nearer" to 156 by a hair while in stops it is
    almost exactly halfway. Brightness is what we care about and it scales with
    the ratio, not the difference.
    """
    return min(range(len(rungs)),
               key=lambda i: abs(np.log(value / rungs[i])))


class AutoExposure:
    def __init__(self, cap, target=120.0, tol=30.0, interval=0.4,
                 exposure=156, gain=0, damp=0.6, exp_cap=500,
                 gain_cap=160, gain_step=16, clip_guard=0.05,
                 settle=0.5, flush=5, trade_gain=48, max_trades=4,
                 verbose=False):
        self.cap = cap
        self.target = float(target)
        # tol is a quarter of the target, not the 8 it started as, because a
        # tolerance finer than the actuator's resolution is not a tolerance -
        # it is an instruction to use the other actuator. Exposure moves in
        # whole stops, so the rung nearest a given target can be up to 41 %
        # away in luma; demanding +-8 of 120 meant every gap had to be filled
        # with gain, and the simulation showed the loop settling at gain 93-111
        # to buy a luma it did not need. Gain amplifies read noise into thin
        # bright streaks, which is precisely the feature the crack model keys
        # on, so paying noise for an arbitrary luma target is the worst
        # available trade. A band of +-30 admits a rung directly in most scenes
        # and leaves gain at zero.
        self.tol = float(tol)
        self.interval = float(interval)
        self.damp = float(damp)
        self.exp_cap = int(min(exp_cap, EXP_MAX))
        self.gain_cap = int(min(gain_cap, GAIN_MAX))
        self.gain_step = int(gain_step)
        self.clip_guard = float(clip_guard)
        self.settle = float(settle)
        self.flush = int(flush)
        self.trade_gain = int(trade_gain)
        self.max_trades = int(max_trades)
        self.verbose = verbose

        self.exp = int(exposure)
        self.gain = int(gain)
        self._trades = 0
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

    @property
    def rungs(self):
        """The rungs this instance may use, honouring exp_cap.

        exp_cap exists to bound motion blur, so a rung above it is not a legal
        operating point - and keeping illegal rungs in the list would let
        _step_exposure "move up" to a value that then gets clipped back, which
        reads as a step that did nothing.
        """
        usable = [r for r in EXP_RUNGS if EXP_MIN <= r <= self.exp_cap]
        return tuple(usable) if usable else (EXP_MIN,)

    def _apply_exposure(self, value):
        """Snap to the nearest legal rung and write it.

        Writing an unsnapped value is not harmless: the camera clamps it to a
        rung anyway, so self.exp would record a setting the hardware is not
        using, and every subsequent proportional step would be computed from a
        number that was never real.
        """
        rl = self.rungs
        self.exp = rl[rung_index(max(float(value), 1.0), rl)]
        self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exp)

    def _step_exposure(self, up):
        """Move one rung. Returns False if already at that end."""
        rl = self.rungs
        i = rung_index(self.exp, rl)
        j = i + (1 if up else -1)
        if not 0 <= j < len(rl):
            return False
        self.exp = rl[j]
        self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exp)
        return True

    def _apply_gain(self, value):
        self.gain = int(np.clip(round(value), GAIN_MIN, self.gain_cap))
        self.cap.set(cv2.CAP_PROP_GAIN, self.gain)

    def hold(self, seconds=None):
        """Discard in-flight frames and stay blind briefly.

        update() already does this after every change it makes, but it cannot
        know about changes made from outside - and there is always one: the
        stream has only just started and the controls were applied immediately
        afterwards, because the C920 ignores exposure set before STREAMON. The
        first frames of a fresh stream are therefore exactly the transients the
        blind window exists for.

        Without this, a freshly constructed controller has _blind_until = 0 and
        evaluates the very first frame it is handed. Measured consequence on the
        three-camera rig: a dark start-up frame read as "too dark", the loop
        stepped a whole rung, and luma walked 148 -> 199 -> 255 across scan
        positions until it clipped - a runaway caused entirely by trusting the
        first frame after a stream begins.
        """
        for _ in range(self.flush):
            self.cap.grab()
        self.med_ema = None            # don't average across the transient
        self._blind_until = time.time() + (self.settle if seconds is None
                                           else float(seconds))

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
            # Deadband: don't hunt.
            #
            # An earlier version traded gain for exposure here, to climb off a
            # noisy-but-correct operating point (exposure=71 gain=111: right
            # luma, needless noise). That trade assumed exposure was continuous.
            # It is not - it moves in whole stops - so "exposure * 1.2" snapped
            # straight back to the rung it started on while the gain cut
            # actually took effect. The image got darker, the loop left the
            # deadband, raised gain again, and re-entered: a limit cycle built
            # out of a correction for a limit cycle.
            #
            # The underlying problem is real, though, and a rung-aware version
            # of the trade does work. One rung up is exactly twice the light, so
            # stepping up and roughly halving gain holds brightness while moving
            # noise down a stop. The simulation reaches exp=38 gain=134 on a
            # white wall approached from the stuck-dark state: right luma, a
            # stop and a half of avoidable noise.
            #
            # Two guards keep it from becoming a cycle of its own. It only fires
            # above a gain worth paying a step for, and it fires a bounded number
            # of times per instance - so if halving gain overshoots and the loop
            # climbs back, the trade cannot re-trigger indefinitely. gain // 2 is
            # deliberately approximate: the gain-to-luma curve is not specified
            # by the device, so the loop corrects the remainder rather than
            # pretending to model it.
            if self.gain > self.trade_gain and self._trades < self.max_trades                     and self._step_exposure(up=True):
                self._apply_gain(self.gain // 2)
                self._trades += 1
                for _ in range(self.flush):
                    self.cap.grab()
                self._blind_until = time.time() + self.settle
                return True
            return False

        # Scale the gain step with the error so unwinding a maxed-out gain
        # takes a few steps rather than a few dozen.
        gstep = int(np.clip(abs(err) / max(target, 1.0) * 96, self.gain_step, 96))

        # Exposure is a COARSE actuator here - one stop per rung - so a step is
        # only taken when it does not overshoot the target. Luma is ~linear in
        # exposure, so the next rung up predicts ~med * (next / current); if
        # that lands past the target, the rung is the wrong tool and gain, which
        # is continuous, does the trim instead. Preferring exposure
        # unconditionally is what made the loop oscillate: from luma 99 it
        # stepped 156 -> 312, overshot to 148, then came back down, forever.
        rl = self.rungs
        i = rung_index(self.exp, rl)

        if err < 0:                             # too dark -> brighten
            nxt = rl[i + 1] if i + 1 < len(rl) else None
            predicted = med * (nxt / self.exp) if nxt else None
            if nxt is not None and predicted <= target + self.tol:
                self._step_exposure(up=True)
            elif self.gain < self.gain_cap:
                self._apply_gain(self.gain + gstep)
            elif nxt is not None:
                # Gain is capped and the only rung left overshoots. Overshooting
                # beats staying underexposed: a too-bright frame still carries
                # crack contrast unless it clips, and clip_guard above already
                # pulls the target down when it does.
                self._step_exposure(up=True)
            else:
                return False                    # at both caps; nothing to do
        else:                                   # too bright -> darken
            if self.gain > GAIN_MIN:
                self._apply_gain(self.gain - gstep)
            elif not self._step_exposure(up=False):
                return False                    # already at the shortest rung
            else:
                # Dropping a rung halves the light, so this usually undershoots
                # and the next iteration will raise gain to climb back. That
                # two-step path is the only way to reach a luma that sits
                # between two rungs, and it is why gain must stay available
                # rather than being driven to zero on principle.
                pass

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
