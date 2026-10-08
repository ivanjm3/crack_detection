#!/usr/bin/env python3
"""Sequential multi-camera capture for the stationary three-camera rig.

WHY SEQUENTIAL
--------------
Phase 0 measured the bus directly (multicam_probe.py, 2026-10-07). Every Type-A
port on this Orin Nano hangs off one Realtek USB 2.0 hub on Bus 01, so all three
C920s share 480 Mbit/s. Measured on this board:

    3 cameras @ 640x480  MJPG 15 fps  -> all three stream, 15.0 fps each
    3 cameras @ 1280x720 MJPG 30 fps  -> the THIRD camera opens but never
                                         delivers a frame

That is UVC isochronous bandwidth reservation, not a real data-rate limit: the
reservation is computed from the figure the camera firmware declares, not from
what it actually sends. The uvcvideo bandwidth quirk is already active on this
board (quirks = 0xFFFFFFFF) and the third camera still fails, so the usual
remedy is already spent.

Dropping to 640x480 to fit three streams is not an option. The resolution budget
is the whole point of the design: at the planned standoff 1080p gives
0.29 mm/px, while 640x480 gives 0.88 mm/px - a 0.3 mm crack stops being
representable at all. Resolution is what we are buying; spending it to gain
simultaneity would trade the goal for the means.

And simultaneity is not needed. The rig is STATIONARY: nothing in the scene
moves while a scan position is captured. Three cameras looking at a static
surface can be read one after another and the result is indistinguishable from
reading them at the same instant. Opening a camera, capturing, then releasing it
frees the bandwidth reservation before the next camera asks for one, so only one
reservation is ever outstanding and the wall is never reached.

The cost of that choice is open latency, paid once per camera per scan position,
which is what `bench` measures.

ROLE MAPPING
------------
Roles come from /dev/v4l/by-path, not from /dev/videoN. The videoN numbers are
assigned in device-enumeration order and shuffle on reboot or replug; if left
and right silently swap, the ground-frame merge yields a plausible-looking
mosaic that is mirrored, which is much worse than an error. A by-path symlink
names the physical hub port, so a role tracks the BRACKET: if a camera dies and
is replaced, whatever sits in the left mount is still "left". (by-id names the
camera's own serial, which would follow the hardware to a different mount - the
wrong behaviour here, so it is recorded for diagnostics only.)

Run `identify` once to find out which port is which; it writes a labelled frame
per camera so the mapping can be checked by eye. That is the only step that
needs a human.

    python3 capture.py identify            # who is where
    python3 capture.py bench               # cost of a scan position
    python3 capture.py capture --out scan  # one scan position to disk
"""
import argparse
import glob
import json
import os
import subprocess
import time

import cv2
import numpy as np

from camera_ctl import AutoExposure

HERE = os.path.dirname(os.path.abspath(__file__))
RIG_JSON = os.path.join(HERE, "rig.json")
SETUP_SH = os.path.join(HERE, "camera_setup.sh")
ROLES = ("left", "top", "right")


def discover():
    """Capture-capable cameras, keyed by stable port path.

    Each C920 exposes two video nodes; the -index1 node is a metadata node that
    opens cleanly and then never yields a frame, so matching only -index0 avoids
    counting every camera twice.
    """
    serial_of = {}
    for link in glob.glob("/dev/v4l/by-id/*-video-index0"):
        serial_of[os.path.realpath(link)] = os.path.basename(link)

    found = []
    for link in sorted(glob.glob("/dev/v4l/by-path/*-video-index0")):
        node = os.path.realpath(link)
        found.append({
            "port": os.path.basename(link),
            "dev": node,
            "index": int("".join(c for c in os.path.basename(node) if c.isdigit())),
            "serial": serial_of.get(node, "unknown"),
        })
    return found


def load_rig():
    """role -> camera dict. Falls back to port order when rig.json is absent."""
    cams = discover()
    if not cams:
        raise RuntimeError("no cameras found under /dev/v4l/by-path")

    by_port = {c["port"]: c for c in cams}
    if os.path.exists(RIG_JSON):
        with open(RIG_JSON) as f:
            mapping = json.load(f)
        rig = {}
        for role, port in mapping.items():
            if port not in by_port:
                raise RuntimeError(
                    "rig.json maps %s to %s, which is not plugged in.\npresent: %s"
                    % (role, port, ", ".join(sorted(by_port))))
            rig[role] = by_port[port]
        return rig

    # Port order is a guess, not a mapping. It is deterministic, so it is a
    # usable default for timing work, but it must be confirmed with `identify`
    # before any geometry depends on it.
    return {role: cam for role, cam in zip(ROLES, cams)}


class Camera:
    """One C920, opened and released around each capture.

    Exposure and gain are remembered across open/close. Reconverging the
    exposure loop from scratch at every scan position would cost seconds per
    camera, and worse, would let a converged camera drift to a different
    operating point between positions - so two frames of the same pavement taken
    minutes apart would not be comparable. Lighting over a stationary survey is
    near-constant, so carrying the converged values forward and letting the loop
    only trim them is both faster and more consistent.
    """

    def __init__(self, role, cam, width=1920, height=1080, fps=30, verbose=False):
        self.role = role
        self.info = cam
        self.w, self.h, self.fps = width, height, fps
        self.verbose = verbose
        self.exposure, self.gain = 156, 0      # camera_setup.sh starting point
        self.cap = None
        self.ae = None
        self.primed = False

    def open(self):
        cap = cv2.VideoCapture(self.info["index"], cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError("%s: cannot open %s" % (self.role, self.info["dev"]))
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # before size
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Opening is lazy: V4L2 reserves isochronous bandwidth at STREAMON, which
        # OpenCV triggers on the first read. A camera is therefore not proven
        # until it has actually handed over a frame. These reads are also what
        # gets the stream running before controls are re-applied - the C920
        # firmware drops out of manual mode during stream init and ignores
        # anything set before that point (see camera_ctl.py).
        got = False
        for _ in range(20):
            ok, frame = cap.read()
            if ok and frame is not None:
                got = True
                break
            time.sleep(0.05)
        if not got:
            cap.release()
            raise RuntimeError(
                "%s: opened %s but got no frame - another camera is probably "
                "still streaming (bandwidth)" % (self.role, self.info["dev"]))

        # Put the camera into MANUAL exposure, and only now that the stream is
        # running. Without this the C920 stays in firmware auto-exposure, where
        # the driver silently discards every CAP_PROP_EXPOSURE write: the first
        # version of this file omitted it and the symptom was subtle rather than
        # loud - the loop's own exposure number drifted 139 -> 123 -> 110 across
        # three passes while the measured luma stayed at exactly 133.0, because
        # the firmware was holding the image steady and the loop was winding a
        # number that reached nothing. A control loop whose actuator is
        # disconnected looks like it is working right up until you check that
        # the output moved.
        if os.path.exists(SETUP_SH):
            subprocess.run(["bash", SETUP_SH, self.info["dev"]],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        self.cap = cap
        # camera_setup.sh pins exposure to its own default, so the values carried
        # over from the previous scan position have to be re-asserted after it.
        # exp_cap is raised past live.py's 500 because the cap exists to bound
        # MOTION BLUR, and this rig does not move. 500 admits a top rung of 312
        # (exposure snaps to a x2 ladder - see camera_ctl.EXP_RUNGS); 700 admits
        # 624, a full stop more light, which the loop then does not have to buy
        # with gain. On dark asphalt that is the difference between gain ~106
        # and gain near zero, and gain is what turns read noise into the thin
        # bright streaks the crack model mistakes for cracks.
        self.ae = AutoExposure(cap, exposure=self.exposure, gain=self.gain,
                               exp_cap=700, verbose=self.verbose)
        cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
        cap.set(cv2.CAP_PROP_GAIN, self.gain)
        # Those two writes, plus camera_setup.sh above, plus the stream itself
        # having just started, are three control changes in a row. Declare the
        # blind window so the loop does not converge on their transients.
        self.ae.hold()
        return cap

    def close(self):
        if self.cap is not None:
            self.exposure, self.gain = self.ae.exp, self.ae.gain
            self.cap.release()
            self.cap, self.ae = None, None

    def settle(self, max_seconds=2.5, stable=2, min_seconds=0.3):
        """Let the exposure loop trim the remembered values to this scene.

        Adaptive, not a fixed wait. Because exposure carries over from the
        previous scan position, the usual case is that nothing needs changing
        and the loop is already in its deadband - so this returns as soon as the
        smoothed median has been in band for `stable` consecutive evaluations, and
        only burns the full budget when the scene genuinely changed. A fixed
        1.2 s wait per camera was 3.7 s of the 7.3 s scan position, almost all
        of it spent confirming an answer already known.

        min_seconds exists because the first reads after STREAMON are not
        trustworthy: the sensor needs several frames to honour a new exposure
        and the C920 emits garbage in between (medians of 4 and 241 at identical
        settings, see camera_ctl.py), so converging on them would mean
        converging on a transient.

        Convergence is counted in AE EVALUATIONS, not in frames read. The loop
        self-throttles to ae.interval (0.4 s) and the stats it exposes only
        refresh when it actually evaluates, so counting frames would let three
        reads taken off one stale snapshot end the settle immediately. Two fresh
        in-band evaluations is ~0.4 s of real confirmation.
        """
        t0 = time.perf_counter()
        frames, good = 0, 0
        seen = None                    # ae._last changes on every evaluation
        while True:
            elapsed = time.perf_counter() - t0
            if elapsed >= max_seconds:
                return frames, False
            ok, frame = self.cap.read()
            if not ok or frame is None:
                continue
            self.ae.update(frame)
            frames += 1
            if elapsed < min_seconds or self.ae._last == seen:
                continue
            seen = self.ae._last
            med = self.ae.stats["median"]
            # Ignore the blind window after a control change: the stats still
            # describe the pre-change image, so counting that as "in band" would
            # end the settle on stale evidence.
            if abs(med - self.ae.target) <= self.ae.tol \
                    and time.time() >= self.ae._blind_until:
                good += 1
                if good >= stable:
                    return frames, True
            else:
                good = 0

    def grab(self, average=4):
        """Average `average` independent frames into one uint8 BGR image.

        Averaging N frames of a static scene cuts temporal sensor noise by
        sqrt(N) without touching spatial detail, which matters here because the
        model responds to thin dark filaments and read noise produces exactly
        that.

        The frames must be INDEPENDENT. With BUFFERSIZE 1 a read returns the
        newest buffered frame, so reading faster than the camera produces hands
        back the same frame twice and the average is of nothing. Duplicates are
        detected on a subsampled grid and re-read, rather than assumed away.
        """
        acc = None
        taken, dupes, last = 0, 0, None
        deadline = time.perf_counter() + 2.0 + 0.2 * average
        while taken < average and time.perf_counter() < deadline:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                continue
            if last is not None and frame.shape == last.shape and \
                    np.array_equal(frame[::16, ::16], last[::16, ::16]):
                dupes += 1
                time.sleep(1.0 / max(self.fps, 1))
                continue
            acc = frame.astype(np.float32) if acc is None else acc + frame
            last = frame
            taken += 1
        if acc is None:
            raise RuntimeError("%s: no frames to average" % self.role)
        return np.rint(acc / taken).astype(np.uint8), taken, dupes


class SequentialRig:
    """Open-capture-release each camera in turn; one scan position per call.

    Exposure settling is split into two phases, because measurement showed the
    one-phase version was paying the wrong cost. With a flat 2.5 s budget per
    camera the loop hit the cap at EVERY position without converging - 7.5 s of
    an 11.7 s scan position - because the loop can only take a step every
    ae.interval plus its blind window, so convergence from a cold start needs
    several seconds no matter how long any single position waits.

      prime   once per camera per survey, with a generous budget: converge from
              the camera_setup.sh default to this site's lighting.
      confirm every position after that, with a short budget: the carried-over
              exposure is already right, so this only has to verify that the
              median is still in band. If the light genuinely changed it will
              not converge in the short budget, which is reported rather than
              hidden, and the loop keeps trimming across the next positions.

    This is only sound because the rig is stationary and lighting over a survey
    is near-constant. On a moving platform, or outdoors with passing cloud, the
    prime/confirm split would have to become a per-position budget again.
    """

    def __init__(self, width=1920, height=1080, average=4, settle=1.2,
                 prime=8.0, verbose=False):
        self.cams = [Camera(role, cam, width, height, verbose=verbose)
                     for role, cam in load_rig().items()]
        self.average = average
        self.settle_s = settle
        self.prime_s = prime

    def capture(self):
        """role -> (frame, stats). Only ever one camera streaming at a time."""
        out = {}
        for cam in self.cams:
            t0 = time.perf_counter()
            cam.open()
            t_open = time.perf_counter() - t0

            t0 = time.perf_counter()
            # The confirm phase asks for ONE in-band evaluation, the prime
            # phase for two. Priming is deciding where to sit and deserves
            # corroboration; confirming is checking that a value already known
            # to be right is still right, and ae.hold() has just reset the
            # smoothing, so that single reading is a fresh measurement rather
            # than a smoothed guess. Demanding two here only bought a longer
            # budget: hold()'s blind window plus two evaluations is 1.3 s, so a
            # 0.8 s confirm could never satisfy it and every position reported
            # "not converged" while sitting at a perfectly stable exposure.
            if cam.primed:
                _frames, converged = cam.settle(self.settle_s, stable=1)
            else:
                _frames, converged = cam.settle(self.prime_s, stable=2)
            cam.primed = True
            t_settle = time.perf_counter() - t0

            t0 = time.perf_counter()
            frame, taken, dupes = cam.grab(self.average)
            t_grab = time.perf_counter() - t0

            stats = {"open": t_open, "settle": t_settle, "grab": t_grab,
                     "converged": converged,
                     "frames": taken, "dupes": dupes,
                     "exposure": cam.ae.exp, "gain": cam.ae.gain,
                     "median_luma": float(np.median(
                         cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))}
            cam.close()
            out[cam.role] = (frame, stats)
        return out

    def close(self):
        for cam in self.cams:
            cam.close()


def cmd_identify(args):
    cams = discover()
    print("%d capture-capable camera(s)\n" % len(cams))
    os.makedirs(args.out, exist_ok=True)
    for cam in cams:
        print("  %s" % cam["port"])
        print("      node   %s" % cam["dev"])
        print("      serial %s" % cam["serial"])
        c = Camera(cam["port"], cam, args.width, args.height)
        try:
            c.open()
            c.settle(8.0)
            frame, _, _ = c.grab(1)
        finally:
            c.close()
        cv2.putText(frame, cam["port"], (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 255), 2)
        path = os.path.join(args.out, "identify_%s.jpg" % cam["port"])
        cv2.imwrite(path, frame)
        print("      wrote  %s\n" % path)
    print("Look at the images, then write the mapping to rig.json, e.g.")
    print(json.dumps({r: c["port"] for r, c in zip(ROLES, cams)}, indent=2))


def cmd_sweep(args):
    """Measure the exposure -> median-luma curve on one camera, open loop.

    This exists because the closed loop would not converge and the useful
    question was not "which gain do I tune" but "what does the actuator
    actually do". It holds the scene fixed, steps exposure over a range, waits
    for the sensor to honour each value, and prints the median luma. That gives
    three things no amount of loop tuning would: whether the response is
    monotonic, how many frames a change takes to appear, and whether the target
    luma is reachable at all on this scene.
    """
    cams = discover()
    cam = Camera("sweep", cams[args.camera], args.width, args.height)
    cap = cam.open()
    print("exposure  median luma   (settling %d frames per step)\n" % args.flush)
    try:
        for exp in range(args.lo, args.hi + 1, args.step):
            cap.set(cv2.CAP_PROP_EXPOSURE, exp)
            for _ in range(args.flush):          # let the sensor catch up
                cap.read()
            meds = []
            for _ in range(3):
                ok, frame = cap.read()
                if ok and frame is not None:
                    meds.append(float(np.median(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))))
            # Three readings, not one: a single median cannot distinguish a
            # settled value from a transient, and the spread is the evidence
            # for how long settling really takes.
            print("%8d  %s" % (exp, "  ".join("%5.1f" % m for m in meds)))
    finally:
        cam.close()


def cmd_bench(args):
    rig = SequentialRig(args.width, args.height, args.average, args.settle,
                        args.prime)
    print("%d camera(s) at %dx%d, averaging %d frames\n"
          % (len(rig.cams), args.width, args.height, args.average))
    for n in range(args.repeat):
        t0 = time.perf_counter()
        res = rig.capture()
        total = time.perf_counter() - t0
        print("pass %d" % (n + 1))
        for role, (frame, s) in res.items():
            print("  %-6s open %6.0f ms  settle %6.0f ms  grab %6.0f ms  "
                  "(%d frames, %d dup)  exp %4d gain %3d  luma %5.1f%s"
                  % (role, s["open"] * 1000, s["settle"] * 1000,
                     s["grab"] * 1000, s["frames"], s["dupes"],
                     s["exposure"], s["gain"], s["median_luma"],
                     "" if s["converged"] else "  (not converged)"))
        print("  %-6s %6.0f ms per scan position\n" % ("total", total * 1000))
    rig.close()


def cmd_capture(args):
    os.makedirs(args.out, exist_ok=True)
    rig = SequentialRig(args.width, args.height, args.average, args.settle,
                        args.prime)
    res = rig.capture()
    rig.close()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for role, (frame, s) in res.items():
        # PNG, not JPEG: averaging frames to remove noise and then re-quantising
        # through a lossy codec would put a different kind of noise back.
        path = os.path.join(args.out, "%s_%s.png" % (stamp, role))
        cv2.imwrite(path, frame)
        print("%-6s %s  exp %d gain %d luma %.1f%s"
              % (role, path, s["exposure"], s["gain"], s["median_luma"],
                 "" if s["converged"] else "  (not converged)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("identify", "bench", "capture", "sweep"))
    ap.add_argument("--out", default="scan")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--average", type=int, default=4)
    ap.add_argument("--prime", type=float, default=8.0,
                    help="one-off exposure convergence budget per camera")
    ap.add_argument("--settle", type=float, default=1.2,
                    help="upper bound on exposure settling per camera")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--camera", type=int, default=0, help="sweep: which camera")
    ap.add_argument("--lo", type=int, default=50)
    ap.add_argument("--hi", type=int, default=500)
    ap.add_argument("--step", type=int, default=50)
    ap.add_argument("--flush", type=int, default=10)
    args = ap.parse_args()
    {"identify": cmd_identify, "bench": cmd_bench,
     "capture": cmd_capture, "sweep": cmd_sweep}[args.mode](args)


if __name__ == "__main__":
    main()
