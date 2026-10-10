# 04 — Cameras and capture

*Paper: §III-D multi-camera capture, §III-E photometric control; §IV-D capture
layer; Figs. 6, 7, 16; Tables VI, VII.*

Code: `jetson/capture.py`, `jetson/camera_ctl.py`, `jetson/camera_setup.sh`,
`jetson/multicam_probe.py`.

This is where the project met hardware that did not behave as the datasheet
promised. Almost every section below is *a thing that was assumed, measured, and
found false*.

---

## 1. Why the three images are never stitched

The obvious design: join the three camera feeds into one wide picture, run the
model once. It was rejected on two grounds, both measurable.

### Resolution collapse

The model's input is a fixed 512 px square. Three 1920 px frames side by side
are 5760 px wide.

| | Scale into 512 | A 2-px crack becomes | mm/px at 0.40 m |
|---|---|---|---|
| One camera, native tiles | 1.0 | 2.0 px | 0.29 |
| Stitched 3 → 512 | 512/5760 = **0.089** | **0.18 px** (≈ 0.31 px for a 3328-px mosaic) | ≈ 3.2 |

Below one pixel a crack is not represented at all. The point of three cameras is
to cover more ground *without* losing resolution; stitching spends the
resolution to buy the coverage.

> **Corrected.** Earlier project documents (and the first draft of the paper)
> paired the scale 0.089 with "0.31 px". 0.089 is for the 5760-px mosaic and
> gives 2 × 0.089 = 0.18 px; 0.31 px belongs to the earlier plan's 3328-px mosaic
> of 1280-px frames (2 × 0.154). The paper and `01-camera-weaving.md` now say
> 0.18 px. The conclusion is the same either way: the crack is below one pixel.

### Seams are crack-shaped

A stitch seam is a long, thin, nearly straight intensity discontinuity. That is
the model's target *and* the geometry the shape filter is built to **keep**.
Blending softens it into a soft dark line, which is worse — a soft dark line is a
crack. Feature-based stitching also needs texture, which concrete lacks, so the
seam would *move* between frames and flicker.

### What was done instead

Each camera is processed **independently and completely**. A composite image
exists for human eyes only, and says so on itself:

```
DISPLAY COMPOSITE - detection is per-camera
```

---

## 2. The USB bandwidth wall — Table VI, Fig. 6

### The measurement

`multicam_probe.py` opens cameras **cumulatively** (one, then two, then three)
so the exact failure point is identified, and holds streams open for 60 s
reporting per-camera frame rate.

| Mode | Cameras streaming | Rate each | Source |
|---|---|---|---|
| 640×480 MJPG 15 fps | **3 of 3** | 15.0 fps | MEASURED |
| 1280×720 MJPG 30 fps | **2 of 3** | 15.2 fps | MEASURED |

The third camera at 720p **opens successfully and then never delivers a frame**.

### Why

```
Bus 01  USB 2.0   480 Mbit/s  ──▶ Realtek 4-port hub ──▶ all four Type-A ports
Bus 02  USB 3.0   5 Gbit/s    ──▶ Realtek 4-port hub ──▶ unreachable (C920 is USB 2.0)
```

Every port hangs off one USB 2.0 hub, so the cameras share 480 Mbit/s. UVC
reserves bandwidth from what the firmware *declares*, not what it sends
([02 §7](02-theory-primer.md)). The standard remedy,
`/sys/module/uvcvideo/parameters/quirks`, was already `4294967295` (all bits) —
so the remedy was spent and the third camera still failed.

### Fig. 6

![Fig. 6](figures/fig_usb_bw.png)

**What it shows.** Required bandwidth for 720p30 in two pixel formats.

**How the numbers were obtained.**

```
YUYV = 1280 × 720 × 16 bit × 30 fps = 442,368,000 bit/s = 442.4 Mbit/s    DERIVED
MJPG ≈ 45 Mbit/s       (from perception.html §1)                           MEASURED/ESTIMATED
practical payload ≈ 320 Mbit/s  (rule of thumb for isochronous video)      ASSUMED
```

YUYV exceeds the practical ceiling, so the camera silently negotiates down to
roughly 10 fps — a *silent* failure, discussed in §9. MJPG sidesteps this at the
cost of JPEG quantisation (an 8×8 DCT that discards the high-frequency detail a
hairline lives in). That is a deliberate trade: slow capture causes motion blur,
which destroys more crack evidence than JPEG does.

### Why 640×480 is not an escape

| Resolution | mm/px at 0.40 m | A 0.3 mm crack |
|---|---|---|
| 1920×1080 | 0.29 | ≈ 1 px — marginal |
| 640×480 | 0.88 | 0.34 px — not representable |

Dropping resolution to gain simultaneity trades away what the project exists to
measure.

---

## 3. Sequential capture

**Key insight: the rig is stationary, so simultaneity is not needed.** Three
cameras watching a static wall can be read one after another and the result is
indistinguishable from reading them at the same instant.

```
for each camera in (left, top, right):
    open  →  settle exposure  →  grab & average  →  release
                                                      │
            release frees the bandwidth reservation ──┘ before the next camera asks
```

Only **one** reservation is ever outstanding, so the wall is never reached; all
three deliver full 1080p, every pass.

### What `Camera.open()` does, in order

(`capture.py:143`) — every step exists because a shortcut failed.

| # | Step | Why |
|---|---|---|
| 1 | `VideoCapture(index, CAP_V4L2)` | Use V4L2 directly |
| 2 | Set **FOURCC = MJPG first**, then width/height/fps | Set after size, the driver has already committed to YUYV and silently caps ~10 fps |
| 3 | Read up to 20 frames until one arrives | Opening is lazy: bandwidth is reserved at the first read. A camera is not proven until it hands over a frame |
| 4 | Run `camera_setup.sh` | Puts the camera in **manual** exposure — *only possible once streaming* |
| 5 | Create `AutoExposure(exp_cap=700)` | Remembered exposure/gain re-asserted |
| 6 | `ae.hold()` | Declare a blind window (see §6) |

Step 4 once went missing. Symptom: the loop's exposure number drifted 139 → 123
→ 110 across passes while the measured luma stayed **exactly 133.0**. The
firmware was in auto mode and silently discarding every write; the loop was
winding a number connected to nothing. *A controller whose actuator is
disconnected looks like it works right up until you check the output moved.*

### The cost of one scan position — Fig. 16

![Fig. 16](figures/fig_capture_timeline.png)

**What it shows.** One scan position as a timeline: three sequential camera
captures (each four segments), then inference.

**How the values were obtained** (MEASURED with `capture.py bench`, clocks
locked, 4-frame averaging):

| Segment | Per camera | ×3 |
|---|---|---|
| open + STREAMON | 0.99 s | 2.97 s |
| exposure confirm | 0.53 s | 1.59 s |
| grab + average 4 | 0.22 s | 0.66 s |
| **release** | **0.24 s** | **0.71 s** |
| **capture total** | 1.98 s | **5.93 s** |
| inference (15 tiles) | 0.22 s | 0.66 s |
| **scan position** | | **≈ 6.6 s** |

> **A discrepancy in the source docs.** `docs/phase2/01-camera-weaving.md`'s
> per-camera table omits the `release` row, so its parts sum to 5.22 s while it
> states 5.93 s. The missing 0.71 s is exactly 3 × 0.238 s (release was measured
> separately at 238 ms). The figure and paper use the complete breakdown.

**What dominates.** Capture is 5.93 / 6.59 = **90 %** of the cycle. The first
frame after STREAMON alone costs **675 ms** (sensor start + bus negotiation), paid
three times: ≈ 2 s that cannot be removed without keeping streams open — which is
exactly what the bandwidth wall forbids. It is the *price of three cameras on one
USB 2.0 bus*, not an inefficiency.

---

## 4. The exposure ladder — Fig. 7

![Fig. 7](figures/fig_exposure_ladder.png)

### How it was found

The control advertises `exposure_time_absolute` 3…2047, step 1. The observation:

- **No stream open:** the driver accepts all **747** values.
- **Streaming at 1080p:** it does not. Writing every value 5…899 and reading back
  what the camera settled on, the only values that ever came back were

```
38    77    156    312    624          (each exactly twice the previous)
```

The value `exposure_time_absolute` is in **100 µs** units, so these are
3.8, 7.7, 15.6, 31.2 and 62.4 ms.

### How Fig. 7 is drawn

The dashed grey diagonal is "what you asked for". The blue staircase is "what
you got", drawn by **snapping each request to the nearest rung in log space** —
the same rule as `camera_ctl.rung_index`. So the figure is a *model of the
measured behaviour*, not a plot of raw readbacks; the five red dots are the
measured rungs.

> **Honest footnote.** `camera_ctl.EXP_RUNGS` is
> `(19, 38, 77, 156, 312, 624, 1250, 2047)`. Only 38…624 were observed; 19, 1250
> and 2047 are extrapolated (and 1250/2047 are above the working cap anyway).

### Why log space

Between 156 and 312, linear distance calls 230 "nearer" to 156 by a hair, but in
photographic stops it is almost exactly halfway. Brightness scales with the
*ratio*, so the metric is $|\ln(\text{value}/\text{rung})|$.

### Why this broke the loop, and why "drifting" was not drift

Median luma can only move by about a factor of two per step. On the test scene
it could be **99 or 148, never 120**. (148/99 = 1.5, not 2: the sensor + gamma +
JPEG response is not linear, so a stop is not exactly ×2 in luma.) A loop aimed
at 120 oscillated forever. An even earlier bug — "exposure drifted 156 → 312 on
stream start" — was the same fact: 156 and 312 are *adjacent rungs*, not a small
drift.

---

## 5. The controller, step by step

`camera_ctl.AutoExposure`. Parameters (defaults): `target=120`, `tol=30`,
`interval=0.4 s`, `exp_cap=500` (capture uses **700**), `gain_cap=160`,
`gain_step=16`, `trade_gain=48`, `max_trades=4`, `flush=5`, `settle=0.5 s`.

### The decision in one picture

```
              median luma (smoothed)
                    │
        ┌───────────┴───────────┐
   |err| ≤ 30 ?                 no
        │ yes                    │
   (deadband: maybe trade   too dark ?──── yes ──▶ next rung's PREDICTED luma ≤ target+tol ?
    gain for exposure)                                   │ yes → step exposure UP
                                                         │ no  → raise GAIN by gstep
                                                    no (too bright)
                                                         └──▶ lower GAIN first;
                                                              only at gain 0, drop a rung
```

### Worked examples

Target 120, tolerance 30, so the dead-band is **[90, 150]**.

| State | Luma | Decision | Why |
|---|---|---|---|
| exp 156, gain 0 | 99 | **nothing** | inside [90,150] |
| exp 77, gain 0 | 40 | **step up to 156** | predicted 40×156/77 = 81 ≤ 150, no overshoot |
| exp 156, gain 0 | 80 | **gain +32** | next rung predicts 80×2 = 160 > 150 → would overshoot, so use gain; gstep = clip(\|−40\|/120×96, 16, 96) = 32 |
| exp 156, gain 0 | 190 | **drop a rung → 77** | too bright, gain already 0 |
| exp 312, gain 80 | 200 | **gain − gstep** | too bright → gain goes down first |

### Why each rule exists

| Rule | The failure it prevents |
|---|---|
| Control the **median**, target 120 | A p99 = 235 target is unreachable on a bimodal scene and limit-cycled 311↔532 with 4.8 % clipping |
| **No one-way ratchet** | An "anti-windup ceiling" that only decreased fell below the exposure *floor* and drove the camera to exp 3, gain 255 — darkest and noisiest, unrecoverable |
| Brighten = exposure first, **gain only at max**; darken = **gain first** | Makes (exp 3, gain 255) unreachable by construction |
| **Snap** to rungs | Otherwise the controller's record ≠ the hardware's state |
| Step only if **prediction ≤ target + tol** | Preferring exposure unconditionally stepped 156→312, overshot to 148, came back, forever |
| tol 8 → **30** | A tolerance finer than the actuator forces gain to fill every gap; the simulation settled at gain 93–111 — and gain *is* the false-positive signature |
| **Blind window** (flush 5 frames, 0.5 s) after every change | At a *fixed* exp 500 gain 0, consecutive medians were **4 and 241**: the C920 emits black/blown frames after any control write |
| **EMA** on the metric (α = 0.5) | Smooths what remains |
| **`hold()`** after open | A fresh controller evaluated the first post-STREAMON frame, a known transient, stepped a rung on it, and luma walked 148 → 199 → 255 until it clipped |
| Rung-aware **gain↔exposure trade** (`trade_gain=48`, `max_trades=4`) | The first trade multiplied exposure by 1.2, which snapped back to the same rung while the gain cut took effect — a limit cycle inside the fix for a limit cycle |
| `clip_guard`: if > 5 % of pixels ≥ 254, aim at 0.8 × target | Highlights are costing crack contrast now |

---

## 6. Table VII — the simulation, and what it can and cannot say

| Scenario | Exp | Gain | Luma | Result |
|---|---|---|---|---|
| Mid-grey, from default | 247 | 0 | 126.0 | pass |
| Stuck dark (exp 3, gain 255) | 232 | 0 | 118.3 | pass |
| Stuck bright | 239 | 0 | 121.9 | pass |
| White wall | 128 | 0 | 124.0 | pass |
| White wall, from stuck dark | 125 | 0 | 121.1 | pass |
| Dark asphalt | 500 | 44 | 117.3 | pass |
| Very dark scene | 500 | 160 | 88.4 | pass (at caps) |

**Source.** `jetson/test_ae_sim.py` — 7/7. **MEASURED** in WORKLOG §8; the paper
quotes the table (§IV-D).

**Why a simulation at all.** With a person moving in frame the median swings
19 ↔ 178 *at fixed settings*, so live pass/fail is noise. The test drives the
*real controller* against a modelled sensor with actuation lag and garbage
frames — deterministic.

**Caveat to state when you quote it.** This table is from the *first* control
law (continuous exposure, cap 500), before the ladder was discovered. It shows
the control logic is sound; it does not model a quantised actuator. The result
for the ladder-aware controller is the **hardware** one:

> All three cameras, every pass: **exp = 156, gain = 0**, luma 94–99, converged,
> **0.2 %** of pixels clipped on a white ceiling with a blown-out window in view.

---

## 7. Averaging and settling

### Frame averaging (`Camera.grab`)

Takes `average=4` **independent** frames, averages in float32, rounds to uint8.
Noise falls by $\sqrt4 = 2$ ([02 §9](02-theory-primer.md)).

*Independence is enforced.* With `BUFFERSIZE=1`, reading faster than the camera
produces returns the same frame twice. Duplicates are detected on a 1-in-16
subsampled grid (`frame[::16, ::16]`) and re-read.

### Two-phase settling (`SequentialRig`)

| Phase | When | Budget | Evaluations required |
|---|---|---|---|
| **prime** | once per camera per survey | 8 s | 2 in-band |
| **confirm** | every later position | 1.2 s | 1 in-band |

Why: a flat 2.5 s budget hit its cap at *every* position — 7.5 s of an 11.7 s
cycle — because the loop can step only once per `interval` plus its blind window,
so convergence from cold takes several seconds however long any one position
waits. Exposure and gain are **remembered across open/close**, so after priming
the loop only has to *verify*.

A "not converged" false alarm was fixed along the way: `hold()`'s 0.5 s blind
window plus two 0.4 s evaluations = 1.3 s, which a 0.8 s confirm could never
satisfy. Confirm now needs one evaluation.

> **Validity.** Both phases are sound *only* because the rig is stationary and
> lighting is near-constant over a survey. On a moving platform or under passing
> cloud they would have to become a per-position budget again.

---

## 8. Role mapping — never trust `/dev/videoN`

`/dev/video0, 2, 4` are assigned in enumeration order and **shuffle on reboot or
replug**. If left and right swap silently, the output is a plausible but
*mirrored* mosaic — worse than an error, because nothing fails.

```
/dev/v4l/by-path/platform-3610000.usb-usb-0:2.1:1.0-video-index0 → /dev/video2
/dev/v4l/by-path/platform-3610000.usb-usb-0:2.2:1.0-video-index0 → /dev/video0
/dev/v4l/by-path/platform-3610000.usb-usb-0:2.4:1.0-video-index0 → /dev/video4
```

`by-path` names the **physical hub port**, so a role follows the *bracket*: a
replaced camera in the left mount is still `left`. (`by-id` follows the camera's
serial — wrong for this purpose, so it is kept for diagnostics only.) Each C920
exposes a second `-index1` metadata node that opens but never yields a frame;
only `-index0` is matched.

One-time setup: `python3 capture.py identify` writes a labelled frame per camera;
the mapping goes in `jetson/rig.json`. **That file does not yet exist** — the
cameras were unplugged at last check, so roles currently fall back to port
order (a guess).

---

## 9. Live preview mode

A scan position takes ≈ 6.6 s, so a stream built from scans publishes one image
per 6.6 s — about nine a minute. That is not lag; there are simply that many
frames.

Phase 0 measured the other end of the trade: **at 640×480 all three cameras
stream together at 15 fps indefinitely.** So `--mode live` opens each camera once
and holds it open.

| | scan | live |
|---|---|---|
| Resolution | 1920×1080 native tiles | 640×480, centre-cropped square, resized to 512 |
| Rate | one per ≈ 6.6 s | **10 fps** (measured) |
| Inference | 223 ms/camera, 15 tiles | 15 ms/camera, 1 tile |
| mm/px at 0.4 m | 0.29 | 0.83 |
| Smallest crack | ≈ 0.59 mm | ≈ 1.65 mm |

Both modes publish `widest_mm`, `coverage` and `components` under the **same
names**, but they are **not comparable**. The mode is therefore stated in four
places on the console (header pill, stage caption, status payload, and the
"scan cycle"/"frame rate" row label). *Live is for aiming; a scan is for
measuring.*

---

## 10. The failure catalogue

The recurring lesson: **a component reports success while being wrong.**

| Symptom | Actual cause | Stage |
|---|---|---|
| Frame rate silently ~10 fps | FOURCC set after resolution | open |
| Exposure "locked" yet reverts 156 → 312 | Firmware leaves manual mode at STREAMON | open |
| Loop number drifts, luma constant at 133.0 | `camera_setup.sh` skipped → auto mode discards writes | open |
| Exposure oscillates between two values | Setpoint between rungs | control |
| Controller thrashes | Reacting to transient frames (4 and 241) | control |
| Luma walks 148 → 199 → 255 | Evaluated the first frame after STREAMON | control |
| White wall → black, never recovers | Anti-windup ceiling fell below the floor | control |
| Noisy at correct brightness | Converged to exp 71, gain 111 | control |
| Mirror-image mosaic | `/dev/videoN` reshuffled | roles |
| Everything 2.6× slower | `jetson_clocks` lost on reboot (306 MHz) | power |

---

## 11. Reproducing

```bash
# on the Jetson
python3 multicam_probe.py          # the bandwidth measurement
python3 capture.py identify        # which port is which
python3 capture.py bench           # the cost of a scan position (Fig. 16)
python3 capture.py sweep           # open-loop exposure -> median-luma curve (how the ladder showed up)
python3 test_ae_sim.py             # 7/7 simulated scenarios (Table VII)
```

From Windows: `cracknet check` reports the camera count and GPU clock before
anything else. See [10](10-reproduce-and-extend.md).
