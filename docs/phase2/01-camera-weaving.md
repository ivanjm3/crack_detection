# Camera weaving — getting three C920s to produce usable frames

Chronological. Each section is a problem that was hit, measured, and solved.

---

## 1. The original idea, and why it was rejected

**Proposal:** mount three C920s (left, top, right), stitch their feeds into one
wide image, send that to the model.

**Rejected on two grounds.**

### Resolution collapse

The model takes a fixed 512×512 input. Stitching three 1920-wide frames gives a
5760-wide image. Squeezing that into 512 is a scale factor of 0.089.

| | mm per pixel at the model |
|---|---|
| One 1080p camera, native tiles | 0.29 |
| Three cameras stitched into 512 | ~3.2 |

A 2 px crack becomes 0.31 px — it stops existing. The whole reason for three
cameras is to cover more ground *without* losing resolution, so a design that
spends the resolution to gain the coverage defeats itself.

### Seams look exactly like cracks

A stitch seam is a long, thin, roughly straight intensity discontinuity. That is
the precise description of what the model fires on and what the shape filter is
built to *keep*. Blending softens it into a soft dark line, which is worse — a
soft dark line is a crack.

### What was done instead

Each camera is tiled and inferred **independently and completely**. Nothing is
stitched before inference. A composite is built afterwards, for display only,
and it is labelled on the image itself:

```
DISPLAY COMPOSITE - detection is per-camera
```

---

## 2. Stationary scanning

The rig does not move while a position is captured. This was chosen because a
moving platform was not available, but it turns out to remove most of the hard
constraints:

| Constraint on a moving rig | On a stationary rig |
|---|---|
| Motion blur caps exposure time | No limit; long exposures are free |
| Cameras must be synchronised | Irrelevant — the scene is static |
| Frame rate sets ground speed | No frame-rate requirement at all |
| Frames must not be averaged (subject moves) | Averaging is free accuracy |

The last two matter most, and the first enables the exposure fix in §6.

---

## 3. The USB bandwidth wall — measured, not assumed

Every Type-A port on this Orin Nano hangs off **one Realtek USB 2.0 hub on Bus
01**, so all three cameras share 480 Mbit/s.

Measured with `multicam_probe.py`, which opens cameras cumulatively so the exact
failure point is identified:

| Mode | Cameras streaming | Rate each |
|---|---|---|
| 640×480 MJPG 15 | 3 of 3 | 15.0 fps |
| 1280×720 MJPG 30 | **2 of 3** | 15.2 fps |

The third camera at 720p **opens successfully and then never delivers a frame**.

### Why it fails, and why the usual fix was already spent

This is UVC *isochronous bandwidth reservation*. The driver reserves bus
bandwidth from the figure the camera firmware **declares**, not from what it
actually sends. A C920 declares a worst case far above its real MJPEG rate.

The standard remedy is the uvcvideo bandwidth quirk. On this board:

```
/sys/module/uvcvideo/parameters/quirks  =  4294967295   (0xFFFFFFFF)
```

Every quirk is already enabled, including `UVC_QUIRK_FIX_BANDWIDTH`. The
remedy was already applied and the third camera still fails.

### Why 640×480 is not an escape

| Resolution | mm/px at 0.40 m |
|---|---|
| 1920×1080 | 0.29 |
| 640×480 | 0.88 |

A 0.3 mm crack is not representable at 0.88 mm/px. Dropping resolution to gain
simultaneity trades away the thing the project exists to measure.

---

## 4. Sequential capture

**The rig is stationary, so simultaneity is not needed.** Three cameras looking
at a static surface can be read one after another and the result is
indistinguishable from reading them at the same instant.

```
for each camera:
    open  →  settle exposure  →  grab & average  →  release
```

Releasing frees the bandwidth reservation before the next camera asks for one,
so **only one reservation is ever outstanding** and the wall is never reached.
All three deliver full 1080p this way, every pass.

### What it costs

Per camera, measured directly:

| Step | ms |
|---|---|
| **First frame after STREAMON** | **675** |
| `release()` | 238 |
| Flush after control writes | 152 |
| `VideoCapture()` + format calls | 55 |
| `camera_setup.sh` | 30 |
| **Total** | **~1150** |

675 ms is the camera's own startup — sensor start plus bus negotiation — and it
is paid three times per scan position. That ~2.7 s is irreducible without
keeping streams open, which is exactly what the bandwidth wall forbids.

Full cycle, steady state:

| Stage | Per camera | ×3 |
|---|---|---|
| open + stream start | 0.99 s | 2.97 s |
| exposure confirm | 0.53 s | 1.59 s |
| grab + average 4 frames | 0.22 s | 0.66 s |
| **capture total** | | **5.93 s** |
| inference | 0.22 s | 0.66 s |
| **scan position** | | **~6.5 s** |

**Capture is ~90% of the cycle.** Inference is not the bottleneck.

---

## 5. Role mapping — never trust `/dev/videoN`

`/dev/video0,2,4` are assigned in device-enumeration order and **shuffle on
reboot or replug**. If left and right silently swap, the output is a
plausible-looking mosaic that is mirrored — far worse than an error, because
nothing fails.

Roles are mapped through `/dev/v4l/by-path`, which names the **physical hub
port**:

```
platform-3610000.usb-usb-0:2.1:1.0-video-index0  →  /dev/video2
platform-3610000.usb-usb-0:2.2:1.0-video-index0  →  /dev/video0
platform-3610000.usb-usb-0:2.4:1.0-video-index0  →  /dev/video4
```

A role therefore tracks the **bracket**, not the hardware: if a camera dies and
is replaced, whatever sits in the left mount is still `left`. (`by-id` names the
camera's serial, which would follow the hardware to a different mount — the
wrong behaviour here, so it is recorded for diagnostics only.)

Each C920 exposes two video nodes; the `-index1` node is a metadata node that
opens cleanly and never yields a frame, so only `-index0` is matched.

```bash
python3 capture.py identify      # writes a labelled frame per port
```

Then the mapping goes in `jetson/rig.json`. This is the only step that needs a
human.

---

## 6. The exposure actuator is a 1-stop ladder

The largest single discovery of Phase 2.

### The symptom

The auto-exposure loop would not converge. Exposure oscillated, luma swung
between two values, and no amount of gain or damping tuning helped.

### The measurement

`exposure_time_absolute` advertises a range of 3–2047 in steps of 1. With **no
stream open**, the driver accepts all 747 values. **While streaming at 1080p it
does not** — it snaps to a ×2 ladder and clamps everything else to the nearest
rung:

```
38,  77,  156,  312,  624
```

### What that means

Exposure has **one stop of resolution**. Median luma can therefore only take
values a factor of two apart. On the test scene: **99 or 148, never 120**.

The loop was chasing a setpoint the actuator cannot produce. That is not a
tuning problem and no gain value fixes it.

It also explains a much earlier bug recorded as *"exposure drifted 156 → 312 on
stream start"*. It was not drifting. Those are adjacent rungs; it was being
pushed up exactly one.

### The fixes

| Fix | Reason |
|---|---|
| Snap exposure to the measured rungs | So `self.exp` records what the hardware is actually using, not what was requested |
| Step a rung only when it will not overshoot; use gain for the rest | Exposure is coarse, gain is continuous. Preferring exposure unconditionally caused the oscillation |
| Tolerance ±8 → ±30 | A tolerance finer than the actuator's resolution is really an instruction to use gain. Simulation settled at **gain 93–111**, and gain turns read noise into thin bright streaks — the exact false-positive signature |
| Rung-aware, bounded gain↔exposure trade | The old version multiplied exposure by 1.2, which snapped back to the same rung while the gain cut took effect — a limit cycle inside the fix for a limit cycle |
| `AutoExposure.hold()` — blind window after external changes | A fresh controller evaluated the first frame after stream start, a known transient, and stepped a whole rung on it. Luma walked 148 → 199 → 255 until it clipped |

### Result

All three cameras, every pass: **`exp=156, gain=0`**, luma 94–99, converged,
with **0.2% of pixels clipped** on a white ceiling with a blown-out window in
frame.

---

## 7. Frame averaging

Averaging N frames of a static scene cuts temporal sensor noise by √N without
touching spatial detail. This matters because the model responds to thin dark
filaments, and read noise produces exactly that.

**The frames must be independent.** With `BUFFERSIZE=1`, reading faster than the
camera produces returns the *same frame twice* and the average is of nothing.
Duplicates are detected on a subsampled grid and re-read rather than assumed
away.

Output is written as **PNG, not JPEG** — averaging frames to remove noise and
then re-quantising through a lossy codec puts a different kind of noise back.

---

## 8. Exposure carry-over and two-phase settling

Reconverging the exposure loop from scratch at every scan position would cost
seconds per camera, and worse, would let a converged camera drift to a different
operating point between positions — so two frames of the same pavement taken
minutes apart would not be comparable.

Exposure and gain are therefore **remembered across open/close**, and settling
is split:

| Phase | When | Budget |
|---|---|---|
| **prime** | once per camera per survey | 8 s — converge from the default to this site's lighting |
| **confirm** | every position after that | 1.2 s — verify the median is still in band |

A flat 2.5 s budget hit its cap at *every* position without converging — 7.5 s
of an 11.7 s cycle — because the loop can only step once per `interval` plus its
blind window.

This is only sound because the rig is stationary and lighting over a survey is
near-constant. On a moving platform, or outdoors under passing cloud, it would
have to become a per-position budget again.

---

## 9. Live preview mode

A scan position takes ~6.5 s, so the MJPEG stream publishes one image every
6.5 s. That is not lag — there are only ~9 frames a minute to publish.

Phase 0 measured the other end of the trade: **at 640×480 all three cameras
stream together at 15 fps indefinitely.** So `--mode live` opens every camera
once and holds it open.

| | scan | live |
|---|---|---|
| Resolution | 1920×1080 native tiles | 640×480, centre-cropped to square, resized to 512 |
| Rate | one position per ~6.5 s | **10 fps measured** |
| Inference | 223 ms/camera, 15 tiles | 15 ms/camera, 1 tile |
| mm/px | 0.29 | 0.83 |
| Smallest crack | ~0.59 mm | ~1.65 mm |

**The two modes report different quantities under the same field names.** A
widest-crack figure from preview is not comparable with one from a scan, so the
mode appears in the status payload, a header pill, the stage caption, and the
"scan cycle" row — which becomes "frame rate" when that is what it is.

Live preview is for **aiming and monitoring**. Measurement comes from a scan.

---

## 10. Focus — an open problem

`camera_setup.sh` pins `focus_absolute=30` with autofocus off. On a C920 that
control runs **0 (far) to 255 (close)**, so 30 is very nearly infinity — correct
for a camera looking down a corridor, wrong for one 20 cm from a surface.

A frame captured during this phase measured **16.6** on variance-of-Laplacian
where a sharp frame is in the hundreds.

This matters more here than almost anywhere else: defocus is a low-pass filter.
It spreads a 3 px crack across 15 px and lifts its minimum toward the
surrounding grey. **The crack does not get harder to see — it stops being
present in the image.** Nothing downstream recovers it, and any accuracy number
taken out of focus is measuring the blur.

`focus.py` (on `accuracy-filters`) sweeps the control and reports sharpness, but
it needs a **static, textured target** at the working distance — a sweep against
a blank wall returns noise.
