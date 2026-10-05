# Three-camera coverage — research and implementation plan

**Status:** plan only, nothing implemented. Written 2026-10-05.
**Decided:** three C920s in hand; **stationary scanning** (see §4A).
**Context:** panning one camera with a motor is impractical (heavy head, inefficient
drive), so the proposal is three fixed cameras — left, top, right — covering the full
working width as the rig advances.

---

## 1. The decisive finding, first

**The idea is right. The proposed pipeline order is not.**

Combining three feeds is the correct response to "we cannot pan". But stitching the
three images into one picture and sending *that* to the model destroys the thing the
detector depends on. Two independent reasons, both fatal.

### 1.1 Resolution arithmetic

The model's input is a fixed budget: 512x512. Every square metre of real surface you
push through one inference costs you pixels per millimetre.

| | crop fed to model | scale | a 2 px crack becomes |
| --- | --- | --- | --- |
| **Today, one camera** | 720x720 | 512/720 = **0.711** | **1.42 px** |
| Stitch 3 into one inference | ~3328x720 | 512/3328 = **0.154** | **0.31 px** |

A 3328 px wide mosaic comes from three 1280 px frames with ~20 % overlap. Squashing
that to 512 is a 4.6x further reduction on top of a budget that, per
`perception.html` §3, already has only 1.42 px of margin. **0.31 px is below one
sample. The crack is gone before the model sees it.**

There is no way around this. Three cameras cover 3x the area; covering 3x the area at
the same detail costs 3x the inference. The only question is *where* the three results
get combined — not whether you can avoid paying.

### 1.2 Stitching seams are crack-shaped

This one is worse, because it fails silently.

A blended stitching seam is a **thin, elongated, low-contrast linear feature**. That is
the exact signature the detector is trained to find, and — critically — it is also
exactly what the stage-6 shape filter is built to *keep*. `max_halfwidth` and
`solidity` reject fat compact blobs; a seam artifact is neither. It would pass every
defence in the pipeline and be reported as a crack spanning the full frame height, at
high confidence, in every single frame.

Feature-based stitching makes this worse still. The literature is blunt about why:
ground and road surfaces *"often do not provide sufficient texture"* for reliable
correspondence extraction, and feature-based panoramic imaging *"heavily depends on
achieving sufficient distribution and accurate matching of feature points"*
([ground-plane homography calibration](https://www.researchgate.net/publication/261273387_Online_extrinsic_multi-camera_calibration_using_ground_plane_induced_homographies),
[multi-camera panorama generation](https://doi.org/10.3390/app132212309)). Pavement
and concrete are the canonical low-texture case. ORB/SIFT matching on them is
unreliable frame to frame, so the homography would jitter and the seam would *move*
between frames — a flickering linear false positive that temporal smoothing cannot
suppress.

### 1.3 What to do instead

Keep the combination, move it downstream:

```
per camera:   capture -> auto-exposure -> crop -> resize -> MODEL -> mask -> shape filter
then:         warp each MASK into one common ground plane -> merge -> analyse -> decide
display only: warp the RGB too, for a human to look at
```

Three inferences, one unified result. Masks are binary and warp with
nearest-neighbour, so there is no blending, no seam, and no invented pixels. Each
camera keeps the full 0.711 scale it has today.

---

## 2. Recommended architecture

### 2.1 Fixed homography, not feature matching

The surfaces being inspected are **planar** (pavement, wall, slab) and the cameras are
**rigidly mounted**. Both facts are gifts:

- A plane-to-plane mapping is *exactly* a homography — not an approximation.
- Rigid mounting means that homography is **constant**. Compute it once, offline, from
  a calibration target. Nothing is estimated at runtime.

This is inverse perspective mapping (IPM), the standard approach for road-surface
analysis: *"the bird's eye view of the road plane can be given by inverse perspective
mapping or plane rectification through image warping"*. It needs no texture, cannot
drift, and costs one `cv2.warpPerspective` per camera per frame.

### 2.2 The payoff: thresholds become physical

Once masks live in a common ground plane at a known scale (say **0.5 mm per pixel**),
every stage-6 parameter stops being a pixel count and becomes an engineering quantity:

| Today (pixels, camera-dependent) | In the ground frame |
| --- | --- |
| `max_halfwidth = 12 px` | **crack width in mm** — what civil standards actually specify |
| `min_area = 300 px` | minimum crack **area in mm²** |
| `alert_frac = 1 %` of frame | crack area **per m² surveyed** |

This also repairs a real defect in the current design. `perception.html` §6 admits the
12 px threshold cannot be validated without in-domain data; in millimetres it can be
validated against a ruler. It additionally makes thresholds **transferable** — they
stop depending on standoff distance, currently the single largest uncontrolled
variable in the system.

### 2.3 Merging in a common frame beats three independent alarms

The simplest option — three detectors, three panes, OR the alerts — is a valid Phase 1,
but it has a specific weakness: **a crack crossing a camera boundary becomes two
truncated components.** Each piece may fall below `min_area`, and both carry wrong
shape metrics because they are cut. Merging masks in the ground frame rejoins them
into one component *before* any geometry is measured. Overlap between adjacent cameras
also gives free cross-validation: a detection seen by two cameras is stronger evidence
than one seen by either alone.

---

## 3. The USB bandwidth wall — sidestepped, but worth understanding

> **Stationary scanning does not hit this.** Sequential capture (§4A.2) means only one
> camera streams at a time, so the reservation that breaks the simultaneous case never
> occurs. This section is retained because it still governs whether the three cameras
> can stream *together* — which would cut the scan cycle from ~10 s to ~3 s — and
> because it would return immediately if the rig is ever motorised.

Probed on the board today (`lsusb -t`, `/sys/bus/usb/devices`):

```
Bus 02  root_hub  10000M  ->  Realtek 4-Port USB 3.0 Hub
Bus 01  root_hub    480M  ->  Realtek 4-Port USB 2.0 Hub   <- every Type-A port
```

**All four Type-A ports sit behind one internal Realtek hub, and the C920 is a USB 2.0
device.** It therefore always lands on Bus 01 and shares a single 480 Mbit/s bus — the
5 Gbit/s path on Bus 02 is unreachable to it, whichever physical port is used.

That matters, because UVC reserves *isochronous* bandwidth from the figure the camera
firmware declares (`dwMaxPayloadTransferSize`), not from what it actually sends. The
C920 declares a high value even for MJPG — on the order of 130–200 Mbit/s each. Three
of those is 390–590 Mbit/s against a bus where the USB spec caps periodic transfers at
80 % of the frame, i.e. ~384 Mbit/s usable. **The third camera is likely to fail to
open with `ENOSPC` / "No space left on device".**

This is a documented wall, not a guess. NVIDIA's own forums carry
[multiple](https://forums.developer.nvidia.com/t/running-multiple-usb-cameras-on-jetson-orin-nano-only-2-working-need-all-4/343994/7)
[reports](https://forums.developer.nvidia.com/t/support-required-for-connecting-4-cameras-on-jetson-orin-nano-usb-bandwidth-limitation/351273)
of exactly this on Orin Nano — two cameras working, the third refusing — and NVIDIA's
standing recommendation for multi-camera work is CSI/MIPI rather than USB.

### 3.1 Options, cheapest first

| # | Option | Cost | Confidence |
| --- | --- | --- | --- |
| 1 | **`uvcvideo quirks=128`** (`UVC_QUIRK_FIX_BANDWIDTH`) — the driver computes real bandwidth for compressed formats instead of trusting the descriptor | free | good; [the standard fix](https://www.thegoodpenguin.co.uk/blog/multiple-uvc-cameras-on-linux/), with a [kernel patch](https://github.com/AgRoboticsResearch/uvc_multi_cam_patch) if the quirk alone is not enough |
| 2 | Drop to 15 fps and/or 640x480 per camera | free | partial — reservation is often per alt-setting, not per fps |
| 3 | **2x USB C920 + 1x CSI camera** (IMX219/IMX477). CSI is a wholly separate path: zero USB contention, plus hardware ISP | ~2–3k INR | high, but mixed optics complicate calibration |
| 4 | A genuine USB 3.0 UVC camera for the third position, landing on Bus 02 | higher | high |
| 5 | Three CSI cameras | the dev kit exposes 2 CSI connectors | **not possible without a carrier swap** |

**Recommendation:** test option 1 the day the cameras arrive — it costs nothing and
probably works. Plan for option 3 as the fallback.

### 3.2 The second bottleneck: JPEG decode

`perception.html` §1 records that capture tops out at ~22 fps with **one** camera,
because MJPEG decode runs on the CPU. Three cameras at 15 fps each is 45 fps of decode
— past that ceiling already, on a 6-core part that also has to run three auto-exposure
loops and three shape filters.

Hardware decode is therefore **mandatory, not an optimisation**. Confirmed present on
this board:

```
nvjpeg:  nvjpegdec: JPEG image decoder
nvvidconv:  nvvidconv: NvVidConv Plugin
```

That means replacing `cv2.VideoCapture(dev, cv2.CAP_V4L2)` with a GStreamer pipeline:

```
v4l2src device=/dev/videoN ! image/jpeg,width=1280,height=720,framerate=15/1
  ! nvjpegdec ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert
  ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1
```

This interacts with stage 2, and the interaction is easy to miss: `AutoExposure` drives
exposure through `cap.set(cv2.CAP_PROP_EXPOSURE, ...)`, which **does not exist on a
GStreamer capture**. Control has to move back to `v4l2-ctl` subprocess calls or direct
ioctl. That is a genuine refactor of `camera_ctl.py`, not a drop-in swap.

### 3.3 Inference batching

The current engine is fixed batch 1 — verified on the board today:

```
input  (1, 3, 512, 512)
logits (1, 1, 512, 512)
```

Three sequential calls cost 3 x 8.77 = 26 ms of GPU compute. Re-exporting at batch 3
(or dynamic batch 1–3) and issuing one `execute_async_v3` should land around 20–24 ms
by amortising launch overhead. Either figure fits comfortably in a 15 fps budget
(66 ms). Not urgent, but cheap to do while the ONNX export is already open.

---

## 4. Phased plan

Each phase has a gate. Do not start the next one until its gate passes.

### Phase 0 — Measurement, not a gate (half a day)

Under stationary scanning this no longer blocks anything (§4A.2). Run it to learn
whether simultaneous streaming is available — if it is, the scan cycle drops from ~10 s
to ~3 s — then proceed either way.

- `multicam_probe.py`: enumerate all video nodes, open every camera at MJPG 1280x720,
  hold all streams open for 60 s, report per-camera achieved fps, dropped frames, and
  any open failure with its errno.
- Run it three ways: baseline, with `uvcvideo quirks=128`, and at 15 fps / 640x480.
- Measure aggregate CPU with CPU decode versus `nvjpegdec`.

**If three stream:** keep them open; skip the open/settle cost per camera.
**If only two:** fall back to sequential capture. Nothing else changes.

Also measure here: FOV parity between 720p and 1080p, and achieved 1080p rate — §4A.7
lists why both matter.

### Phase 1 — Three detectors + native tiling (2–3 days)

No calibration. Useful on its own, and this is where most of the gain lands.

- Capture layer that works either way: simultaneous if the bus allows, else sequential.
- **Native-resolution tiling** (§4A.3) replacing centre-crop-then-downscale — the single
  highest-value change, worth 1.8x swath and 2.1x finer detection on its own.
- **Frame averaging** (§4A.5), N configurable, `--no-average` to A/B it.
- One `AutoExposure` instance per camera; long exposure, `gain = 0`, no blur cap.
- Batched engine (tile count per call, not 3).
- Per-camera and per-tile coverage in `status.json`.

**Gate:** a scan of a known surface completes in under ~15 s and resolves a crack that
the current single-camera pipeline misses.

### Phase 2 — Calibration (1 day)

- `calibrate_intrinsics.py` — checkerboard, per camera. **Not skippable:** the C920 has
  visible barrel distortion at frame edges, which bends straight cracks and corrupts
  the very width measurement stage 6 depends on.
- `calibrate_ground.py` — with a known rectangle on the target surface, take >= 4
  correspondences per camera between the undistorted image and ground coordinates;
  `cv2.findHomography` -> `H_cam_to_ground`.
- Store intrinsics, distortion, `H` and mm-per-pixel in `cameras.json`.
- **Validate:** place an object of known size, measure it in the ground frame, report
  the error in mm. A calibration nobody checked is a calibration that is wrong.

**Gate:** a known 100 mm object measures 100 ± 3 mm anywhere in the merged frame.

### Phase 3 — Ground-frame merge (1–2 days)

- Warp each mask with `cv2.warpPerspective(..., INTER_NEAREST)` into the shared frame.
- Merge by OR; run connected components **once**, on the merged map.
- Convert thresholds to physical units; report crack width in mm and crack area per m².
- Rework `clean_mask()` to take millimetres and a scale rather than raw pixels.

**Gate:** a single crack laid across a camera boundary is reported as **one** component
with the correct total length.

### Phase 4 — Display mosaic (half a day, optional)

Warp the RGB into the same frame for a human-viewable bird's-eye image. **Labelled
display-only in the code**, so nobody is ever tempted to feed it to the model.

### Phase 5 — Survey strip (future)

As the rig advances, accumulate the ground-frame map into a continuous strip map — what
road-survey orthomosaic systems do. Needs odometry (wheel encoder, or visual
displacement estimation). Out of scope now; the ground frame from Phase 3 is the
prerequisite that makes it possible later.

---

## 4A. Stationary scanning — what it changes

**Decided:** the rig does not move. It is positioned, it scans, it is repositioned by
hand. That removes three constraints and unlocks two substantial gains. Net effect: a
**better** system than the moving design, not a compromise.

### 4A.1 Constraints that disappear

| Constraint in the moving design | Stationary |
| --- | --- |
| Motion blur caps speed at ~20 mm/s | **Gone.** Exposure is free |
| `exp_cap` must shrink to 15–60 | **Inverted** — use *long* exposure and `gain = 0` for the lowest noise floor |
| Free-running cameras misregister seams by 5–11 px | **Gone.** Nothing moves between captures; no sync needed, no morphological close |
| Illumination needed to buy a short exposure | **Optional.** Still worth having for shadow suppression, no longer needed for speed |
| USB bus must carry three simultaneous streams | **Gone** — see §4A.2 |

### 4A.2 The USB blocker dissolves

§3 is the biggest risk in the moving design: three C920s on one 480 Mbit/s bus, with
the third likely refusing to open. **Stationary scanning does not need them streaming
at once.**

Capture sequentially: open camera 1, let exposure settle, grab and average, close;
repeat for 2 and 3. Only one stream is ever live, so the bandwidth reservation that
breaks the simultaneous case never happens. It works regardless of what the bus would
have allowed.

Cost is time, not quality: each open pays the C920's manual-mode reset and settling
(~1 s, the five-frame prelude from `perception.html` §1). Budget ~2 s per camera.
Irrelevant when the rig is parked.

Still run `multicam_probe.py` — if all three *do* stream simultaneously, keep them
open and skip the switching cost. But it stops being a **gate**. Nothing is blocked on
the answer.

### 4A.3 The real gain: stop throwing pixels away

The centre-crop-then-downscale in the current pipeline exists for one reason — a
single 512×512 inference cannot cover a 16:9 frame without anisotropic distortion
(`perception.html` §3). It costs twice over: the left and right thirds of every frame
are discarded, and what remains is downscaled by 0.711.

Stationary operation removes the reason for that compromise, because **inference is no
longer the scarce resource**. Tile the *full* frame into 512×512 crops at **native
resolution** instead. Each tile is square, so no distortion; nothing is downscaled, so
no detail is lost; nothing is cropped, so the full sensor width is used.

| mode | GSD | swath/cam | 3-cam swath | **min crack** | tiles/cam |
| --- | ---: | ---: | ---: | ---: | ---: |
| current: 720p, crop 720, →512 | 0.62 mm/px | 0.32 m | 0.80 m | 1.23 mm | 1 |
| stationary: 720p full, tiled | 0.44 mm/px | 0.56 m | 1.43 m | 0.88 mm | 6 |
| **stationary: 1080p full, tiled** | **0.29 mm/px** | **0.56 m** | **1.43 m** | **0.58 mm** | 15 |

*(0.4 m standoff. 15 % tile overlap.)*

Against the current pipeline that is **1.8× the swath and 2.1× finer crack
detection, simultaneously** — purely from spending inference that a stationary rig has
to spare.

The cost is trivial at this duty cycle:

```
 720p:  6 tiles/cam x 3 cams = 18 inferences = 0.16 s
1080p: 15 tiles/cam x 3 cams = 45 inferences = 0.39 s
```

Under half a second of GPU per scan position.

### 4A.4 Standoff, restated for stationary 1080p tiling

| standoff | GSD | swath/cam | 3-cam swath | **min crack width** |
| ---: | ---: | ---: | ---: | ---: |
| 0.25 m | 0.18 mm/px | 0.35 m | 0.89 m | **0.36 mm** |
| 0.40 m | 0.29 mm/px | 0.56 m | 1.43 m | **0.58 mm** |
| 0.60 m | 0.44 mm/px | 0.84 m | 2.14 m | **0.88 mm** |
| 1.00 m | 0.73 mm/px | 1.40 m | 3.57 m | **1.46 mm** |

Sub-millimetre detection is now reachable out to 0.6 m, with a 2.1 m swath — a
combination the moving design could not offer at any standoff.

### 4A.5 Frame averaging — free accuracy, stationary only

Nothing moves, so N frames of the same scene can be averaged before inference. Sensor
noise falls as √N while crack contrast is unchanged:

| N | SNR gain | capture time @15 fps |
| ---: | ---: | ---: |
| 4 | 2.0× | 0.3 s |
| 8 | 2.8× | 0.5 s |
| **16** | **4.0×** | **1.1 s** |
| 32 | 5.7× | 2.1 s |

This matters precisely where the system is weakest. A hairline crack is a *low-contrast*
feature; whether it survives depends on contrast against the noise floor, and averaging
attacks the noise floor directly. It also compounds with `gain = 0`, which is now free.

Two honest limits. Averaging improves **contrast**, not **resolution** — a crack far
below one pixel stays unresolvable, and the shape metrics of stage 6 are still bounded
by GSD. And it demands a genuinely rigid mount: any vibration during the N frames
turns averaging into blurring, converting the gain into a loss. A stable tripod or
clamped frame is not optional.

Minor domain note: the model was trained with noise augmentation, so a cleaner input is
at or beyond the clean end of its training distribution. Expected to help or be
neutral, but worth an A/B check against `--no-average` once running.

### 4A.6 Revised scan cycle

```
per position:
  for each camera (sequentially):
      open, settle exposure            ~2 s
      capture and average 16 frames    ~1 s
      close
  tile each frame, batch-infer         ~0.4 s
  warp masks to ground plane, merge    ~0.1 s
  report widths in mm                  -
                                       ---------
                                       ~10 s per scan position
```

Ten seconds parked, covering a 1.43 m swath at 0.58 mm resolution. If all three
cameras stream simultaneously the open/settle cost disappears and it drops to ~3 s.

### 4A.7 What to verify before building

- **FOV identical at 720p and 1080p?** Assumed here; the C920 should read the full
  sensor for both rather than cropping, but measure it — the whole table in §4A.4
  depends on it. Photograph a ruler at a known distance in each mode.
- **1080p MJPG sustains capture?** Confirmed available in the September probe
  (`MJPG: 1280x720, 1920x1080`); confirm the achieved rate.
- **Tiling parity:** re-run `check_parity.py` on native-resolution tiles. The model was
  exported at 512 and has only ever been fed downscaled crops; native-scale input is a
  different spatial frequency distribution and should be checked, not assumed.

---

## 5. Open risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| **Third camera will not open** (§3) | slower scan cycle only | Sequential capture (§4A.2). No longer blocking |
| **Mount is not rigid enough** for frame averaging | averaging becomes blurring — a net loss | Clamped frame or tripod; verify by comparing an averaged frame against a single one |
| **720p and 1080p FOV differ** | every figure in §4A.4 shifts | Measure with a ruler before committing to a standoff (§4A.7) |
| **Native-scale input is a domain shift** — the model has only ever seen downscaled crops | unknown accuracy change, possibly favourable | Re-run `check_parity.py` on native tiles; A/B against the current path |
| **Exposure differs per camera** — left/top/right see different light, so the display mosaic shows brightness steps | cosmetic for the model (each normalises independently), ugly for humans | Accept, or exposure-compensate the display mosaic only |
| **Tile boundaries cut cracks**, as camera seams do | truncated components, wrong shape metrics | 15 % tile overlap + merge in the ground frame before labelling (§2.3) |
| **Calibration drifts** if the rig is knocked | silent mis-measurement | Re-validate with the known object before each session; it takes a minute |
| **Non-planar surface** (kerb, step, rubble) breaks the homography assumption | mis-registration near the discontinuity | Document the assumption; the planar case is the intended use |
| **CPU saturation** from 3 decode + 3 AE + 3 filters | frame rate collapse | `nvjpegdec` (§3.2); AE already self-throttles to 0.4 s |

---

## 6. What can be built before the cameras arrive

Doable now, with one camera or none:

- `multicam_probe.py` — the Phase 0 harness, ready to run on arrival
- Batch-3 ONNX export, engine build, parity re-check (feed one frame duplicated 3x)
- `serve.py` refactor to the N-camera architecture, exercised with N=1
- Homography machinery plus unit tests on synthetic data, no hardware needed
- `nvjpegdec` capture path, testable with the single existing camera

Blocked until three cameras exist: the Phase 0 gate, real calibration, and everything
downstream of it.

---

## 7. Summary

| Question | Answer |
| --- | --- |
| Three cameras instead of panning? | **Yes.** Correct call — no motor, no moving mass, no settling time |
| Stitch the feeds into one, then infer? | **No.** Resolution collapses to 0.31 px per crack, and seams are crack-shaped false positives the shape filter is built to keep |
| So how do they combine? | Three inferences at full per-camera resolution, then merge the **masks** in a calibrated common ground plane |
| Feature matching or fixed homography? | **Fixed homography.** The surface is planar and the cameras are rigid, so it is exact and constant. Feature matching fails on low-texture pavement |
| Moving or stationary? | **Stationary**, as decided — and it is the stronger system: no blur, no sync error, exposure free, and inference to spare |
| Biggest win available? | **Native-resolution tiling** instead of centre-crop-then-downscale: 1.8x swath and 2.1x finer cracks, no new hardware |
| Is the USB bandwidth wall still a problem? | **No.** Sequential capture sidesteps it; simultaneous streaming is now an optimisation, not a prerequisite |
| Biggest unknown? | Whether 1080p and 720p share a FOV, and how the model behaves on native-scale tiles. Both measurable in an afternoon |
