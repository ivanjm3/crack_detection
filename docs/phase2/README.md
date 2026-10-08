# Phase 2 — three cameras, stationary scanning

Everything from "use three cameras instead of a motor" up to the current state
of the repo. Written to be read cold.

## What Phase 2 is

Phase 1 was one C920 pointed at a surface, centre-cropped to a square,
downscaled to 512, run through a TensorRT CrackNet engine, and streamed as an
MJPEG overlay. It worked, but it saw a narrow strip of ground and threw away
most of the sensor.

Phase 2 replaces that with **three fixed cameras** (left, top, right) on a
stationary rig, each inferred independently at native resolution, with the
results reported in **millimetres** rather than as a percentage of pixels.

Two decisions shape everything else:

| Decision | Why |
|---|---|
| Cameras are read **one at a time**, not simultaneously | Three 1080p streams do not fit on this board's single USB 2.0 bus. The rig is stationary, so sequential reads are indistinguishable from simultaneous ones. |
| Images are **never stitched before inference** | A blended seam is a long thin dark line — exactly the thing a crack detector is built to find. The stitched composite is for human eyes only. |

## The documents

| File | Contents |
|---|---|
| [01-camera-weaving.md](01-camera-weaving.md) | Getting three cameras to produce usable frames: bandwidth, sequencing, role mapping, exposure, focus |
| [02-perception.md](02-perception.md) | What happens to a frame after capture: tiling, inference, mask cleaning, measurement, verification |
| [03-ui.md](03-ui.md) | Every number the inspection console shows, and what it does and does not mean |

## Chronology

| # | What | Outcome |
|---|---|---|
| 1 | Proposed three cameras with a stitched feed into the model | Rejected: resolution collapse and seam artefacts |
| 2 | Chose stationary scanning over a moving rig | Removes motion blur, frame-rate and synchronisation constraints |
| 3 | Measured the USB bus with three cameras attached | 3× 640×480 works; 3× 720p does not. Sequential capture required |
| 4 | Built `capture.py` — sequential open/capture/release | 5.93 s per three-camera scan position at 1080p |
| 5 | Found the C920 shutter is a 1-stop ladder | Rewrote auto-exposure; exposure stable at 156, gain 0 |
| 6 | Built `scanner.py` — capture → tile → infer → measure | 6.5 s per position, 223 ms inference per camera, 0.294 mm/px |
| 7 | Built `measure.py` — masks to millimetres | Width from the distance-transform ridge, validated to ±0.75 px |
| 8 | Built `serve_scan.py` + dashboard | Live inspection console over HTTP |
| 9 | Added live preview mode | 10 fps at 640×480, because a scan is not a video |
| 10 | Measured tile-border artefacts | 2.2× raw, 11.3× after shape filtering |
| 11 | Built blending + two verification filters | Curtain scene: 18 false detections → 0 |
| 12 | Reverted (11) on `main` for a demo | Work preserved on `accuracy-filters` |

## Where the code is

Everything runs on the Jetson in `~/cracknet/`, mirrored in `jetson/` here.

| File | Role |
|---|---|
| `capture.py` | Camera discovery, role mapping, sequential and simultaneous rigs |
| `camera_ctl.py` | Auto-exposure control loop |
| `camera_setup.sh` | Puts a C920 into manual exposure/focus via v4l2-ctl |
| `tiler.py` | Tile planning, extraction, mask reassembly |
| `trt_infer.py` | TensorRT engine wrapper |
| `live.py` | `clean_mask()` shape filtering, plus the Phase 1 single-camera loop |
| `measure.py` | Masks to millimetres |
| `scanner.py` | One scan position, end to end |
| `serve_scan.py` | HTTP server for the console |
| `viewer/dashboard.html` | The console page |

Tests: `test_tiler.py`, `test_measure.py`, `test_ae_sim.py`, `test_autoexposure.py`.
Diagnostics: `multicam_probe.py`, `bench_tiling.py`, `analyze_tile_edges.py`.

## Current state

- **`main`** — the sensitive, less precise configuration. Tile masks are unioned;
  no straightness or valley filtering. This is what a demo runs.
- **`accuracy-filters`** — adds tile blending and the two verification filters,
  plus `focus.py` and `inject_crack.py`. Merge when the demo is done.

### Known open items

1. **Scale is uncalibrated.** Every width in millimetres is derived from an
   *assumed* 0.40 m standoff. Widths scale linearly with it, so a camera at
   0.15 m reports everything ~2.7× too wide. Fix by measuring the distance, or
   photographing something of known width and passing `--mm-per-px`.
2. **Focus is wrong for close work.** `camera_setup.sh` pins `focus_absolute=30`,
   which on a C920 is near-infinity. A frame captured at ~20 cm measured 16.6 on
   variance-of-Laplacian where a sharp frame is in the hundreds.
3. **No real crack has been through the pipeline yet.** Accuracy claims are
   based on synthetic cracks and on real false positives.
4. **Smallest detectable crack is undecided.** This fixes the standoff and the
   swath, and Phase 3 calibration cannot start without it.
5. **The viewer is unauthenticated** and binds `0.0.0.0`.
