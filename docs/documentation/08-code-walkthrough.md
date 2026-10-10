# 08 — Code walkthrough

Every file, what it is for, its key functions, and how one scan travels through
them. Paths are relative to the repository root.

> **Two copies exist.** `jetson/` in the repo is the working copy; it is
> deployed to `~/cracknet/` on the Jetson and the two are kept byte-identical.
> Files marked **(branch)** exist only on `accuracy-filters`.

---

## 1. The path of one scan position

```
cracknet.bat live|scan                       (Windows)
   └─ tools/jssh.py ─ssh─▶ cracknet.sh start (Jetson)
        └─ python3 serve_scan.py --mode scan
             ├─ ScanLoop thread ─ run() ─ _loop()
             │     └─ Scanner.scan()                                   scanner.py
             │          ├─ SequentialRig.capture()                     capture.py
             │          │     for each camera:
             │          │       Camera.open()  → camera_setup.sh, AutoExposure(), hold()
             │          │       Camera.settle() → AutoExposure.update()  camera_ctl.py
             │          │       Camera.grab(4)
             │          │       Camera.close()
             │          └─ for role in (left, top, right):
             │               Scanner.infer_frame(frame)
             │                 ├─ tiler.plan_tiles / extract            tiler.py
             │                 ├─ CrackNetTRT.preprocess/infer_logits   trt_infer.py
             │                 ├─ tiler.blend_logits            (branch)
             │                 ├─ live.clean_mask                       live.py
             │                 └─ verify.apply                  (branch) verify.py
             │               measure.summarise(mask, gsd)               measure.py
             │     └─ _encode() → JPEGs;  _publish(jp, status)
             └─ HTTP Handler  ← reads published state                    serve_scan.py
```

---

## 2. Core modules (`jetson/`)

### `capture.py` — cameras (560 lines)

| Symbol | Purpose |
|---|---|
| `discover()` | List capture-capable cameras keyed by `/dev/v4l/by-path/*-video-index0`; serial from `by-id` for diagnostics only |
| `load_rig()` | `role → camera` from `rig.json`; falls back to port order (a guess) |
| `class Camera` | One C920. `open()`, `settle()`, `grab()`, `close()`. Remembers exposure/gain across open/close |
| `Camera.open()` | FOURCC → size → read until a frame → `camera_setup.sh` → `AutoExposure` → `hold()` |
| `Camera.settle(max_s, stable, min_s)` | Adaptive wait; counts **AE evaluations** (not frames) via `ae._last`; returns `(frames, converged)` |
| `Camera.grab(average)` | N independent frames → float32 mean → uint8; duplicate detection on `frame[::16, ::16]` |
| `class SequentialRig` | Open-capture-release in turn; prime (8 s, stable 2) once, confirm (1.2 s, stable 1) after |
| `class SimultaneousRig` | All cameras held open at 640×480 for live preview |
| CLI | `identify`, `bench`, `capture`, `sweep` |

### `camera_ctl.py` — exposure control (313 lines)

| Symbol | Purpose |
|---|---|
| `EXP_RUNGS` | `(19, 38, 77, 156, 312, 624, 1250, 2047)` — only 38…624 *observed* |
| `rung_index(value, rungs)` | Nearest rung **in log space** |
| `class AutoExposure` | Median-luma loop: `update(frame)`, `hold()`, `_step_exposure`, `_apply_gain`, `_apply_exposure`, `rungs` (honours `exp_cap`) |
| Defaults | `target 120, tol 30, interval 0.4, exp_cap 500 (capture: 700), gain_cap 160, gain_step 16, trade_gain 48, max_trades 4, flush 5, settle 0.5` |

Control flow is the decision diagram in [04 §5](04-camera-and-capture.md).

### `camera_setup.sh`

`v4l2-ctl` lock-down: autofocus off, `focus_absolute=30`, manual exposure,
`exposure_dynamic_framerate=0`, exposure 156, 50 Hz power-line. Each control is
tried under two kernel naming schemes. **Must be re-run after the stream
starts** — the firmware discards it before.

### `tiler.py` — tiling and reassembly (157 lines)

| Function | Purpose |
|---|---|
| `plan_tiles(w, h, tile=512, overlap=0.15)` | Origins; last in each direction clamped, never padded |
| `extract(frame, tiles)` | Slice, no resampling |
| `stitch_masks(masks, tiles, w, h)` | Union (`np.maximum`) — used on `main` |
| `tile_window(tile, taper)` | Raised-cosine weight; floored at 1e-3; `np.outer` |
| `blend_logits(logits, tiles, w, h)` | `Σ w·ℓ / Σ w`; returns `(logit_map, weight_map)` **(branch)** |
| `coverage_map` | How many tiles see each pixel (0 ⇒ gap) |
| `describe` | One-line cost estimate |

### `trt_infer.py` — TensorRT wrapper (72 lines)

| Symbol | Purpose |
|---|---|
| `logit(p)` | `ln(p/(1−p))`; `0.55 → 0.2007` |
| `CrackNetTRT(engine)` | Deserialises, allocates pinned host + device buffers, binds tensor addresses |
| `.preprocess(rgb)` | static; ImageNet normalise → NCHW contiguous float32 |
| `.infer(x)` | probability map (applies sigmoid) |
| `.infer_logits(x)` | **raw logits** — what the scanner uses |

### `live.py` — Phase 1 loop + `clean_mask` (214 lines)

| Symbol | Purpose |
|---|---|
| `open_c920`, `lock_camera_controls`, `center_square` | Phase 1 single-camera helpers |
| **`clean_mask(mask, min_area, max_area_frac, max_halfwidth, max_solidity, shape_filter)`** | The shape filter; area from `CC_STAT_AREA`, never `contourArea` |
| `main()` | The original single-camera detection loop with EMA verdict |

### `verify.py` **(branch)** — physical verification (222 lines)

| Function | Purpose |
|---|---|
| `valleyness(gray, scales)` | Per-pixel valley depth; Hessian only for direction |
| `_straightness(ys, xs, width_px)` | Minor-eigenvalue RMS ÷ `w/√12` |
| `_local_valleyness(...)` | Valley map on a margin-padded crop — whole-frame cost 1.1 s/position |
| `report(mask, gray, ...)` | Per-component `straightness`, `valleyness`, `long_enough` — inspect before choosing thresholds |
| `apply(mask, gray, max_straightness=1.12, min_valleyness=0, min_len_px=120)` | Drop components; returns `(mask, {"straight":n, "flat":n})` |

Note the default `min_valleyness` is **0 (off)** in `apply`; the scanner passes
**2.0**.

### `measure.py` — millimetres (177 lines)

| Function | Purpose |
|---|---|
| `gsd_mm_px(standoff_m, width_px, hfov_deg=70.42)` | Eq. 1 |
| `_width_from_distance(sub)` | border → `distanceTransform` → ridge → `(median, p95, max)` as `2d−1` |
| `components(mask, gsd, min_area)` | Per component: `width_mm` (p95), `width_median_mm`, `width_max_mm`, `width_mean_mm`, `length_mm`, bbox; widest first |
| `summarise(mask, gsd)` | `coverage`, `components`, `widest_mm`, `longest_mm` |

### `scanner.py` — one position (318 lines on branch)

| Symbol | Purpose |
|---|---|
| `class Scanner` | Owns engine + rig; computes `gsd`, `max_halfwidth_px`, tiles |
| `.infer_frame(frame)` | tile → infer → blend → clean → verify |
| `.scan()` | capture, then per camera `infer_frame` + `summarise`; returns `(cams, totals)` |
| `overlay`, `composite` | 60 % red fill + yellow contour; labelled composite. Mask resize uses `INTER_NEAREST` (a mask is labels, not intensities) |
| `add_arguments` | Every flag — table in [05 §12](05-perception-pipeline.md) |
| `MIN_AREA_NATIVE = 600` | The rescaled area floor |

### `serve_scan.py` — the console server (490 lines)

| Symbol | Purpose |
|---|---|
| `class ScanLoop(Thread)` | `_loop` (scan), `_loop_live`, `_publish`, `_encode`, `log`, `wait_frame` |
| `Handler` | `/`, `/status.json`, `/events.json`, `/stream.mjpg`, `/cam/<i>/snapshot.jpg` |
| `soc_temp_c`, `gpu_mhz`, `power_mode` | System readouts; `power_mode` parses the zero-padded `pmode:0002` |
| `main()` | `--mode scan|live`, `--port`, composite/thumb widths, JPEG quality |

### `serve.py` — the Phase 1 single-camera server

Original MJPEG server (`:8080`); superseded by `serve_scan.py` but kept as
`CRACKNET_MODE=single`.

### `cracknet.sh`, `netlink.sh`

| | |
|---|---|
| `cracknet.sh start|stop|restart|status|logs|check` | Preflight → performance mode → per-camera setup → server → wait for port. `CRACKNET_MODE=live|scan|single` selects server, port, pidfile and log |
| `netlink.sh ap|wifi|status|psk|boot` | Create/switch the AP with NetworkManager; `boot` chooses the mode that survives reboot; restores client autoconnect if the AP fails |

---

## 3. Analysis and diagnostic scripts

| Script | Question it answers | Result it produced |
|---|---|---|
| `multicam_probe.py` | How many cameras stream at once? | 3 @ 640×480, 2 @ 720p |
| `bench_tiling.py` | What does the tiled path really cost? | 88.2 / 221.1 ms per camera |
| `analyze_tile_edges.py` | Do tile borders generate detections? | 2.18× / 11.33× |
| `analyze_blobs.py` | What do false blobs look like geometrically? | half-width 11.2, solidity 0.715 |
| `check_parity.py` | Does FP16 TensorRT agree with ONNX? | 0.0552 % |
| `capture_samples.py` | Grab webcam frames as parity inputs | — |
| `check_fov_parity.py` | Does 1080p use the full sensor? | **no result recorded** |
| `explain.py` **(branch)** | Why was each detection kept/dropped, with margin? | — |
| `focus.py` **(branch)** | Which focus setting is actually sharp? | needs a textured target |
| `inject_crack.py` **(branch)** | Does a *plausible* crack survive the filters? | — |

### `inject_crack.py` — why it exists

There is no labelled crack on the device, so the question the filters were never
asked is "does a real crack survive?". A synthetic mask on a synthetic
background cannot answer it (the valley test reads the *image*). So a crack is
drawn **into a real captured frame**, and made plausible: it *meanders* (a ruled
line would be correctly rejected), *tapers*, is a *valley* (multiplicatively
darker so depth follows local illumination), and is *soft-edged* (a lens has a
point-spread function).

---

## 4. Tests

| Test | Checks | Count |
|---|---|---|
| `test_tiler.py` (branch) | coverage, bounds, byte-identical sub-arrays, round trip, union, connectivity across seams, blend window, blend rejects seam spike | **23/23** (13 on `main`) |
| `test_measure.py` | width vs analytic shapes: straight, diagonal, branched, tapered | 13 cases |
| `test_verify.py` (branch) | valley vs step; straightness monotonic | 5 + 5 |
| `test_ae_sim.py` | convergence from both extremes, incl. stuck state | **7/7** |
| `test_autoexposure.py` | on-camera recovery smoke test | — |

Run locally (no GPU needed for the first three):

```powershell
cd docs\documentation\figures\_code        # populated by make_figures.py
$env:PYTHONPATH="."; python test_measure.py; python test_verify.py
```

---

## 5. Host-side tools (`tools/`)

| File | Purpose |
|---|---|
| `jssh.py` | paramiko: run a command / `--put` / `--get` / `--file` / `--sudo`. Reads `tools/.jetson.env`; tries each host in `JETSON_HOSTS` with a 1.5 s TCP probe; redacts the password from output |
| `jlink.py` | `join` (fetch AP password over the live route, install a WLAN profile with `netsh`, no admin), `leave`, `status`, `host` (with automatic reconnect) |
| `jproxy.py` | HTTP/HTTPS proxy tunnelled over SSH so the Jetson can `apt`/`pip` via the PC |
| `demo_server.py` | Synthetic backend for the console; advertises `source: synthetic` |

`cracknet.bat` at the root wraps all of them.

---

## 6. Data shapes worth knowing

```python
frame        : (1080, 1920, 3) uint8 BGR                 # averaged capture
tiles        : [(x, y), ...]     15 origins              # plan_tiles
crop         : (512, 512, 3)     uint8                   # extract
x (model in) : (1, 3, 512, 512)  float32, contiguous     # preprocess
logits       : (512, 512)        float32                 # infer_logits
blended      : (1080, 1920)      float32                 # blend_logits[0]
mask         : (1080, 1920)      uint8 {0, 255}          # threshold, clean, verify
component    : dict(area_px, width_mm, width_median_mm, width_max_mm,
                    width_mean_mm, length_mm, x, y, w, h)
cam          : dict(name, frame, mask, coverage, components, widest_mm,
                    exposure, gain, median_luma, converged, infer_ms,
                    dropped_straight, dropped_flat)
```
