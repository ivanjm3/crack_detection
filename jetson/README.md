# CrackNet on the Jetson Orin Nano

Deployment code for the CrackNet crack-segmentation model on a Jetson Orin
Nano with a Logitech C920. These files are the working copy; they are deployed
to `~/cracknet` on the Jetson and the two copies are kept identical.

## Files

| File | Purpose |
| --- | --- |
| `trt_infer.py` | Minimal TensorRT 10 runner. Loads the engine, preprocesses, returns a probability map. |
| `live.py` | Camera → model → overlay loop. Also owns `open_c920()`, `center_square()` and `clean_mask()`, which the other scripts import. |
| `serve.py` | Serves the live overlay as MJPEG over HTTP. Stdlib only. |
| `camera_ctl.py` | Closed-loop exposure/gain control (`AutoExposure`). |
| `camera_setup.sh` | One-shot V4L2 control lock-down (setup.md §4.2). |
| `run_server.sh` | Start/restart the viewer. |
| `build_engines.sh` | Builds the FP16 and FP32 TensorRT engines (§3). |
| `capture_samples.py` | Grabs C920 frames into `val_samples/` as parity/tuning inputs. |
| `check_parity.py` | TensorRT vs ONNX Runtime parity check (§5). |
| `analyze_blobs.py` | Measures blob geometry; justifies the shape-filter thresholds. |
| `test_ae_sim.py` | Deterministic test of the exposure control law against a simulated sensor. |
| `test_autoexposure.py` | On-camera smoke test for exposure recovery. |

## Running

```bash
cd ~/cracknet
bash camera_setup.sh                 # optional; live.py re-applies this itself
python3 live.py --headless           # detection loop, console output
python3 live.py --headless --record run.mp4
bash run_server.sh --rotate 0        # network viewer on :8080
python3 test_ae_sim.py               # 7/7 expected
```

Useful flags (`live.py` and `serve.py` share them): `--thresh`, `--min-area`,
`--alert-frac`, `--max-halfwidth`, `--max-solidity`, `--max-area-frac`,
`--no-shape-filter`, `--ae-target`, `--gain`, `--no-auto-exposure`, `--rotate`.

## Measured on this device

| | |
| --- | --- |
| FP16 engine | 8.77 ms GPU compute, 113.7 qps |
| FP32 engine | 18.44 ms |
| TensorRT vs ONNX parity | 0.0552 % mask disagreement (bar: 0.1 %) |
| Live end-to-end | ~15–19 fps, 18.4 ms inference |
| Capture ceiling | ~22 fps (CPU MJPEG decode), so the loop is camera-bound |

## Where this departs from `setup.md.txt`

The guide is wrong or incomplete in these places for this board. The code here
already accounts for all of them.

1. **§2.2 `nvpmodel -m 0` is not max performance.** On this board mode 0 is
   15 W; `MAXN_SUPER` is **mode 2**. Using mode 0 *lowers* the power budget.
2. **§4.2's exposure lock does not survive stream start.** The C920 firmware
   drops out of manual mode during stream init and ignores exposure set before
   streaming, so `open_c920()` reads a few frames and then re-applies
   `camera_setup.sh`. Focus survives; exposure does not.
3. **§2.4 swap is unnecessary** — the board already has 3.7 GB zram and
   422 GB free on NVMe.
4. **§5 has no input images.** `crack500_val.json` holds Colab Drive paths and
   no CRACK500 crops exist on the device; `capture_samples.py` fills the gap.
5. **§3.3 opset re-export is not needed.** TensorRT 10.3 parses the opset-18
   model directly.
6. **`--alert-frac` default raised 0.002 → 0.01.** The guide's value fires at
   the ~0.06 % speck level a clean surface produces.
7. **`clean_mask()` filters on shape, not just area.** See below.

## Two behaviours worth knowing before changing this code

**The model cannot say "this is not pavement."** It is a single-channel U-Net
sigmoid with no abstain path, trained on CRACK500 where every image contains a
crack, so its prior is "find the crack in this picture". Pointed at a person it
reported 19 % coverage — three times denser than the ~6.3 % of pixels that are
crack in an average CRACK500 mask. Shape filtering removes 96 % of that, but it
is a post-hoc filter, not a fix. A real fix needs a surface-gate classifier or
fine-tuning with hard negatives.

Shape thresholds are measured, not guessed: false blobs have median half-width
(area/perimeter) 11.2 px and median solidity 0.715; a thin crack measures
1.94 px and 0.041. **Do not tighten `--max-halfwidth` below 12** without
in-domain data — CRACK500's 6.3 % average implies real cracks up to ~25 px
wide, so tightening silently drops them.

**The C920 emits garbage frames after any control change.** At a *fixed*
`exposure=500 gain=0`, consecutive medians measured 4 and 241. `AutoExposure`
flushes in-flight frames, stays blind briefly, and smooths the metric. Removing
any of those makes the loop thrash.

Its control law also has two non-obvious constraints, both learned by breaking
them: the target must be **reachable** (median 120, not p99 235 — an
unreachable target limit-cycles on bimodal scenes), and there must be **no
one-way ratchet** anywhere. An anti-windup "ceiling" that only decreased once
drove the camera to `exposure=3, gain=255` — the darkest, noisiest image
possible — with no way back. `test_ae_sim.py` covers both.
