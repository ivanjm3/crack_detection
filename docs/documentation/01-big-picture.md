# 01 — The big picture

*Paper: §I Introduction, §III-A System overview, §III-B Hardware, §VI Conclusion.*

---

## 1. The problem, in plain words

Tunnels, bridges and walls crack. Cracks are the first visible sign of trouble,
and **how wide a crack is** — in millimetres — is what engineers grade it by. A
0.3 mm hairline is routine; a 3 mm crack is a problem.

Today this is inspected by people with torches and rulers. That is slow, risky
(tunnels, heights), and inconsistent between inspectors. The idea: put cameras
on a carriage, let a neural network find the cracks, and **report each one's
width in millimetres**.

Finding a crack in a photo is the *easy* half. The paper is about the harder
half — everything between "the network outputs a mask" and "an engineer can
trust a number":

| Hard part | Why it is hard | Where we solved it |
|---|---|---|
| **A millimetre needs a scale** | An image has pixels, not millimetres. A pixel's size on the wall depends on how far the camera is | Ground sample distance (GSD), [02 §6](02-theory-primer.md) |
| **More than one camera** | One camera sees a strip; a tunnel is wide. But combining images creates fake crack-shaped seams | Process each camera separately, [04](04-camera-and-capture.md) |
| **The network cannot say "that's not a wall"** | It was trained only on images that contain cracks, so it finds "cracks" in curtains and people | Verification filters, [05](05-perception-pipeline.md) |
| **Cheap cameras fight back** | The shutter is quantised, the USB bus is shared, the clocks reset on reboot | [04](04-camera-and-capture.md), [06](06-deployment-console-network.md) |

---

## 2. What was built

![Fig. 1 — system architecture](figures/fig_arch.png)

*Paper Fig. 1. SCHEMATIC.*

### The hardware

| Part | Detail | Why it matters |
|---|---|---|
| **NVIDIA Jetson Orin Nano 8 GB** | Small GPU computer; run in `MAXN_SUPER` power mode with clocks locked | Runs the network on the device — no cloud |
| **3 × Logitech C920** | Ordinary 1080p USB webcams, roles `left`, `top`, `right` | Cheap, but with real constraints (§4 below) |
| **Operator PC (Windows)** | Only shows the result and sends start/stop | Keeps the Jetson headless |
| **Wi-Fi access point hosted by the Jetson** | SSID `CrackNet`, address `10.42.0.1` | Campus Wi-Fi blocks device-to-device traffic |

### The rig geometry

![Fig. 5 — three cameras covering wall, crown, wall](figures/fig_geometry.png)

*Paper Fig. 5. SCHEMATIC — the intended deployment, not a measured scene.*

The three cameras are named by what they look at, not where they sit in a row:
**left** looks at the left wall, **top** at the crown, **right** at the right
wall. Together they cover a tunnel cross-section.

> **Honesty note.** This geometry is a *design*. The system has been exercised
> on a bench against indoor targets, not inside a tunnel.

### The software, as a chain

Each box feeds the next; each is explained in its own document.

```
capture ─▶ tile ─▶ infer ─▶ blend ─▶ verify ─▶ measure ─▶ console
 [04]      [05]    [03,06]   [05]     [05]      [05]       [06]
```

| Stage | One-line job |
|---|---|
| **capture** | Get one clean, averaged 1080p frame from each camera, one camera at a time |
| **tile** | Cut each frame into 512×512 crops at full resolution (the network's input size) |
| **infer** | Run the network on every crop → a "crack-likeness" score per pixel (a *logit*) |
| **blend** | Average the scores where crops overlap, then draw the crack mask |
| **verify** | Throw away detections that are not physically cracks (blobs, steps, ruled lines) |
| **measure** | For each surviving crack, compute its width and length in millimetres |
| **console** | Publish the result as a web page the operator watches |

---

## 3. How the project unfolded (chronology)

This is the order things were discovered. It matters, because several decisions
only make sense as reactions to a measurement.

| Phase | What | Outcome | Doc |
|---|---|---|---|
| **1** | One C920, centre-crop, TensorRT, MJPEG viewer | Worked at 15–19 fps, but threw away two-thirds of each frame | [06](06-deployment-console-network.md) |
| 1b | Pointed at a person: `CRACK 19 %` | Model has no "not a crack" answer → shape filter (96 % of the false coverage removed) | [03](03-model-and-training.md) |
| 1c | Auto-exposure, three designs, two failed | Median-luma loop with a two-stage ladder | [04](04-camera-and-capture.md) |
| **idea** | "Use three cameras and stitch them" | **Rejected**: resolution collapses 0.089×, and seams look like cracks | [04 §1](04-camera-and-capture.md) |
| **0** | Measure the USB bus with three cameras | 3 × 640×480 works; 3 × 720p does not | [04 §2](04-camera-and-capture.md) |
| **2** | Sequential capture → tiling → measurement → console | 6.6 s per position, mm widths | [04](04-camera-and-capture.md), [05](05-perception-pipeline.md) |
| 2b | Found the exposure ladder | Controller redesigned around 38/77/156/312/624 | [04 §4](04-camera-and-capture.md) |
| 2c | Measured tile-border artefacts | 2.18× raw → 11.33× filtered | [05 §3](05-perception-pipeline.md) |
| 2d | Blending + valley + straightness filters | Curtain scene: 18 false components → 0 | [05](05-perception-pipeline.md) |
| 2e | Reverted 2d on `main` for a teacher demo | Kept on branch `accuracy-filters` | §6 below |
| **net** | Jetson hosts its own Wi-Fi AP | PC drives it with no cable | [06 §5](06-deployment-console-network.md) |
| **paper** | Old paper described unbuilt work; rewrote around what exists | This documentation | [09](09-paper-walkthrough.md) |

---

## 4. The constraints that shaped every decision

You can predict most design choices from these four facts.

1. **One USB 2.0 hub, 480 Mbit/s, shared by all cameras.** Three 1080p streams
   will not fit. → read cameras one at a time. *(Paper Table VI)*
2. **The shutter has only five usable settings while streaming.** → gain, not
   exposure, must be the fine control. *(Paper Fig. 7)*
3. **The network was trained only on cracks.** → it cannot abstain, so we must
   verify its output ourselves. *(Paper Fig. 14, Fig. 17)*
4. **A pixel has no size until you give it one.** → every millimetre figure is
   only as good as the assumed distance to the wall. *(Paper Eq. 1, Fig. 21)*

---

## 5. What is built and what is not

Read this table before quoting anything from the paper.

| Capability | State | Evidence |
|---|---|---|
| Model trained, evaluated on CRACK500 | **Built, measured** | `crack_outputs/`, [03](03-model-and-training.md) |
| TensorRT FP16 on the Jetson | **Built, measured** | 8.77 ms/tile, parity 0.0552 % |
| Three-camera sequential capture | **Built, measured** | 5.93 s capture per position |
| Exposure control around the ladder | **Built; simulated 7/7; settled on hardware** | [04 §4](04-camera-and-capture.md) |
| Tiling and union reassembly | **Built, tested** (`main`) | `test_tiler.py` |
| Logit blending, valley, straightness | **Built, tested** (branch only) | 23/23, `test_verify.py` |
| Width in millimetres | **Built, validated on analytic shapes** | worst error 1.00 px |
| Web console, live + scan modes | **Built** | [06](06-deployment-console-network.md) |
| Self-hosted Wi-Fi AP + Windows launcher | **Built, reboot-tested** | [06 §5](06-deployment-console-network.md) |
| **Calibrated millimetres** | **Not done** — assumes 0.40 m | open |
| **Recall on a real crack in a continuous surface** | **Not measured** | open |
| **Focus** | **Unresolved** — pinned near infinity | open |
| ArUco wall-frame mapping, crack IDs, change detection | **Not built** (were in the *old* paper) | none |
| EV3 rover / rail trials, Pi 5 comparison | **Not built / not done** (old paper) | none |
| Moving carriage | **Not built** — every result is stationary | none |

**Why the old paper was rewritten.** It described the last four rows as if they
existed. Nothing in the repository implements them. The new paper moves them to
Future Work (paper §VII) and reports only what was built and measured.

---

## 6. The two branches

| Branch | Contains | Use |
|---|---|---|
| `main` | Tiling with **union** reassembly, shape filter only | The **demo** configuration |
| `accuracy-filters` | + logit **blending**, **valley** and **straightness** tests, `explain.py`, `focus.py`, `inject_crack.py` | The configuration the paper's accuracy results describe |

**Why two.** A cardboard sample with a visible crack was flagged on `main` and
not on `accuracy-filters`. Diagnosis showed the filters were behaving correctly
(the "crack" was a torn edge — a step — and the frame was badly defocused), but
for a teacher demonstration the more sensitive behaviour was wanted. See
[05 §9](05-perception-pipeline.md) for the full case.

This is a good thing to understand: **the filters trade sensitivity for
precision, and which side you want depends on what a miss costs.**

---

## 7. Where each part of the paper comes from

| Paper section | Based on | Documented in |
|---|---|---|
| §III-D multi-camera capture | `capture.py`, Phase 0 probe | [04](04-camera-and-capture.md) |
| §III-E photometric control | `camera_ctl.py`, `test_ae_sim.py` | [04](04-camera-and-capture.md) |
| §III-F model | notebooks → `crack_outputs/` | [03](03-model-and-training.md) |
| §III-G tiling, blending | `tiler.py` | [05](05-perception-pipeline.md) |
| §III-H verification | `live.py:clean_mask`, `verify.py` | [05](05-perception-pipeline.md) |
| §III-I measurement | `measure.py` | [05](05-perception-pipeline.md) |
| §III-J console / network | `serve_scan.py`, `netlink.sh`, `cracknet.bat` | [06](06-deployment-console-network.md) |
| §IV Results | tests, benches, `crack_outputs/` | [07](07-every-figure-and-table.md) |
