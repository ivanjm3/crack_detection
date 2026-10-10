# 09 — The paper, section by section

**"Stationary Multi-Camera Crack Detection and Metric Width Measurement on Edge
Hardware for Tunnel-Lining Inspection."**

For each section: what it argues, what evidence supports it, which document
explains the detail, and what to be careful of. Read this with the paper open.

---

## The argument in five sentences

1. A crack mask is not an inspection result; a **width in millimetres you can
   trust** is.
2. Getting there on cheap edge hardware means beating four physical limits:
   **bus bandwidth, exposure quantisation, tile seams, and a model that cannot
   abstain**.
3. Each limit was *measured*, and a specific design answers each.
4. The result is a pipeline that measures width to within a pixel on known
   shapes, runs a scan in ≈ 6.6 s, and rejects the curtains and seams that fooled
   the raw model.
5. It is **not yet** a calibrated instrument: widths assume a standoff, focus is
   unresolved, and no real crack in a continuous surface has been through it.

---

## Abstract

| Claim | Number | Evidence |
|---|---|---|
| Three cameras on one USB 2.0 hub, read one at a time | three simultaneous streams fail at 720p (only 2 of 3 deliver) | Table VI |
| MobileNetV3-L U-Net, TensorRT FP16 | 8.77 ms/tile | Table V |
| Test IoU / Dice | 0.652 / 0.790 | Table III |
| Scan position | ≈ 6.6 s, 90 % capture | Fig. 16 |
| Border artefacts after shape filtering | 11.3× interior | Fig. 17(a) |
| Verification filters on a curtain scene | 18 → 0 | Fig. 17(c) |
| Limitations stated | assumed standoff, focus, no real-crack validation | §V-C |

*Be careful.* The Abstract says "the model reaches a test IoU of 0.652" — that is
**CRACK500 pavement**, not the deployment domain. The next sentence-group
supplies the caveat; keep them together when quoting.

---

## I. Introduction

| Sub | Says | Detail in |
|---|---|---|
| A. Motivation | Cracks matter; width grades them; manual inspection is slow/inconsistent | [01](01-big-picture.md) |
| B. Gap between mask and measurement | Three missing things: scale, abstention, coverage without seam artefacts | [01 §1](01-big-picture.md) |
| C. Contributions | Six, each tied to a measured result | below |
| D. Scope | Stationary; flat/gently curved lining; motion/registration out of scope | [01 §5](01-big-picture.md) |

**The six contributions and where each is demonstrated**

| # | Contribution | Evidence |
|---|---|---|
| 1 | Sequential capture on one USB 2.0 hub, ports not device numbers | Table VI, Fig. 16 |
| 2 | Exposure actuator is a 1-stop ladder; controller redesigned | Fig. 7, Table VII |
| 3 | Native tiling + logit blending; border artefact measured | Figs. 8, 9, 17(a) |
| 4 | Valley and straightness verification | Figs. 18, 19, 17(c) |
| 5 | Width from the distance-transform ridge, validated on known shapes | Fig. 20, Table X |
| 6 | An open account of validity | §V-C |

---

## II. Related Work

| Sub | Cited for | Takeaway |
|---|---|---|
| A. Vision-based detection | FCN, U-Net variants, patch CNNs, benchmarks | all image-level; none asks "how wide in mm" |
| B. Efficient / edge | MobileNet, YOLOv8, dual-attention, prompt-based, hardware studies, Jetson survey | feasible and priced; stops at the mask |
| C. Spatially grounded / tunnel | LiDAR/SLAM fusion, tunnel photogrammetry, metro-tunnel CNNs, change detection | capable but multi-sensor and costly |
| D. Multi-camera, tiling | stitching [Brown, Szeliski]; tiling [Unel, Akyon] | stitching is a poor fit; tiling is the standard remedy |
| E. Width, fiducials | distance transform; Steger/Frangi; ArUco; calibration | the measurement tools used here |
| F. Positioning | Table I | the one row with *width in mm* and *assumed-standoff grounding* |

*Be careful.* Table I is **qualitative** — drawn from titles/abstracts, with no
benchmark. Do not describe it as a comparison of performance. All 42 references
were verified against Crossref/DataCite (see [10](10-reproduce-and-extend.md)).

---

## III. Methodology

| § | Title | What it establishes | Doc | Figs/Tables/Eqs |
|---|---|---|---|---|
| A | System overview | five stages + console | [01](01-big-picture.md) | Fig. 1 |
| B | Hardware | Jetson, 3 × C920, one hub | [01](01-big-picture.md) | Figs. 2–4, Table II |
| C | Rig geometry & resolution budget | GSD formula; 0.59 mm at 0.4 m | [02 §5](02-theory-primer.md) | Fig. 5, Eq. 1 |
| D | Multi-camera capture | no stitching; sequential; role mapping | [04](04-camera-and-capture.md) | Fig. 6 |
| E | Photometric control | ladder + controller | [04 §4–5](04-camera-and-capture.md) | Fig. 7 |
| F | Model & deployment | architecture, data, loss, logit threshold | [03](03-model-and-training.md) | Eq. 2 |
| G | Tiling & reassembly | native tiles; clamping; logit blending | [05 §2–5](05-perception-pipeline.md) | Figs. 8, 9, Eq. 3 |
| H | Candidate verification | shape, valley, straightness | [05 §6–7](05-perception-pipeline.md) | Fig. 10, Eq. 4 |
| I | Metric measurement | ridge p95; 2d−1; length | [05 §8](05-perception-pipeline.md) | Eq. 5 |
| J | Console & network | publish-once, AP | [06](06-deployment-console-network.md) | Fig. 11 |
| K | Evaluation protocol | four things, in order of how well they can be validated | below | — |

**§III-K's honesty is structural.** It lists what *can* be validated here (model
accuracy, deployment cost, measurement accuracy on known shapes, false positives
on crack-free scenes) and says plainly that **genuine-crack recall could not be
validated**. Results §IV mirrors that order.

---

## IV. Results

| § | Result | Number | Verdict |
|---|---|---|---|
| A | Segmentation accuracy | test IoU 0.652, Dice 0.790 | **Strong** — leak-free split, threshold chosen on val |
| A | Threshold sensitivity | IoU 0.646–0.652 over 0.30–0.90 | **Strong** — fully from data |
| B | Crack-free surfaces | 10.85 % of 20 000 fire; median 165 px | **Strong** — large n |
| C | Edge deployment | 8.77 ms; 2.1×; parity 0.055 % | **Strong** — measured on device |
| D | Capture layer | 3/3 vs 2/3; 5.93 s capture | **Strong** — measured on device |
| D | Exposure sim | 7/7 | **Moderate** — first control law; simulation |
| D | Exposure on hardware | exp 156, gain 0, luma 94–99 | **Moderate** — one scene |
| E | Tile border artefacts | 2.18× → 11.33× | **Moderate** — 3 frames, indoor |
| E | Blending | coverage 75.2→63.7 etc. | **Weak** for accuracy — out-of-domain frames, no recall |
| F | Person: 38 → 1.44 % | 96.2 % removed | **Moderate** — one scene type |
| F | Curtain: 18 → 0 | valley removes all | **Moderate** — one scene |
| G | Width accuracy | within 0.35 px (non-taper), worst 1.00 px | **Strong** — analytic truth, but synthetic |
| H | Resolution vs standoff | 6.6 mm at 4.5 m | **Strong** — arithmetic, *given* the FOV |
| I | Cardboard case | raw 0.27 %, step, defocus | **Illustrative** — one sample |
| J | Code verification | 23/23, 13 cases, 7/7 | **Strong** — reproduced |

Verdict key: *Strong* = directly measured or exact; *Moderate* = measured but
narrow; *Weak/Illustrative* = does not support a general claim.

---

## V. Discussion

| § | Says |
|---|---|
| A | What the results support: the resolution budget, the capture cost, the artefact rates |
| B | **Why a post-hoc filter, and what it costs** — a network trained only on cracks cannot abstain; the filters will discard a crack whose profile looks like a step |
| C | **Threats to validity** (list below) |
| D | Operational findings: silent failures |

### The threats list (§V-C) and what each protects against

| Threat | If ignored, a reader might conclude… |
|---|---|
| Scale is uncalibrated (0.40 m assumed) | "widths are accurate to a pixel" — they are accurate to a pixel *in pixels* |
| No genuine crack in a continuous surface | "the system finds 82 % of cracks" — that is CRACK500 recall |
| Focus unresolved | "accuracy is as measured" — out-of-focus frames measure the blur |
| One camera model / revision | "the ladder is a property of C920s" |
| FOV parity unverified | "mm/px is exact" |
| Out-of-domain FP numbers from a few indoor scenes | "FP rate on tunnels is 0" |
| Stationary only | "this works while moving" |

---

## VI. Conclusion and VII. Future Work

**Conclusion.** Restates the measured results and ends: *"Those gaps, rather than
the detector, now bound its usefulness."* That sentence is the paper's thesis in
miniature.

**Future Work** — each item answers a threat above or a removed old-paper claim:

| Item | Closes |
|---|---|
| Calibration | scale threat |
| Focus target + sharpness gating | focus threat |
| Genuine-crack validation | recall threat |
| Hard negatives / surface gate | no-abstain problem |
| Fiducial wall-frame registration | old paper's ArUco claim (now future) |
| Repeat-pass change detection | old paper's change-detection claim |
| Close-range acquisition (mast, telephoto) | the 4.5 m resolution limit |
| GPU preprocessing, INT8 | performance |
| Moving carriage, stop-and-scan | stationary-only |

*No simulation or Gazebo appears anywhere in the paper, by instruction.*

---

## Claim → evidence: the ones people will challenge

| If asked… | Point to |
|---|---|
| "How do you know the third camera fails?" | Table VI; `multicam_probe.py`; quirks already `0xFFFFFFFF` |
| "Why not just stitch?" | [04 §1](04-camera-and-capture.md): 0.089× scale; seams are crack-shaped |
| "Is the exposure ladder real?" | Fig. 7: measured 38/77/156/312/624 with a stream open |
| "Is blending actually better?" | Fig. 9 (mechanism) + Fig. 17(a) (artefact); **no recall claim** |
| "How accurate is the width?" | Fig. 20/Table X: ≤ 0.35 px non-taper, 1.00 px worst; *on synthetic shapes* |
| "Is it accurate in mm?" | **Only given the standoff** — [01 §5](01-big-picture.md) |
| "Does it find real tunnel cracks?" | **Not measured.** Say so |
| "Why 0.55?" | Best validation Dice; chosen without the test split |

---

## What changed from the old paper

| Old paper | New paper |
|---|---|
| ArUco wall-frame mapping, persistent IDs | **Future Work** (not built) |
| Repeat-pass change detection | **Future Work** (not built) |
| EV3 rover + rail trials | removed (not built) |
| CPU-only vs GPU comparison (Pi 5) | removed (not done) |
| Ended at Methodology (no Results) | full IMRaD: Results, Discussion, Conclusion, Future Work |
| 14 references, several untitled/mis-titled | 42, all registry-verified |
| Title "Motion-Aware … Mapping" | "Stationary Multi-Camera … Metric Width Measurement …" |

The old paper's problem was never its writing; it was that **its central
contributions had no code and no data behind them.**
