# CrackNet — the complete explanation

This folder explains **everything** about the project and the paper written from
it: the theory behind each technique, what was built, how every number and every
graph was obtained, and what the code does. It is written to be *learned from*,
not just looked up.

It cites the paper we wrote — **"Stationary Multi-Camera Crack Detection and
Metric Width Measurement on Edge Hardware for Tunnel-Lining Inspection"** — by
section (§), figure (Fig.) and table (Table), so you can read the paper and this
folder side by side.

> **One rule runs through all of it.** Every statement is tagged by how we know
> it. If a number is not tagged `MEASURED`, `REPRODUCED` or `DERIVED`, do not
> quote it as a result.

---

## How we know things (provenance tags)

| Tag | Meaning | Example |
|---|---|---|
| **MEASURED** | Taken on the Jetson with the real hardware and recorded in a project document | FP16 engine = 8.77 ms |
| **REPRODUCED** | Re-run locally from the repo's own code for this documentation | tiler 23/23 checks |
| **DATA** | Computed from a data file in the repo | test IoU from `test_sweep.npy` |
| **DERIVED** | Follows from a formula plus measured inputs | 0.294 mm/px at 0.40 m |
| **SCHEMATIC** | A drawing; carries no data | Fig. 1, Fig. 5, Fig. 11 |
| **ASSUMED** | A value we did not measure, stated openly | the 0.40 m standoff |

---

## Reading paths

| You have | Read, in order |
|---|---|
| **30 minutes** | [01 Big picture](01-big-picture.md) → [07 Figures & tables](07-every-figure-and-table.md) |
| **2 hours, to understand it all** | 01 → [02 Theory](02-theory-primer.md) → 03 → 04 → 05 → 06 |
| **To explain it to someone / viva** | 01 → [09 Paper walkthrough](09-paper-walkthrough.md) → [11 Q&A](11-viva-qa.md) |
| **To change the code** | [08 Code walkthrough](08-code-walkthrough.md) → [10 Reproduce](10-reproduce-and-extend.md) |
| **To write another paper** | [09](09-paper-walkthrough.md) → [10](10-reproduce-and-extend.md); also `context.md` at the repo root |

---

## The documents

| # | File | What it answers |
|---|---|---|
| 01 | [Big picture](01-big-picture.md) | What is this, why, how is it put together, what is and is not built |
| 02 | [Theory primer](02-theory-primer.md) | Every concept used, from first principles, with worked numbers |
| 03 | [Model & training](03-model-and-training.md) | The network, the data, the training curve, the threshold sweep, false positives |
| 04 | [Cameras & capture](04-camera-and-capture.md) | The USB wall, sequential capture, the exposure ladder, the controller |
| 05 | [Perception pipeline](05-perception-pipeline.md) | Tiling, blending, shape/valley/straightness filters, measuring in mm |
| 06 | [Deployment, console, network](06-deployment-console-network.md) | TensorRT, latency, the web console, the Wi-Fi access point |
| 07 | [Every figure & table](07-every-figure-and-table.md) | For each of Fig. 1–21 and Table I–XII: what it shows, where the data came from, how to read it |
| 08 | [Code walkthrough](08-code-walkthrough.md) | File by file, function by function, and the call path of one scan |
| 09 | [Paper walkthrough](09-paper-walkthrough.md) | The paper section by section, with the evidence behind each claim |
| 10 | [Reproduce & extend](10-reproduce-and-extend.md) | Commands to regenerate everything, and what to do next |
| 11 | [Viva Q&A and glossary](11-viva-qa.md) | Likely questions with grounded answers; terms defined |

`figures/` holds every figure as a PNG plus `make_figures.py`, which regenerates
all of them from the repo (see [10](10-reproduce-and-extend.md)).

---

## The whole project on one page

```
 3 × Logitech C920  (left / top / right)      ← one USB 2.0 hub, 480 Mbit/s shared
        │  read ONE AT A TIME (stationary rig)
        ▼
 [capture]  open → settle exposure → average 4 frames → release        ≈ 5.9 s
        ▼
 [tile]     1920×1080 → 15 tiles of 512×512, native scale, 15% overlap
        ▼
 [infer]    U-Net / MobileNetV3-L, TensorRT FP16, 8.77 ms per tile     ≈ 0.22 s/camera
        ▼
 [blend]    average overlapping tile LOGITS (raised cosine), threshold once
        ▼
 [verify]   shape filter → valley test → straightness test
        ▼
 [measure]  distance-transform ridge, p95 → width in millimetres
        ▼
 [console]  HTTP page: overlays, verdict, widths, log                  ≈ 6.6 s per position
```

### The four findings worth remembering

1. **Three cameras cannot stream together at 720p+ on one USB 2.0 hub** — so a
   stationary rig reads them one at a time, and the cost is time (≈ 6 s), not
   resolution. *(Table VI, Fig. 6, Fig. 16)*
2. **The camera's exposure control is a ladder, not a dial** — one stop per
   step while streaming. A controller that assumes a dial cannot converge.
   *(Fig. 7)*
3. **Tiling creates detections at tile borders, and the shape filter makes them
   worse** (2.18× → 11.33×); blending the logits removes them. *(Fig. 9, Fig. 17a)*
4. **A crack is a valley, a shadow is a step** — a two-sided minimum separates
   them exactly, which no curvature measure does. *(Fig. 18)*

### What the system is *not* (yet)

Not a calibrated instrument (widths assume a 0.40 m standoff), not validated on
a genuine crack in a continuous surface, not in focus at close range, and not
moving. [01 Big picture §5](01-big-picture.md) lists every gap.
