# 03 — The model, its training, and its limits

*Paper: §III-F (model and deployment), §IV-A (segmentation accuracy), §IV-B
(crack-free surfaces); Figs. 12, 13, 14; Tables III, IV.*

The model is **frozen** — the contribution of the paper is the system around it
— but every claim about accuracy and false positives rests on understanding how
it was made and what it was never shown.

---

## 1. What the model is

| Property | Value | Source |
|---|---|---|
| Framework | `segmentation_models_pytorch` `smp.Unet` | `setup.md.txt` |
| Encoder | `timm-mobilenetv3_large_100`; ImageNet mean/std are applied, which implies ImageNet pre-training (the inference build uses `encoder_weights=None` because the trained weights are loaded over it) | `setup.md.txt`, WORKLOG §4 |
| Decoder channels | (256, 128, 64, 32, 16) | `setup.md.txt` |
| Output | **1 channel of logits** | `setup.md.txt` |
| Input at deployment | 1×3×512×512, ImageNet mean/std | `trt_infer.py` |
| Exported as | ONNX opset 18 → TensorRT FP16 engine | WORKLOG §4 |

Concepts: [02 §2](02-theory-primer.md).

```
RGB 512×512 ─▶ MobileNetV3-L encoder ─▶ U-Net decoder ─▶ 1 logit per pixel
                  (ImageNet weights)     (skips from encoder)
```

---

## 2. The data

### CRACK500

CRACK500 [Yang 2020] is a set of **pavement** crack photographs with pixel masks,
taken from roughly 1 m with a phone. It is what the model learned.

| Fact | Value |
|---|---|
| Crops used | 2333 train / 526 validation / 509 test |
| Crack pixels | ≈ 6.3 % overall, median 5 % per image |
| Empty masks | none — **every image contains a crack** |

That last row is the root of the model's false-positive behaviour ([§7](#7-what-the-model-cannot-do)).

### The split was rebuilt — and why

The *official* CRACK500 splits put crops of the **same photograph** in both train
and test (5 parent photos leak train↔test, 1 leaks val↔test). A network then
"recognises" the test photo rather than generalising, inflating the scores.

The splits were rebuilt **by parent photograph**, 70/15/15, giving 2333 / 526 /
509 crops with **zero overlap**. This is why the accuracy in Table III is
believable.

### Augmentation

| Augmentation | Why |
|---|---|
| RandomResizedCrop to 448 | scale variety |
| Flips, 90° rotations | cracks have no preferred orientation |
| Brightness / contrast | lighting variety |
| Noise | sensor noise |
| **Motion blur** | webcam footage, camera shake |
| **JPEG compression** | MJPG from the C920 is JPEG |

The last two were chosen specifically because the deployment camera is a webcam
producing MJPG.

---

## 3. Training

| Setting | Value | Evidence |
|---|---|---|
| Loss | Dice + focal | `setup.md.txt` |
| Epochs | 60 | `train_log.csv` |
| LR schedule | warm-up over 3 epochs to 3×10⁻⁴, then cosine to ≈ 0 | `train_log.csv` `lr` column |
| Time | 1.22 h total; ≈ 63.5 s/epoch (epoch 0 took 614 s) | `train_log.csv` `secs` |
| **Not recorded** | optimiser, batch size, hardware, Dice/focal weights | — |

> The first epoch took ten times longer than the rest (614 s vs ≈ 63 s). That is
> consistent with one-time start-up cost (data caching, kernel warm-up), but the
> log does not say which.

### Reading the learning-rate column (DATA)

```
epoch   lr
  0     9.9e-5   ┐ linear warm-up
  1     2.0e-4   │
  2     3.0e-4   ┘ peak 3.0e-4 (epoch 3)
 ...    cosine decay
 59     1.1e-11  ≈ 0
```

---

## 4. The training curve — Fig. 12

![Fig. 12](figures/fig_training.png)

**What it shows.** Left axis (grey): training loss. Right axis: validation IoU
(blue) and Dice (orange), one point per epoch. Red dot: best IoU.

**Where the data came from.** `crack_outputs/train_log.csv`, columns
`epoch, train_loss, val_iou, val_dice` (DATA). Nothing is smoothed.

**How to read it.**

| Observation | Number | Meaning |
|---|---|---|
| Loss falls fast then slowly | 0.96 → 0.34 (ep 3) → 0.28 (ep 10) → 0.225 (ep 59) | steady convergence |
| IoU climbs quickly | 0.18 (ep 0) → 0.62 (ep 3) | most learning in 3 epochs |
| IoU plateaus | > 0.64 from epoch 14 | little left to gain |
| Last 20 epochs | mean 0.6437, **std 0.0016** | stable, not overfitting |
| Best | **0.6505 at epoch 34** | marginal; final = 0.6434 |

**What it does *not* show.** Validation is a *different split*, not different
domain: the curve says the model generalises across CRACK500, not to tunnels.

---

## 5. Operating point — Fig. 13, Tables III & IV

![Fig. 13](figures/fig_sweep.png)

### How the sweep was obtained

`val_sweep.npy` and `test_sweep.npy` are arrays of shape **(5, 17)** (DATA):

| Row | Contents |
|---|---|
| 0 | thresholds 0.10, 0.15, …, 0.90 |
| 1 | precision |
| 2 | recall |
| 3 | Dice |
| 4 | IoU |

Verified: for every column, Dice = 2PR/(P+R) and IoU = Dice/(2 − Dice)
reproduce the stored values, so the row order above is certain.

```python
import numpy as np
v = np.load("crack_outputs/val_sweep.npy")
thr, P, R, dice, iou = v
k = int(np.argmin(abs(thr - 0.55)))
print(P[k], R[k], dice[k], iou[k])   # 0.7752 0.8101 0.7923 0.6560
```

### The numbers

| | Precision | Recall | Dice | IoU |
|---|---|---|---|---|
| **Validation @ 0.55** | 0.7752 | 0.8101 | **0.7923** | 0.6560 |
| **Test @ 0.55** | 0.7609 | 0.8206 | **0.7896** | **0.6524** |
| Test, best Dice (T = 0.65) | — | — | 0.7897 | 0.6525 |

**Why 0.55.** It is the threshold with the best validation Dice (0.7923) — it was
chosen *without looking at the test split*, so the test row is an honest
held-out number. On the test split the best threshold would be 0.65, but the gain
is 0.0001 Dice: the curve is flat.

### What the curves say

- **Dice and IoU are almost flat** across 0.30–0.80 (range 0.777–0.790 on test).
  The threshold does not change *how accurate* the model is.
- It trades **precision for recall**: raising T from 0.30 to 0.90 gives
  **+6.4 points precision** and **−7.5 points recall** (test).
- Precision and recall cross near T ≈ 0.80 (test).

> **Operational consequence.** For inspection a missed crack usually costs more
> than a false alarm, which argues for a *lower* threshold than the Dice-optimal
> one. The project nevertheless keeps 0.55 and puts its false-alarm control
> *after* the network (the verification filters) — so the network can stay
> sensitive.

### Two sets of IoU values

| Source | Best val IoU |
|---|---|
| Training log (`train_log.csv`) | 0.6505 |
| Sweep (`val_sweep.npy`) at 0.55 | 0.6560 |

They are not identical because they come from different scripts (the paper says
so under Table III). Quote the sweep for the operating point, the log for the
training curve, and never mix them in one sentence.

---

## 6. Crack-free surfaces — Fig. 14

![Fig. 14](figures/fig_metu.png)

### The experiment

The model was run on **20,000 crack-free images** from the METU building-crack
dataset [Özgenel 2018] (negative class), at $t=0.55$. Any pixel above threshold
is a false positive.

### Result (DATA: `metu_negative_pixels.npy`)

| Quantity | Value |
|---|---|
| Images with ≥ 1 false-positive pixel | **10.85 %** (n = 2169 stored) |
| Median false blob | **165 px** |
| Mean | 734 px |
| 90th / 99th percentile | 2003 / 8297 px |
| Maximum | 17,040 px |
| Mean as fraction of a 227×227 image | 1.4 % |

### How to read the histogram

The x-axis is **logarithmic**. The distribution is a broad hump centred on
~100–300 px with a long right tail. The mean (734) lies to the right of the
median (165) because a few huge blobs dominate it — which is exactly the
case a person or curtain produces.

**Takeaway.** About 1 clean image in 9 fires. A min-area filter handles the
median (165 px) but not the tail — hence the shape filters in
[05](05-perception-pipeline.md).

---

## 7. What the model cannot do

### It cannot say "this is not a crack surface"

Trained only on images containing cracks, it has no abstain path ([02 §16](02-theory-primer.md)).
Shown a person it reported **38 % coverage**; a real CRACK500 crack averages
6.3 %. The false signal was ~6× denser than the true one.

### It has never seen the deployment domain

| Training | Deployment |
|---|---|
| Pavement, ~1 m, phone camera | Tunnel lining / indoor test targets, webcam |
| Natural daylight | Dim, uneven, artificial light |
| Rich texture | Smooth concrete, painted surfaces |

No in-domain cracks were available on the device, so **no recall number exists
for the deployment domain**. This is the first limitation the paper states.

### Shape thresholds were measured, not guessed

From 51 components in `analyze_blobs.py`:

| | synthetic thin crack | false blobs |
|---|---|---|
| half-width (area/perimeter) | 1.94 px | median 11.2 px, max 43 |
| solidity | 0.041 | median 0.715, max 1.09 |

A near-miss worth remembering: the blobs cluster at half-width 11, tempting a
limit of 8 px. But CRACK500's 6.3 % mean coverage implies genuine cracks up to
~25 px wide (half-width ≈ 12), so 8 px would have silently dropped real wide
cracks. The limit was kept at 12 px (later expressed as 15 mm).

---

## 8. How the model reaches the Jetson

```
best.pt ──export──▶ cracknet.onnx (+ .data) ──trtexec──▶ cracknet_fp16.engine
 (PyTorch)            opset 18, 1×3×512×512                  (on the Jetson)
```

| Check | Result |
|---|---|
| ONNX contract | input [1,3,512,512] → logits [1,1,512,512] |
| FP16 engine | 8.77 ms, 113.7 qps |
| FP32 engine | 18.44 ms |
| TensorRT vs ONNX Runtime | 0.0552 % mask disagreement (bar 0.1 %), 38 % of pixels above threshold |
| Original notebook parity | 50 val images: max probability diff 2×10⁻⁵, 0 mask pixels differ |

The engine is never copied between machines — it is tied to the GPU and the
TensorRT version; `build_engines.sh` regenerates it.

---

## 9. Summary: what you may and may not claim

| You may say | Evidence |
|---|---|
| "Test IoU 0.652, Dice 0.790 on a leak-free CRACK500 split at T = 0.55" | `test_sweep.npy` |
| "The threshold was chosen on validation" | val best Dice at 0.55 |
| "10.85 % of crack-free images produced a detection" | METU, `setup.md.txt` |
| "FP16 is 2.1× faster and agrees with ONNX to 0.055 %" | WORKLOG §4 |

| You may **not** say | Why |
|---|---|
| "The system detects 82 % of tunnel cracks" | recall 0.82 is on CRACK500 pavement, not tunnels |
| "Trained with Adam, batch 16 …" | not recorded |
| "The model rejects non-crack surfaces" | it cannot; the filters do |
