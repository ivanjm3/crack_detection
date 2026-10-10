# 02 — Theory primer

Every concept the paper uses, built up from first principles, each with a
**worked example using this project's real numbers**. Skip what you know; the
sections are independent.

*Paper: §III (all), Eqs. 1–5.*

| § | Topic | Used for |
|---|---|---|
| 1 | Segmentation, logits, thresholds | the model's output |
| 2 | U-Net and MobileNetV3 | the network |
| 3 | Losses: why Dice and focal | training under imbalance |
| 4 | Precision, recall, Dice, IoU | accuracy numbers |
| 5 | Field of view and ground sample distance | millimetres |
| 6 | Exposure, gain, luma, clipping | image quality |
| 7 | USB bandwidth | why cameras are read one at a time |
| 8 | Quantised actuators | the exposure ladder |
| 9 | Averaging frames | noise |
| 10 | Edge effects, tiling, blending | tile borders |
| 11 | Connected components and shape | the shape filter |
| 12 | Distance transform | width |
| 13 | Hessian, valleys, steps | the valley test |
| 14 | Principal axis, straightness | the straightness test |
| 15 | FP16 and TensorRT | speed |
| 16 | Out-of-distribution | why the model cannot abstain |
| 17 | Defocus | the focus problem |

---

## 1. Segmentation, logits and thresholds

**Segmentation** means labelling every pixel. For cracks it is binary: each pixel
is *crack* or *not crack*. The network does not output a label directly. For each
pixel it outputs one real number, the **logit** $z$, which can be any value from
$-\infty$ to $+\infty$:

```
 z very negative  ──── "definitely not a crack"
 z ≈ 0            ──── "no idea"
 z very positive  ──── "definitely a crack"
```

A **sigmoid** squashes the logit into a probability:

$$p = \sigma(z) = \frac{1}{1+e^{-z}}$$

A **threshold** $t$ turns it into a decision: crack if $p > t$. This project
uses $t = 0.55$.

### Why the sigmoid is never computed (paper Eq. 2)

$\sigma$ is monotonic — bigger $z$ always gives bigger $p$ — so "$p > t$" is
exactly the same test as "$z > \ell$" where $\ell = \sigma^{-1}(t)$:

$$\ell = \ln\frac{t}{1-t}$$

**Worked example.** For $t = 0.55$:

```
ℓ = ln(0.55 / 0.45) = ln(1.2222) = 0.2007
```

So the code compares logits against `0.2007` and skips the sigmoid entirely.
`trt_infer.py` implements this as `logit(p)`. The result is *pixel-exact* — 0 of
3,932,160 pixels differed over a full 15-tile frame — and saved 22 % of
inference time (169 ms → 130 ms per 1080p camera). **[MEASURED]**

There is a second reason to keep logits: when tiles overlap we *average* them,
and averaging is better done on the raw evidence (logits) than on squashed
probabilities (§10).

---

## 2. The network: U-Net with a MobileNetV3 encoder

A **U-Net** [paper ref. Ronneberger] is an *encoder–decoder*:

```
input 512×512×3
  │
  ▼ encoder  ── shrinks the image, learns "what" (texture → shape → object)
  │   ╲  skip connections copy fine detail across
  ▼    ╲
  bottleneck ╲
  │           ╲
  ▼ decoder  ──▶ grows back to 512×512, learns "where"
  ▼
output 512×512×1   (one logit per pixel)
```

- The **encoder** is **MobileNetV3-Large** [Howard 2019]: a small, fast network
  built from depth-wise separable convolutions, designed for phones. In this
  project it is `timm-mobilenetv3_large_100`; ImageNet mean/std are applied to
  inputs, which implies ImageNet pre-training.
- The **decoder** has channels `(256, 128, 64, 32, 16)`; each stage doubles the
  resolution.
- **Skip connections** matter for cracks: a crack is 2–3 pixels wide, and the
  deep layers have long since lost that detail. The skips hand the fine spatial
  detail straight to the decoder.

**Why this network.** Small enough to run in 8.77 ms on a Jetson, good enough at
thin structures because of the skips. It was *not* chosen to be state of the art;
the paper's claim is about the system around it.

---

## 3. Losses: why Dice and focal

A crack image is almost all background — about **6.3 %** of pixels are crack
(median 5 %). A network that says "no crack anywhere" is **93.7 % accurate** and
useless. Plain pixel-wise cross-entropy is dominated by the easy background, so
two losses are combined.

**Dice loss** [Milletari 2016] measures *overlap* rather than pixel accuracy:

$$\text{Dice} = \frac{2\,|P \cap G|}{|P| + |G|}, \qquad L_{\text{Dice}} = 1-\text{Dice}$$

Predicting all-background scores Dice = 0 however many background pixels it gets
right, so it cannot be gamed by ignoring the minority class.

**Focal loss** [Lin 2017] down-weights pixels the network is already confident
about:

$$FL = -(1-p_t)^{\gamma}\,\log p_t$$

An easy background pixel has $p_t \approx 1$, so $(1-p_t)^\gamma \approx 0$ and it
barely contributes; hard pixels near the crack edge dominate.

The weights between the two are **not recorded** anywhere in the repo.

---

## 4. Precision, recall, Dice, IoU — with one worked example

Compare predicted crack pixels to ground truth. Count:

| | truth = crack | truth = background |
|---|---|---|
| **predicted crack** | TP (true positive) | FP (false positive) |
| **predicted background** | FN (false negative) | TN |

| Metric | Formula | Question it answers |
|---|---|---|
| **Precision** | TP / (TP + FP) | Of what I called crack, how much was? |
| **Recall** | TP / (TP + FN) | Of the real crack, how much did I find? |
| **Dice** (F1) | 2TP / (2TP + FP + FN) | Harmonic mean of the two |
| **IoU** | TP / (TP + FP + FN) | Overlap ÷ union |
| relation | IoU = Dice / (2 − Dice) | IoU is always the lower of the two |

**Worked example — the paper's test numbers** (Table III): precision 0.7609,
recall 0.8206 at $t = 0.55$. Take TP = 1000:

```
FP = 1000 × (1/0.7609 − 1) = 314.2
FN = 1000 × (1/0.8206 − 1) = 218.6

Dice = 2000 / (2000 + 314.2 + 218.6) = 0.7897   ✓ (table: 0.7896)
IoU  = 1000 / (1000 + 314.2 + 218.6) = 0.6524   ✓ (table: 0.6524)
```

That the table's four numbers are mutually consistent is how we know the
sweep array was read correctly (DATA).

---

## 5. Field of view and ground sample distance (paper Eq. 1)

A camera sees a cone. At distance $d$ the cone is $2\,d\,\tan(\text{HFOV}/2)$
wide — the **swath**. Divide by the pixel count and you get how much wall each
pixel covers: the **ground sample distance (GSD)**.

$$\text{GSD} = \frac{2\,d\,\tan(\text{HFOV}/2)}{W}$$

```
        camera
          /\
         /  \        HFOV
        /    \
       /      \
      /________\  ← wall, swath = 2·d·tan(HFOV/2)
     |<-- W pixels -->|
```

**Worked example** (HFOV = 70.42°, W = 1920 px, d = 0.40 m):

```
tan(35.21°) = 0.7057
swath = 2 × 0.40 × 0.7057 = 0.5646 m = 564.6 mm
GSD   = 564.6 / 1920      = 0.294 mm/px
```

### The 2-pixel rule

A feature must span about **two pixels** to be resolved. So the smallest
resolvable crack is $2\times\text{GSD}$ = **0.59 mm** at 0.40 m.

### Why 70.42° and not 78°

The C920 datasheet gives 78°, but that is the **diagonal** field. The horizontal
field of a 16:9 sensor is smaller:

```
tan(H/2) = tan(78°/2) × 16/√(16²+9²) = 0.8098 × 0.8716 = 0.7057  →  H = 70.42°
```

Using 78° where 70.42° belongs inflates the swath by $0.8098/0.7057 = 1.148$,
i.e. **+14.8 %**, and every width reported inherits it.

> **Correction.** Earlier project documents (and a code comment in `measure.py`)
> say "~11 %". That is the ratio of the *angles* ($78/70.42 = 1.108$), not of the
> swaths. Swath scales with $\tan$, not the angle. The paper and `context.md`
> now say 15 %.

### The linear-in-distance trap

$\text{GSD} \propto d$. If the camera is really at 0.15 m but the software
assumes 0.40 m, every width is reported **0.40/0.15 = 2.67×** too large. This
is the single biggest caveat in the paper (§V-C).

---

## 6. Exposure, gain, luma and clipping

Three quantities set how bright an image is:

| Control | What it is | Cost |
|---|---|---|
| **Exposure** | How long the sensor collects light | Motion blur if too long |
| **Gain** | Electronic amplification after the sensor | Amplifies noise too |
| **Aperture** | Fixed on a webcam | — |

**Luma** is image brightness (0–255 per pixel). The control loop uses the
**median luma**, aiming for 120 (mid-grey).

**Clipping.** A pixel cannot exceed 255. If the sensor saturates, many pixels
sit at 255 and the information that distinguishes them is gone *at the sensor*:
no later processing recovers it. A hairline crack is a faint dark line on a
bright surface; if that surface clips, the line's contrast is destroyed.

**Why noise matters here.** Gain scales signal and noise together. Sensor noise
is speckle; amplified, it forms thin bright streaks — which is exactly what a
crack detector responds to. So *high gain creates false cracks*. The controller
therefore treats gain as a last resort (§8).

---

## 7. USB bandwidth

USB 2.0 signals at 480 Mbit/s, but video uses *isochronous* transfers, with a
practical payload of about **320 Mbit/s** (a rule of thumb from the project's
reference; the exact figure depends on the host controller).

**Uncompressed video** needs:

```
YUYV 1280×720, 16 bit/pixel, 30 fps = 1280 × 720 × 16 × 30 = 442.4 Mbit/s   > 320
```

It does not fit. **MJPG** compresses in the camera (≈ 45 Mbit/s here) and does.

But the harder wall is *reservation*, not data rate. UVC reserves bus time at
stream start **based on what the camera firmware declares as its maximum**, not
what it actually sends. A C920 declares a lot. The third camera asks for a slot
that is no longer available: it opens, and never delivers a frame.

```
camera 1 reserves ████████
camera 2 reserves ████████
camera 3 asks for ████████   ← no room left → "opens" but 0 frames
```

The usual fix (a driver quirk that computes real bandwidth) was already enabled
(`quirks = 0xFFFFFFFF`) and the third camera still failed. **[MEASURED]**

**The way out.** If the scene is static, reading the cameras *one after another*
is indistinguishable from reading them at once. Release each before opening the
next, so only one reservation exists at a time.

---

## 8. Quantised actuators and control

A **control loop** measures an output, compares to a target, and adjusts an
actuator. It assumes the actuator is roughly continuous. The C920's exposure is
not:

```
requested:  3 ... 38 ... 77 ... 156 ... 312 ... 624 ... 2047   (idle driver accepts all)
applied while streaming:   38   77   156   312   624           (snaps to these)
```

Each rung is exactly **twice** the previous — one photographic *stop*. Because
brightness is roughly proportional to exposure, median luma can only change by
factors of about two.

**Consequence.** Aim for luma 120 when the reachable values are, say, 99 and 148:
the loop steps up, overshoots, steps down, overshoots — forever. No gain setting
fixes it; the target is *between* the available values.

**The design that works.**

1. Snap every request to a real rung, so the controller's record matches the
   hardware.
2. Step a rung only if the *predicted* result does not overshoot.
3. Let **gain** (continuous) do the trimming.
4. Widen the tolerance to ±30 — a tolerance finer than the actuator is an
   instruction to spend gain, and gain makes noise.

This is the paper's Fig. 7 and Table VII; [04 §4](04-camera-and-capture.md)
walks through the code.

---

## 9. Averaging frames

A sensor's noise is random frame to frame; the scene (static rig!) is not.
Averaging $N$ independent frames leaves the scene unchanged and shrinks noise by
$\sqrt{N}$:

| N | noise reduction |
|---|---|
| 1 | 1.0× |
| 4 (used) | **2.0×** |
| 16 | 4.0× |

Two requirements: the scene must not move, and the frames must be *independent*
(reading faster than the camera produces returns the same frame twice and
averages nothing — duplicates are detected and re-read). Output is saved as PNG,
not JPEG, because re-compressing would put different noise back.

---

## 10. Edge effects, tiling and blending (paper Eq. 3)

### Why tile

The network takes 512×512. A 1920×1080 frame is not that. Options:

| Option | Result |
|---|---|
| Resize 1920×1080 → 512 | scale 0.27; a 2-px crack becomes 0.5 px — gone |
| Centre-crop then resize (Phase 1) | threw away 2/3 of the frame, scale 0.711 |
| **Tile at native scale** | nothing resized, nothing cropped: 15 tiles |

### The edge effect

A convolution at the edge of its input has no pixels beyond it, so the network
fills in padding that is not in the scene. Predictions there are less reliable —
and *systematically* so: they invent detections along the tile border.

```
   tile A                 tile B
 ┌──────────┐          ┌──────────┐
 │          │ ← border │          │
 │  good    │ artefact │  good    │
 └──────────┘          └──────────┘
```

Measured: detections within 2 px of a tile border are **2.18×** more frequent
than in the interior on the raw mask. **[MEASURED]**

### The fix: weight by confidence

Give each pixel a weight that is 1 in the middle of a tile and falls towards its
edge — a **raised cosine**:

$$w(u)=\tfrac12\bigl(1-\cos\pi u\bigr),\quad u\in[0,1]\ \text{across the taper}$$

Then, where tiles overlap, take the weighted average of their logits:

$$L(x)=\frac{\sum_i w_i(x)\,\ell_i(x)}{\sum_i w_i(x)}$$

and threshold once.

**Worked example.** At a seam, tile A (its own border) outputs a spurious
$\ell = +6$ with weight ≈ 0.001 (the floor). Tile B sees the same pixel in its
interior with $\ell = -4$ and weight 1.0:

```
L = (6 × 0.001 + (−4) × 1.0) / (0.001 + 1.0) = −3.99     → not a crack ✓
```

The union of per-tile masks would have kept the +6. That is the figure in the
paper (Fig. 9).

**Why a raised cosine and not a straight ramp.** Weight and its slope are both
continuous, so there is no visible kink inside the overlap. **Why never exactly
zero.** A pixel covered only by tapered edges must still get a value (the code
floors the weight at 10⁻³).

---

## 11. Connected components and shape

A **connected component** is a group of touching crack pixels — one "blob".
Everything after thresholding works per component.

| Measure | Definition | Reads as |
|---|---|---|
| **Area** | pixel count (`CC_STAT_AREA`) | size |
| **Perimeter** | length of the outline | |
| **Area / perimeter** | ≈ w/2 for a stroke of width w | half-width |
| **Convex hull** | smallest convex shape enclosing it | |
| **Solidity** | area ÷ hull area | how "filled" it is |

```
thin crack            blob
  ╲_╱╲___               ██████
                        ██████      solidity ≈ 0.04     solidity ≈ 0.7–1.0
                        ██████      half-width ≈ 2 px   half-width ≈ 11 px
```

Why `area / perimeter ≈ w/2`: a stroke of width $w$, length $L$ has area $wL$
and perimeter $\approx 2L$, so the ratio is $w/2$.

> **A trap worth knowing.** OpenCV's `contourArea` returns ≈ 0 for a 1-pixel-wide
> line, so area must come from the pixel count. Using the contour area would
> reject precisely the finest cracks.

---

## 12. The distance transform and measuring width (paper Eq. 5)

The **distance transform** assigns each foreground pixel its distance to the
nearest background pixel. In a stroke, that distance is largest down the
centre: it is the **local half-width**.

```
   background 0 0 0 0 0 0 0
   crack row  0 1 1 1 1 1 0     ← 3-px stroke, cross-section
   crack row  0 1 2 2 2 1 0
   crack row  0 1 1 1 1 1 0       distance at centre = 2
   background 0 0 0 0 0 0 0
```

The line of maxima down the middle is the **ridge**. Its values give the width
profile along the whole crack, so it survives branching and tapering — which
`area / perimeter` (a *mean*) does not.

### Width = 2d − 1, not 2d

The transform measures to the *centre of the nearest zero pixel*, which lies a
full pixel beyond the last foreground pixel, while the stroke's true edge is only
half a pixel beyond it. So the ridge value overstates half-width by 0.5 px:

$$w = 2d_r - 1$$

**Worked example.** A 3-px stroke: centre pixel is 2 away from a zero pixel
($d_r = 2$): $w = 2\cdot 2 - 1 = 3$ ✓. A 5-px stroke: $d_r = 3$, $w = 5$ ✓.
Using $2d$ would call a 3-px crack 4 px — **0.29 mm** of error at 0.294 mm/px, on
a measurement whose whole purpose is sub-millimetre.

### Which statistic of the ridge?

| Statistic | Fails on |
|---|---|
| median | branching: junctions add short distances → 1.80 on a true 3 px fork (40 % low) |
| max | one ragged junction pixel → 10.59 on a true 9 px fork |
| **p95** | neither; lands on a taper's wide end — the end that is graded |

---

## 13. Hessians, valleys and steps (paper Eq. 4)

Two things look alike in a mask — a crack and a shadow edge — but differ in
*profile*:

```
brightness across the feature

crack (VALLEY)          shadow / fold / object edge (STEP)

 ▔▔▔▔╲  ╱▔▔▔▔             ▔▔▔▔▔▔╲
      ╲╱                         ╲▁▁▁▁▁
 bright-DARK-bright        bright → dark
```

A crack is a gap light does not return from, so the surface is brighter on
**both** sides. A step is brighter on **one** side only.

### The test that does not work: curvature

The second derivative (the Hessian's eigenvalue) is large and positive at the
bottom of a valley. It is *also* positive on the dark side of a blurred step. So
a high-contrast step scored **29.9** against **26.4** for a shallow crack — the
wrong way round. Curvature measures how sharply brightness *bends*, not *which
sides are brighter*.

### The test that works: a two-sided minimum

$$V(p)=\min\bigl(I(p+d\,\mathbf n),\ I(p-d\,\mathbf n)\bigr)-I(p)$$

with $\mathbf n$ across the feature and $d$ about its half-width.

| Case | Samples either side vs the centre | $\min$ | $V$ |
|---|---|---|---|
| Valley | both brighter than the centre | positive | **positive** |
| Step | one side brighter, **one side darker** | the darker side → below the centre | **negative → clamped to 0** |

The `min` is an AND over the two sides; a step can never satisfy it, however
strong its contrast.

The Hessian is still used — only to find the direction $\mathbf n$ (the
principal direction of curvature), not to score. The angle is $\tfrac12\arctan2$
because orientation repeats every 180°.

**Reproduced results** (Table VIII): 3-px line 14.28, 6-px line 26.49, shallow
crack 6.06, both steps **0.00**.

---

## 14. Principal axis and straightness (paper Fig. 19, Table IX)

A crack is a fracture and wanders; a panel seam is manufactured and is ruled.

Take the component's pixel coordinates. The **principal axis** is the direction
of greatest spread (the first eigenvector of the 2×2 covariance matrix). The
spread *perpendicular* to it — the square root of the smaller eigenvalue —
measures how much the stroke wanders.

A perfectly straight stroke of width $w$ is *uniform* across its width, so its
perpendicular RMS is $w/\sqrt{12}$ (the standard deviation of a uniform
distribution of width $w$). Normalise by it:

$$S = \frac{\text{RMS}_\perp}{w/\sqrt{12}}$$

$S \approx 1$ is a ruled line; $S$ grows with wander.

**Worked check.** Width 3 px → expected spread $3/\sqrt{12} = 0.866$ px. A
sinusoid of amplitude $A$ has RMS $A/\sqrt2$; combined in quadrature with the
width term:

```
A = 2 px:  √(2²/2 + 0.75) = 1.66 px  →  S ≈ 1.92        (measured 1.97)
A = 4 px:  √(4²/2 + 0.75) = 2.96 px  →  S ≈ 3.42        (measured 3.55)
```

Good agreement; the measured values are slightly higher because of
rasterisation. The default rejects $S < 1.12$ — within 12 % of ruled — and only
for components longer than 40 mm (a short crack has no room to bend).

---

## 15. FP16 and TensorRT

Neural nets are normally computed in 32-bit floats. **FP16** uses 16 bits:

- half the memory traffic,
- the Orin's tensor cores run FP16 much faster than FP32.

**TensorRT** is NVIDIA's inference compiler: it fuses layers, picks kernels, and
produces an *engine* tied to one GPU and one TensorRT version (never copy an
engine between machines).

| | FP32 | FP16 |
|---|---|---|
| time per 512² tile | 18.44 ms | **8.77 ms** (2.1×) |

**Parity check.** Faster is worthless if the answer changes. TensorRT output was
compared with ONNX Runtime on real webcam frames: **0.0552 %** of mask pixels
differed (bar 0.1 %). It was also checked to be *non-vacuous*: 38 % of pixels
cleared the threshold, so "0 disagreement" could not be the trivial result of an
all-background output.

---

## 16. Out-of-distribution inputs and why the model cannot abstain

The network ends in one sigmoid. It was trained on CRACK500, where **every
image contains a crack**, so its learned prior is "there is a crack here; find
it". Show it a curtain and it still partitions the image into more- and
less-crack-like regions — it has no "this is not pavement" output.

Evidence: on 20,000 crack-free METU images, **10.85 %** produced at least one
pixel above threshold, with a median false blob of 165 px and mean 734 px. A
person in frame produced 38 % coverage (about 6× a typical *real* crack's 6.3 %).

Remedies, in increasing cost:

1. post-hoc filters (what we built) — mitigation, not a fix;
2. a "surface" gate classifier;
3. fine-tuning with hard negatives.

---

## 17. Defocus

A lens at the wrong focus *low-pass filters* the image: it spreads each point
into a blur disc. A 3-px crack blurred over 15 px has its darkest value pulled
toward the surrounding grey, so after enough blur **the crack is no longer in
the image** — nothing downstream can recover it.

**Detecting it.** The variance of the Laplacian (a second-derivative filter)
measures how much fine detail an image has. A sharp frame scores in the hundreds;
the cardboard frame scored **16.6**.

On the C920, `focus_absolute` runs 0 (far) to 255 (near). The project pinned it
at **30** — close to infinity — which suits a corridor and defocuses anything at
20 cm. This is an open item.
