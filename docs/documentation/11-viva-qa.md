# 11 — Questions you will be asked, and a glossary

Answers are grounded in what was measured. Where the honest answer is "we did
not measure that", it says so — that is the correct answer, and it is better
received than a guess.

Cross-references point to the documents with the evidence.

---

## A. Motivation and framing

**1. What is the contribution? Isn't crack detection solved?**
Detection in a photo is largely solved on benchmarks. The gap is everything
between a mask and a trustworthy *width in millimetres*: a known scale, handling
a model that cannot say "this is not a wall", and covering a scene with several
cameras without creating seam artefacts. The paper's contribution is the measured
solution to those on commodity edge hardware, plus an honest account of what is
still missing. ([01](01-big-picture.md))

**2. Why on an edge device and not the cloud?**
The practical reason is independence from a network: the whole pipeline runs on
the device and only a small status document and JPEGs leave it. Connectivity in a
tunnel was not measured here, so treat that as a design assumption. What *was*
measured is that the device is fast enough: 8.77 ms per tile, so inference is not
the bottleneck at all (capture is).

**3. Why did you rewrite the earlier paper?**
Its central contributions — ArUco wall-frame mapping, persistent crack IDs,
repeat-pass change detection, rover trials — had no code or data behind them.
The new paper reports what was built and moves the rest to Future Work.
([09](09-paper-walkthrough.md))

---

## B. Design decisions

**4. Why not stitch the three images and run the model once?**
Two measured reasons. (1) Resolution: a 5760-px mosaic squeezed into the model's
512 px is a 0.089× scale; a 2-px crack becomes 0.18 px, below one sample.
(2) Seams are long thin straight discontinuities — exactly what a crack detector
fires on and what the shape filter is built to *keep*. ([04 §1](04-camera-and-capture.md))

**5. Why read the cameras one at a time?**
Every Type-A port is behind one USB 2.0 hub (480 Mbit/s). At 720p only 2 of 3
cameras deliver frames, because UVC reserves bandwidth from what the firmware
*declares*, and the usual driver quirk was already enabled. Because the rig is
stationary, sequential reads are indistinguishable from simultaneous ones.
The cost is ≈ 6 s per position, 90 % of it camera start-up. ([04 §2–3](04-camera-and-capture.md))

**6. Why not drop to 640×480 so all three stream?**
It costs 0.88 mm/px instead of 0.29. A 0.3 mm crack is then 0.34 px: not
representable. That would trade away the thing the project measures.

**7. Why is the exposure controller so complicated?**
Because the actuator is a ladder, not a dial. While streaming, exposure snaps to
38/77/156/312/624 — one stop apart — so median luma can only change by factors of
about two and a setpoint between rungs is unreachable. A naive loop oscillates
forever. The controller snaps to real rungs, steps only without overshoot, and
uses gain (continuous) as the fine control. ([04 §4–5](04-camera-and-capture.md))

**8. Why keep gain near zero?**
Gain amplifies read noise into thin bright streaks — the same signature the
model reads as a crack. High gain *manufactures* false positives.

**9. Why tile instead of resizing?**
Resizing 1920×1080 to 512 is a 0.27× scale (a 2-px crack → 0.53 px). Native
tiling keeps every pixel; versus the earlier centre-crop it gives 1.8× the swath
and 2.1× finer detection simultaneously. ([05 §2](05-perception-pipeline.md))

**10. Why clamp the last tile instead of padding?**
Zero padding creates a hard black edge — a perfectly straight linear feature the
detector would fire on. Clamping re-covers a few pixels at no cost.

**11. Why average logits rather than probabilities?**
The logit is what the network produces; it is unbounded and roughly linear in
evidence. A sigmoid compresses differences near 0 and 1 — exactly where a
confident tile should carry the vote. Thresholding is exact either way because
the sigmoid is monotonic. ([02 §10](02-theory-primer.md))

**12. Why a raised cosine window?**
Weight and its first derivative are both continuous, so there is no step inside
the overlap. It is floored at 10⁻³ so no pixel is left without a value.

---

## C. The model and the numbers

**13. What accuracy does it achieve?**
On a leak-free CRACK500 test split at threshold 0.55: precision 0.761, recall
0.821, **Dice 0.790, IoU 0.652**. The threshold was chosen on *validation*
(best Dice 0.7923), not on test. ([03 §5](03-model-and-training.md))

**14. Is 0.65 IoU good?**
The paper does not benchmark against other published IoU values, so it makes no
claim of being state of the art — the model is frozen and the contribution is the
system. It is worth knowing that IoU is harsh on 2–3 px structures: a one-pixel
boundary error is a large fraction of the object, which is why Dice (0.790) reads
higher than IoU (0.652) on the same masks.

**15. How do you know there's no train/test leakage?**
The official splits leak 5 parent photographs between train and test and 1
between validation and test. They were rebuilt by parent photograph (70/15/15)
with zero overlap. ([03 §2](03-model-and-training.md))

**16. What was the optimiser / batch size?**
Not recorded in the repository. The log shows 60 epochs, ≈ 63.5 s/epoch, warm-up
to 3×10⁻⁴ then cosine decay. We state only what is evidenced.

**17. Why do you use the median, p95, not the mean width?**
Tested on shapes of known width: the ridge **median** reads 1.80 on a true 3-px
fork (40 % low); **max** reads 10.59 on a true 9-px fork; area/perimeter is biased
low by branching. **p95** is exact on uniform and branched strokes and lands on a
taper's wide end — the end that is graded. ([05 §8.3](05-perception-pipeline.md))

**18. Why 2d − 1 and not 2d?**
The distance transform measures to the *centre of the nearest zero pixel*, one
full pixel beyond the last foreground pixel, while the true edge is half a pixel
beyond it. So the ridge overstates half-width by 0.5 px. 2d would call a 3-px
crack 4 px — 0.29 mm of error at this scale.

**19. How accurate is the width, really?**
On analytic shapes: exact on uniform and branched strokes, ≤ 0.35 px on
diagonals, **worst case 1.00 px** on a taper's wide end. That is accuracy *in
pixels on synthetic shapes*. In millimetres it additionally depends on the
standoff, which is assumed. ([05 §8.3](05-perception-pipeline.md))

---

## D. The hard questions (answer these straight)

**20. Does it detect real tunnel cracks?**
**We have not measured that.** No genuine crack in a continuous surface has been
through the full pipeline. Recall 0.82 is on CRACK500 pavement. The paper states
this as its first limitation.

**21. Are the millimetre values correct?**
Only given the assumed 0.40 m standoff. Width scales linearly with it: if the
camera is at 0.15 m, every width is reported ≈ 2.7× too large. Calibration
against an object of known width is the first thing to do. ([10 §8.1](10-reproduce-and-extend.md))

**22. How do you know the field of view is 70.42°?**
It is derived from the datasheet's 78° diagonal at 16:9. That 1080p reads the
whole sensor (rather than a cropped window) was to be verified by
`check_fov_parity.py`, but no result is recorded. If 1080p is cropped, every GSD
is too large. It is listed under threats to validity.

**23. Your shape filter and your verification tests remove cracks too, don't they?**
Yes — by design they trade sensitivity for precision, and they can discard a real
crack whose profile resembles a step. The cardboard case is exactly this
trade-off. The filters run *after* the network so it can stay sensitive.
([05 §11](05-perception-pipeline.md))

**24. The model can't tell a wall from a person. Isn't that a fatal flaw?**
It is a real limitation, not fatal: a single-sigmoid network trained only on
cracks has no abstain path. The filters mitigate (38 % → 1.44 % on a person;
18 → 0 components on curtains), but they are post-hoc. The principled fixes —
a surface gate or hard-negative fine-tuning — are future work.

**25. Why are some results on `main` and some on `accuracy-filters`?**
The paper's accuracy results describe the `accuracy-filters` configuration
(blending + valley + straightness). `main` is the more sensitive configuration
kept for a demonstration. ([01 §6](01-big-picture.md))

**26. A road tunnel's wall is 4.5 m away. Can your cameras see 0.3 mm cracks?**
No: 6.6 mm at 4.5 m. A C920 in the middle of a road tunnel cannot see the cracks
that matter. It needs close-range passes (≈ 0.4 m → 0.59 mm), longer focal
length, or a mast. The paper presents this as a result, not a hidden flaw.
([05 §10](05-perception-pipeline.md))

**27. Why is the 11.33× not just noise from three frames?**
It is an effect measured over 3 frames × 15 tiles, indoors, and it is the *ratio*
of detection rates between the 0–2 px band and the interior. The mechanism (no
context beyond a convolution's input edge) and the amplification by the shape
filter (a border line is exactly what it keeps) are both explained and
demonstrated (Fig. 9). A larger, labelled set would firm up the *size* of the
effect; the direction is well supported by the mechanism. Say "measured on
three frames".

---

## E. Short theory checks

**28. What is a logit?** The network's raw per-pixel score; sigmoid(logit) is the
probability. Thresholding the logit at ln(t/(1−t)) is identical to thresholding
the probability at t. ([02 §1](02-theory-primer.md))

**29. Dice vs IoU?** Dice = 2TP/(2TP+FP+FN); IoU = TP/(TP+FP+FN); IoU =
Dice/(2−Dice), so IoU is always lower. ([02 §4](02-theory-primer.md))

**30. Why Dice + focal loss?** ≈ 6.3 % of pixels are crack. Cross-entropy is
dominated by easy background; Dice scores overlap and focal down-weights
confident pixels. ([02 §3](02-theory-primer.md))

**31. What is GSD?** Millimetres of surface per pixel = 2·d·tan(HFOV/2)/W.
0.294 mm/px at 0.40 m. ([02 §5](02-theory-primer.md))

**32. Valley vs step?** A crack is brighter on both sides of the dark line; a
step is brighter on one side. `min(I(p+dn), I(p−dn)) − I(p)` is positive for a
valley and ≤ 0 for a step at any contrast. ([02 §13](02-theory-primer.md))

**33. Why does straightness divide by w/√12?** A perfectly straight uniform
stroke of width w has perpendicular RMS w/√12 (a uniform distribution), so the
ratio is 1 for a ruled line and grows with wander. ([02 §14](02-theory-primer.md))

**34. Why FP16?** Half the memory traffic and faster tensor-core math: 8.77 ms vs
18.44 ms (2.1×), with 0.055 % mask disagreement vs ONNX. ([02 §15](02-theory-primer.md))

---

# Glossary

| Term | Meaning |
|---|---|
| **AE** | Auto-exposure — here a median-luma control loop |
| **AP** | Wi-Fi access point; the Jetson hosts one (`CrackNet`, 10.42.0.1) |
| **Blend** | Weighted average of overlapping tile logits |
| **Clamp** | Shift a tile origin back inside the frame (vs zero-padding) |
| **Clipping** | Pixels saturated at 255 — information lost at the sensor |
| **Component** | A connected group of crack pixels |
| **CRACK500** | Pavement-crack dataset the model was trained on |
| **Deadband** | Luma range (90–150) where the controller does nothing |
| **Defocus** | Optical blur that low-pass filters thin cracks away |
| **Dice / IoU** | Overlap metrics for masks |
| **Distance transform** | Distance from each foreground pixel to the nearest background |
| **Engine** | A TensorRT-compiled network, tied to one GPU + version |
| **FOV / HFOV** | (Horizontal) field of view — 70.42° here |
| **GSD** | Ground sample distance, mm per pixel |
| **Gain** | Electronic amplification; amplifies noise too |
| **Hessian** | Matrix of second derivatives; gives the direction across a line |
| **Isochronous** | USB transfer mode with reserved bandwidth (video) |
| **Logit** | Raw network score before the sigmoid |
| **Luma** | Image brightness, 0–255 |
| **METU** | Dataset of crack-free building images used as negatives |
| **MJPG** | Per-frame JPEG video from the camera |
| **OOD** | Out-of-distribution — unlike the training data |
| **Parity** | Agreement between two implementations (TensorRT vs ONNX) |
| **Prime / confirm** | Long first exposure settle / short per-position check |
| **Raised cosine** | Smooth 0→1 window with continuous slope |
| **Ridge** | Line of maxima down the middle of a stroke's distance transform |
| **Rung** | One of the five exposure values the camera accepts while streaming |
| **Solidity** | Area ÷ convex-hull area |
| **Standoff** | Camera-to-surface distance |
| **Step edge** | Brightness changing from one level to another (shadow, fold) |
| **Straightness** | Wander normalised so a ruled line = 1.0 |
| **STREAMON** | The V4L2 call that starts a stream and reserves bus bandwidth |
| **Swath** | Width of surface one camera covers |
| **TensorRT** | NVIDIA's inference compiler/runtime |
| **UVC** | USB Video Class — the webcam protocol |
| **Valley depth** | How much darker a pixel is than both sides |
