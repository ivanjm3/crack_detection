# 05 — The perception pipeline

*Paper: §III-G tiling and reassembly, §III-H verification, §III-I measurement;
§IV-E, F, G, H, I; Figs. 8, 9, 10, 17, 18, 19, 20, 21; Tables VIII–XII;
Eqs. 2–5.*

Code: `tiler.py`, `trt_infer.py`, `live.py:clean_mask`, `verify.py` (branch),
`measure.py`, `scanner.py`.

![Fig. 10 — pipeline with measured cost](figures/fig_pipeline.png)

*Paper Fig. 10. The orange stages work on one connected component at a time; the
others on whole frames or tiles.*

---

## 1. The order of operations

`scanner.Scanner.infer_frame()` is the whole perception chain for one camera:

```python
crops   = extract(frame, tiles)                       # 1. tile
logits  = [net.infer_logits(preprocess(c)) for c in crops]   # 2. infer
blended, _ = blend_logits(logits, tiles, W, H)        # 3. average overlapping logits
raw     = blended > thr                               # 4. threshold ONCE (thr = logit(0.55))
clean   = clean_mask(raw, min_area, max_halfwidth…)   # 5. shape filter
clean, dropped = verify.apply(clean, gray, …)         # 6. valley + straightness
# measure.summarise(clean, gsd)                       # 7. millimetres
```

**One rule governs the order: shape metrics are computed once, on the whole
reassembled mask, never per tile.** A crack crossing a tile boundary is truncated
in both tiles, and a truncated fragment is short, stubby and solid — the shape
filter would throw it out as a blob.

---

## 2. Tiling — Fig. 8

![Fig. 8](figures/fig_tiles.png)

### The algorithm (`tiler.plan_tiles`)

```python
stride = round(tile * (1 - overlap))        # 512 × 0.85 = 435
origins(extent): range(0, extent - tile + 1, stride)   # then clamp the last one
```

### Worked example — 1920×1080

```
x:  range(0, 1408+1, 435) = 0, 435, 870, 1305       last allowed = 1920−512 = 1408
    1305 ≠ 1408  →  append 1408         ⇒  x = 0, 435, 870, 1305, 1408   (5 columns)
y:  range(0,  568+1, 435) = 0, 435      last allowed = 1080−512 =  568
    435 ≠ 568   →  append 568           ⇒  y = 0, 435, 568               (3 rows)

5 × 3 = 15 tiles
```

The same rule gives 6 tiles at 1280×720 (x = 0, 435, 768; y = 0, 208), and
raises an error for 640×480 (smaller than a tile) — which is why live mode
centre-crops and resizes instead.

### What Fig. 8 shows — and a subtlety

The figure is drawn **by the deployed `plan_tiles`** (CODE). The numbered
rectangles are tiles; shading darkens where they overlap. Notice that the last
column (tile 5 at x = 1408) and last row (y = 568) overlap their neighbours by
**more** than the nominal 15 %:

```
columns 4 and 5:  x = 1305 and 1408  →  only 103 px apart, overlap = 512 − 103 = 409 px (80 %)
rows 2 and 3:     y =  435 and  568  →  only 133 px apart, overlap = 379 px (74 %)
```

That is the cost of **clamping instead of padding**:

| Option | Effect |
|---|---|
| **Zero-pad** the last tile | introduces a hard black edge — a perfectly straight linear feature, which is what the detector fires on |
| **Clamp** the origin back inside | re-covers pixels a neighbour already saw; costs a little inference, creates nothing |

Every pixel is covered by at least one tile (`coverage_map` returns 0 nowhere).

### Why this beats the Phase 1 pipeline

| | Phase 1 (centre-crop 720×720 → 512) | Phase 2 (native tiling) |
|---|---|---|
| Frame used | the middle square; **left & right thirds discarded** | whole frame |
| Scale | 0.711 (downscaled) | 1.0 (native) |
| A 2-px crack becomes | 1.42 px | 2.0 px |
| Swath at 0.4 m | 0.32 m | 0.56 m |
| Result | | **1.8× swath, 2.1× finer, simultaneously** |

(MEASURED/DERIVED from `docs/multicam-plan.md` §4A.3.)

---

## 3. Preprocessing and inference

`trt_infer.CrackNetTRT.preprocess`:

```python
x = (rgb.astype(float32) / 255.0 - MEAN) / STD        # ImageNet statistics
return ascontiguousarray(x.transpose(2, 0, 1)[None])  # HWC → NCHW
```

| Step | Why |
|---|---|
| BGR → RGB (`cvtColor`) | OpenCV loads BGR; the model was trained on RGB. Getting it wrong does not crash — it quietly degrades |
| ÷255, subtract mean, ÷std | The encoder was pre-trained with ImageNet statistics (0.485/0.456/0.406; 0.229/0.224/0.225) |
| `transpose(2,0,1)` | Height-width-channel → channel-first |
| `ascontiguousarray` | `transpose` only changes *strides*; TensorRT reads a raw buffer, so without this it reads the wrong bytes |

Inference (`infer_logits`) copies to the GPU asynchronously, runs
`execute_async_v3`, copies back, synchronises, and returns the raw logit map —
**no sigmoid** ([02 §1](02-theory-primer.md)).

**Costs (MEASURED, 1080p, 15 tiles, per camera):** preprocessing 89.0 ms,
inference 130.3 ms, tile+stitch 1.7 ms → **221.1 ms**. Preprocessing is a CPU
NumPy path and is the largest cost after the engine.

---

## 4. The tile-border artefact — measured

### The hypothesis

A convolution has no pixels beyond its input's edge, so outer rows are predicted
from padding that does not exist in the scene. If so, detections should *pile up*
near tile borders.

### The experiment (`analyze_tile_edges.py`, MEASURED)

1. Capture frames; tile; infer; threshold each tile; **union** the masks
   (`stitch_masks`).
2. For every pixel compute its **distance to the nearest tile border** — taking
   the *maximum* over all tiles that cover it (a pixel in an overlap is at the
   edge of one tile but well inside its neighbour; scoring it by its worst tile
   would blame the overlap for an artefact the union may already have rejected).
3. Bin pixels by that distance: `0–2, 2–4, 4–8, 8–16, 16–32, 32–64, 64–128,
   128–256`.
4. In each bin measure the **detection rate** (fraction of pixels marked).
5. Report the ratio of the **0–2 px band** to the **interior (≥ 128 px)**.

Repeated on the raw mask and on the mask after `clean_mask`. Three frames × 15
tiles.

### Result — Fig. 17(a)

| Mask | Border band ÷ interior |
|---|---|
| Raw | **2.18×** |
| After `clean_mask` | **11.33×** |

**Why cleaning makes it *worse*.** `clean_mask` removes small and blobby
components — which are mostly *interior* noise. A border artefact is a long thin
line along a seam, i.e. exactly what the filter keeps. So the interior rate
falls, the border rate does not, and the ratio rises.

Every one of those border detections is a false positive *by construction*.

---

## 5. Union versus blending — Fig. 9

![Fig. 9](figures/fig_blend.png)

### The two reassembly rules

| | `stitch_masks` (union, on `main`) | `blend_logits` (on `accuracy-filters`) |
|---|---|---|
| Per tile | threshold, producing a binary mask | keep the **logit** map |
| On overlaps | **OR** — marked if *any* tile fires | **weighted average** of logits |
| Threshold | per tile | **once**, on the average |
| Effect on a border artefact | **kept** — the union keeps the strongest artefact from every covering tile | **suppressed** — a tile that sees context outvotes one guessing from padding |
| Cost | favours recall | favours precision |

### The window (`tile_window`)

```python
ramp = round(tile * taper)               # 512 × 0.25 = 128 px
edge = 0.5 * (1 - cos(pi * t))           # raised cosine 0→1 over the ramp
w[:ramp] = edge ; w[-ramp:] = edge[::-1] # flat 1.0 between
w = max(w, 1e-3)                         # never exactly zero
return outer(w, w)                       # separable: rows × columns
```

Top panel of Fig. 9 is that profile: rising over 128 px, flat for 256, falling
over 128.

### What the bottom panel of Fig. 9 shows

A **synthetic** demonstration, produced by the *deployed* `blend_logits`:

- Two tiles on a 768×512 frame (origins x = 0 and x = 256).
- Each tile is given logit −4 everywhere (clean background) **plus a +6 spike on
  its own border** — the artefact (tile 1 at x = 509–511, tile 2 at x = 256–258).
- **Red**: pixels the union would mark — **3,072 pixels**.
- **Blue**: the blended logit — **never above −3.9** — so **0 pixels** survive.

Arithmetic at one pixel (tile 1's own right border, tile 2's interior):

```
L = (6 × 0.001 + (−4) × 1.0) / (0.001 + 1.0) = −3.99
```

This is a *demonstration of the mechanism*, not a measurement on real frames.
The real-frame effect is the coverage drop below.

### Measured effect on real frames (MEASURED)

| Frame | Raw coverage, union | Raw coverage, blended |
|---|---|---|
| 1 | 75.2 % | 63.7 % |
| 2 | 39.6 % | 24.0 % |
| 3 | 18.0 % | 8.5 % |

Large reductions; but these frames were out-of-domain (curtains etc.) and **no
recall was measured**, so "blending improves accuracy" is not claimed — only
that it removes a measured artefact.

### Tests that pin the behaviour (`test_tiler.py`, REPRODUCED 23/23)

Coverage and bounds; byte-identical sub-arrays; round trip; union-on-overlap;
**a crack across seams stays one component**; window shape (full weight at
centre, ≈ 0.002 at the edge, monotonic); **a tile-edge spike does not survive the
blend while the union keeps it**; a detection every tile agrees on *is* kept
(blended logit 6.00); bad input is rejected.

---

## 6. The shape filter — `clean_mask`

Runs on the reassembled mask. For each connected component:

| Test | Rule | Default | Rejects |
|---|---|---|---|
| Area floor | `area < min_area` | **600 px** | specks |
| Area ceiling | `area > max_area_frac × frame` | **20 %** | one giant region |
| Thickness | `area / perimeter > max_halfwidth` | **15 mm / 2 / GSD ≈ 25.5 px** | fat strokes |
| Solidity | `area / hull_area > max_solidity` *and* `area > 2 %` of frame | **0.80** | large compact blobs |

### Two thresholds that had to be rescaled

Both limits are in **pixels**, and native tiling changed what a pixel *is*.
Carrying the old numbers across would have silently changed behaviour:

| Setting | Phase 1 | Phase 2 | What would have happened |
|---|---|---|---|
| `min_area` | 300 px (on pixels 1/0.711 larger) | **600 px** | The same speck covers 1.98× more native pixels; keeping 300 would reject roughly half the physical area it used to |
| `max_halfwidth` | 12 px ≈ "wider than ~15 mm" at 0.62 mm/px | **`--max-width-mm 15`** → 25.5 px at 0.294 | 12 px at 0.294 mm/px is only 7 mm: the widest reportable crack would have been silently halved |

Expressing the limit in **millimetres** is possible only because a pixel now has
a known size.

### Evidence for the thresholds (MEASURED, 51 components)

| | thin synthetic crack | false blobs |
|---|---|---|
| half-width | 1.94 px | median 11.2, max 43 |
| solidity | 0.041 | median 0.715, max 1.09 |

Result on a person in frame: coverage **38 % → 1.44 %** — **96.2 %** of the false
coverage removed. On out-of-domain surfaces: raw 8–32 % → 0.2–1.5 %.

### What it cannot do

It cannot remove anything **long and thin**, because that is the definition of
what it keeps. Every remaining false positive was long and thin: panel seams,
vent louvres, the wall/ceiling join, a folded cardboard corner.

> On an out-of-domain surface the shape filter is not a refinement of the
> detector. **It is the detector.**

---

## 7. Verification: valley and straightness

![Fig. 18](figures/fig_valley.png)

Two further tests, each targeting a property a crack has and a man-made edge
does not. They run **per component** and only on the branch `accuracy-filters`.

### 7.1 Valley depth (Eq. 4, Fig. 18, Table VIII)

`verify.valleyness(gray, scales=(2.0, 3.5, 6.0))` for each scale `d`:

| Step | Operation |
|---|---|
| 1 | Smooth with a Gaussian of σ = max(d/2, 0.8) |
| 2 | Hessian components `gxx, gyy, gxy` (Sobel, 2nd order) |
| 3 | Direction across the feature: `θ = ½·atan2(2·gxy, gxx − gyy)`; `n = (cosθ, sinθ)` |
| 4 | Sample the smoothed image at `p + d·n` and `p − d·n` (`cv2.remap`) |
| 5 | `depth = min(a, b) − I(p)` |
| 6 | Take the maximum over the three scales; clamp at 0 |

The per-component score is the **median** of that map over the component's pixels
(not the mean: a component clipping the end of a real crack would otherwise be
carried by a few strong pixels). `min_valleyness` = **2.0** grey levels in the
scanner.

**How Fig. 18 was made (CODE).** For each of five synthetic 400×400 images
(`test_verify.valley_image`, `step_image`): a flat surface of 150 grey with either
a dark line (3 px depth 60; 6 px depth 60; 3 px depth 25) or a step (depth 60;
depth 110), Gaussian blur σ = 1, plus noise σ = 2. The mask a model would produce
along the feature is `strip_mask`. The top row shows the image crop; the bottom
row the intensity profile across row 200; the number under each is the
deployed `verify.report(...)["valleyness"]`.

| Image | Depth (grey levels) | Verdict |
|---|---|---|
| 3 px dark line | **14.28** | crack-like |
| 6 px dark line | **26.49** | crack-like |
| shallow crack (depth 25) | **6.06** | crack-like, still > 2.0 |
| step, same contrast | **0.00** | rejected |
| step, high contrast | **0.00** | rejected |

(REPRODUCED.) The profile panels explain *why*: the dark lines dip and recover to
the *same* level on both sides; the steps do not come back.

### 7.2 Straightness (Fig. 19, Table IX)

![Fig. 19](figures/fig_straight.png)

`verify._straightness`:

1. Take the component's pixel coordinates; subtract the mean.
2. Covariance matrix; its **smaller eigenvalue** is the squared perpendicular
   spread — no projection loop needed.
3. `rms_perp = √(minor eigenvalue)`.
4. Divide by `width_px / √12` (the spread of a perfectly straight stroke).

`width_px = 2·area/perimeter`. Rejected when `S < 1.12` **and** longer than 40 mm
(`min_len_px = 40 / GSD ≈ 136 px`; the function's own default of 120 px applies
only if called directly).

**How Fig. 19 was made (CODE).** A 3-px vertical stroke is distorted by a sine of
amplitude 0, 2, 4, 8, 16 px (three periods; `test_verify.wavy_mask`), and
`verify.report` returns its score:

| Wander (px) | 0 | ±2 | ±4 | ±8 | ±16 |
|---|---|---|---|---|---|
| Straightness | 0.946 | 1.965 | 3.546 | 7.105 | 15.456 |
| Decision | **dropped** | kept | kept | kept | kept |

Monotonic, as required. The y-axis is logarithmic, which is why the dashed
threshold sits so close to the first point.

### 7.3 The two tests are not redundant (Fig. 17c)

![Fig. 17](figures/fig_fp.png)

On the curtain-and-cable scene (raw coverage 75 %):

| | Verdict | Components | Worst camera |
|---|---|---|---|
| Shape filter only | **CRACK** | 18 | 7.4 %, "23.34 mm widest" |
| + valley + straightness | **clear** | **0** | 0.000 % |

**All 18 fell to the valley test.** Their straightness scored 1.45–2.29 — all
*above* 1.12 — because curtain folds wander like cracks, so straightness alone
would have passed every one. Conversely a ruled panel seam is a perfect valley
(dark on both sides) and would pass the valley test; straightness catches it.

---

## 8. Measurement in millimetres

### 8.1 Ground sample distance

`measure.gsd_mm_px(standoff_m, width_px, hfov_deg=70.42)` implements Eq. 1:
`2·d·tan(hfov/2)/W × 1000`. At 0.40 m, 1920 px → **0.2940 mm/px**.

### 8.2 Width from the distance-transform ridge

`measure._width_from_distance`:

```python
sub  = copyMakeBorder(sub, 1,1,1,1, 0)          # give every stroke an outside
dist = distanceTransform(sub, DIST_L2, 5)
peak = dilate(dist, 3×3)
ridge = dist[(dist > 0) & (dist >= peak − 0.5)] # pixels within 0.5 of the local max
width = 2·percentile(ridge, 95) − 1
```

`measure.components` returns, per component and widest first:
`width_mm` (**p95**), `width_median_mm`, `width_max_mm`, `width_mean_mm`
(area/perimeter), `length_mm = area / mean_width × GSD`, `area_px`, bbox.

### 8.3 How it was validated — Fig. 20, Table X

![Fig. 20](figures/fig_width.png)

Because nobody can sanity-check a millimetre figure by eye, the estimator was
tested against masks whose true width is known **by construction**
(`test_measure.py`, REPRODUCED).

**The generator.** `_stroke` builds every pixel within w/2 of a line segment from
a *distance field*, with width interpolated for tapers. (An earlier version used
`cv2.line`, whose `thickness=3` actually draws **5** rows — the test was checking
against a width never on the image.)

**The shapes.** straight (3, 5, 9, 15), diagonal (3, 5, 9), branched (3, 5, 9),
tapered (2→10, 3→15, 1→6). Run at GSD = 1 mm/px so millimetres *are* pixels.

| Shape | True | p95 (reported) | median | max | area/perim |
|---|---|---|---|---|---|
| straight 3 | 3.00 | **3.00** | 3.00 | 3.00 | 2.99 |
| straight 5 | 5.00 | **5.00** | 5.00 | 5.00 | 4.96 |
| straight 9 | 9.00 | **9.00** | 9.00 | 9.00 | 8.83 |
| straight 15 | 15.00 | **15.00** | 15.00 | 15.00 | 14.47 |
| diagonal 3 | 3.54 | 3.39 | 3.39 | 3.39 | 3.52 |
| diagonal 5 | 4.95 | 4.60 | 4.60 | 4.60 | 4.92 |
| diagonal 9 | 9.19 | 8.99 | 8.99 | 8.99 | 9.04 |
| branched 3 | 3.00 | **3.00** | **1.80** | 3.00 | 2.84 |
| branched 5 | 5.00 | **5.00** | 5.00 | **6.19** | 4.82 |
| branched 9 | 9.00 | **9.00** | **7.79** | **10.59** | 8.60 |
| tapered 2→10 | 10.00 | 9.00 | 5.00 | 9.39 | 5.94 |
| tapered 3→15 | 15.00 | **15.00** | 9.00 | 15.00 | 8.85 |
| tapered 1→6 | 6.00 | 5.00 | 3.00 | 5.39 | 3.39 |

**Reading it honestly.**

- Uniform strokes: exact.
- Branched: p95 exact; the **median** is 40 % low on the 3-px fork (1.80) and
  **max** is 18 % high on the 9-px fork (10.59). That is why p95 is the headline.
- Diagonals read ≈ 0.1–0.35 px low — rasterisation of a diagonal.
- **Worst error: 1.00 px**, on a taper's wide end (2→10 reads 9.00; 1→6 reads
  5.00). Every non-taper case is within **0.35 px**. The test's bars are 0.75 px
  for uniform/branched strokes and 1.5 px for a taper's wide end, so the worst
  case passes — and the paper reports the 1.00 px worst case rather than the
  tighter bar.

**Three bugs this testing found**

| Bug | Symptom | Fix |
|---|---|---|
| Distance transform on a tight bbox | a 2-px line 300 px long read **302 px** wide (no background inside the crop) | 1-px zero border |
| Used `2d` | every 3-px crack read 4 px (0.29 mm error) | `2d − 1` |
| Test used `cv2.line` | measured against a width never present | analytic shapes |

---

## 9. One crack, end to end (a worked example)

Suppose a 1080p frame at 0.40 m (GSD 0.294 mm/px) contains a crack 3 px wide and
about 340 px long, gently meandering (±3 px).

| Stage | What happens | Number |
|---|---|---|
| Tile | Spans two tiles near a seam | 15 tiles planned |
| Infer | Each tile's logits along the crack | ≈ +3…+6 |
| Blend | Seam logits averaged; crack continues across | one component |
| Threshold | logit > 0.2007 | mask |
| Shape filter | area ≈ 1000 px > 600 ✓; half-width ≈ 1.5 px < 25.5 ✓; not compact ✓ | kept |
| Valley | median depth e.g. 14 > 2.0 ✓ | kept |
| Straightness | ±3 px wander over 340 px → S ≈ 2.7 > 1.12 ✓; length 100 mm > 40 mm so test applies | kept |
| Measure | ridge p95 = 2 → `w = 2·2 − 1 = 3 px` | **0.88 mm** wide |
| | length ≈ area / mean_width × GSD ≈ 340 × 0.294 | **≈ 100 mm** long |

A ruled 3-px **seam** would differ at one place: straightness ≈ 0.95 < 1.12 and
length > 40 mm → **dropped**.

---

## 10. Resolution against standoff — Fig. 21, Table XI

![Fig. 21](figures/fig_resolution.png)

### How the plot is made (DERIVED)

`gsd = 2·d·tan(35.21°) / W × 1000` for `W = 1920` (1080p) and `W = 1280` (720p),
times 2 for "a 2-px crack", for `d` from 0.2 to 6 m. Straight lines, because GSD
is linear in distance. The red dashed line is the **0.3 mm working target** — a
project figure (the example width in the paper's Introduction), *not* a cited
standard.

### Table XI (every cell recomputed)

| Standoff | Swath/camera | GSD (mm/px) | Smallest crack (2 px) |
|---|---|---|---|
| 0.25 m | 0.35 m | 0.18 | 0.37 mm |
| **0.40 m** | 0.56 m | 0.29 | **0.59 mm** |
| 0.60 m | 0.85 m | 0.44 | 0.88 mm |
| 1.00 m | 1.41 m | 0.74 | 1.47 mm |
| **4.50 m** | 6.35 m | 3.31 | **6.62 mm** |
| 5.50 m | 7.76 m | 4.04 | 8.09 mm |

### What it means for tunnels

A road tunnel is roughly 9–10 m wide and 6–7 m high; a carriage in the middle is
**4–6 m** from the lining. At 4.5 m a C920 resolves ≈ **6.6 mm** — about **22×**
coarser than the 0.3 mm target.

> **A C920 in the middle of a road tunnel cannot see the cracks that matter.**

Ways out (and what each costs):

| Option | Effect | Cost |
|---|---|---|
| Drive close to a wall (0.4–0.75 m) | 0.29–0.55 mm/px — meets spec | only ~0.5–1 m of lining per pass |
| Telephoto lenses | restores mm/px at standoff | different hardware; narrow swath |
| Mast/arm carrying cameras | meets spec, reaches the crown | mechanical complexity |
| Accept coarse detection | finds ≥ 7 mm structural cracks | fails the durability use case |

This is *the* external-validity limit of the paper, and the reason the choice of
"smallest crack that must be detected" is an open question.

---

## 11. The cardboard case, in full

A cardboard sample with a visible crack was presented at 20 cm.

| Observation | Value |
|---|---|
| `main` (before blending + verification) | flagged CRACK |
| `accuracy-filters` | clear |
| Raw model coverage | **0.27 %** |
| Where the components were | **all four on the bottom frame edge** |
| What the feature was | a **torn-edge silhouette** — a step, not a valley |
| Sharpness | variance of Laplacian **16.6** (sharp ≈ hundreds) |
| Focus setting | `focus_absolute = 30` (≈ infinity) at 20 cm |

**Interpretation.** The verification filters did their job: they rejected a
step. And the underlying image was too defocused for any detector to see a
hairline. The case is in the paper *because it shows both the value and the cost*
of the tests: they discard a real crack whose profile resembles a step.

---

## 12. Every tunable in one table

`scanner.add_arguments` (defaults):

| Flag | Default | Meaning |
|---|---|---|
| `--thresh` | 0.55 | probability threshold (applied as logit 0.2007) |
| `--overlap` | 0.15 | tile overlap fraction |
| `--taper` | 0.25 | fraction of each tile edge that fades |
| `--average` | 4 | frames averaged per camera |
| `--min-area` | 600 | px, after shape filter |
| `--max-area-frac` | 0.20 | reject one component larger than this fraction |
| `--max-width-mm` | 15.0 | too thick to be a crack |
| `--max-solidity` | 0.80 | large and compact → blob |
| `--max-straightness` | 1.12 | ruled-line rejection |
| `--min-straight-len-mm` | 40.0 | straightness only above this length |
| `--min-valleyness` | 2.0 | grey levels darker than both sides |
| `--alert-frac` | 0.01 | coverage that raises CRACK |
| `--standoff` | 0.40 | m — **assumed**, sets mm/px |
| `--hfov` | 70.42 | degrees, horizontal |
| `--mm-per-px` | 0 (off) | **measured** GSD, overrides standoff |
| `--no-verify`, `--no-shape-filter` | off | disable stages |
