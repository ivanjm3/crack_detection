# Perception — what happens to a frame after capture

Chronological. Everything here is OpenCV and geometry; the model itself is
unchanged from Phase 1.

The pipeline:

```
frame (1920×1080 BGR)
  → tile into 512×512 at native scale        tiler.py
  → preprocess + TensorRT inference          trt_infer.py
  → reassemble tile outputs                  tiler.py
  → threshold                                scanner.py
  → shape filtering                          live.py  clean_mask()
  → verification filters                     verify.py   [branch only]
  → measurement in millimetres               measure.py
  → overlay + composite                      scanner.py
```

---

## 1. Native-resolution tiling

### The problem it replaces

Phase 1 centre-cropped 720×720 out of a 1280×720 frame and downscaled it to 512.
That cost twice:

- the left and right thirds of every frame were **thrown away**
- what survived was scaled by **0.711**

### The fix

A stationary rig has inference to spare, so tile the whole frame into 512×512
crops **at native scale**. Each tile is square (no anisotropic distortion),
nothing is resized (no detail lost), nothing is cropped (full sensor width used).

At 1920×1080 with 15% overlap: **15 tiles per camera**.

Measured against the Phase 1 pipeline at 0.4 m standoff: **1.8× the swath and
2.1× finer crack detection, simultaneously.**

### Two choices worth knowing

**Edge tiles are clamped, not padded.** A tile running off the frame could be
zero-padded to 512, but that introduces a hard black edge — a strong, perfectly
straight linear feature, which is exactly what a crack detector finds. Clamping
the origin back inside the frame re-covers a few pixels a neighbour already saw,
which costs nothing.

**Shape metrics are computed once on the reassembled mask, never per tile.** A
crack crossing a tile boundary is truncated in both tiles, and a truncated
fragment is short, stubby and solid — it would be thrown out by the very filter
meant to protect against blobs.

---

## 2. Preprocessing and inference

```python
x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD      # ImageNet normalisation
return np.ascontiguousarray(x.transpose(2, 0, 1)[None])  # HWC → NCHW
```

- **BGR → RGB** — OpenCV loads BGR, the model was trained on RGB. Getting this
  wrong does not crash; it quietly degrades.
- **ImageNet mean/std** — the encoder was pretrained with them.
- **`ascontiguousarray`** — `transpose` only changes strides, not memory order.
  TensorRT reads a raw buffer, so without this it reads the wrong bytes.

### Thresholding in logit space

The sigmoid is never computed. Thresholding a probability at *t* is identical to
thresholding the logit at `log(t/(1−t))`, and only the mask is used downstream —
never the probability values.

The sigmoid costs ~2.5 ms per call. Negligible for one frame; **22% of inference
once a scan runs 15 tiles per camera.**

---

## 3. Reassembly — union vs blending

### `main`: union (`stitch_masks`)

Each tile is thresholded, then overlapping masks are combined with **OR**. A
crack crossing a boundary is truncated in both tiles, so the union keeps
whichever tile saw more of it. This favours recall.

### What that costs — measured

A convolution has no information past the edge of its input, so a tile's
outermost rows are predicted from padding that does not exist in the scene.

Detection rate within 2 px of a tile border vs the deep interior, 3 frames ×
15 tiles (`analyze_tile_edges.py`):

| | ratio |
|---|---|
| raw mask | **2.18×** |
| after `clean_mask` | **11.33×** |

Every one of those is a false positive by construction. The shape filter makes
it **worse**, because a border artefact is a long thin line along the seam —
the geometry the filter is built to keep.

The union keeps the strongest artefact from every tile covering a pixel, which
is the opposite of suppressing it.

### `accuracy-filters`: weighted blending (`blend_logits`)

Each tile is weighted by a **raised cosine** — flat in the middle, fading at the
edges — and overlapping logits are averaged, normalised by accumulated weight,
then thresholded **once**.

A tile that can see the context outvotes one guessing from padding.

- **Raised cosine, not a linear ramp** — weight and its first derivative are both
  continuous, so no visible step partway into the overlap.
- **Averaged in logit space, not probability** — the logit is what the network
  produces, is unbounded and roughly linear in evidence, and the sigmoid is
  monotonic so the threshold maps across exactly. Averaging probabilities would
  compress differences near 0 and 1, exactly where confident tiles should
  dominate.
- **Window never reaches zero** — a pixel covered only by tapered regions still
  needs a value rather than a division by zero.

Measured: raw coverage **75.2 → 63.7%, 39.6 → 24.0%, 18.0 → 8.5%**.

---

## 4. Shape filtering — `clean_mask()`

Area alone cannot tell "one huge crack" from "one huge blob", which is why a
person or a curtain can register 19% coverage — three times denser than the
~6.3% of pixels that are crack in an average CRACK500 mask.

Cracks are thin branching filaments, so two cheap geometric tests separate them
from the fat compact regions out-of-domain objects produce:

| Test | Meaning |
|---|---|
| `area / perimeter` | A stroke of width *w* has area/perimeter ≈ *w*/2 — a direct read of thickness in pixels |
| `area / convex-hull area` (solidity) | A meandering filament fills very little of its hull; a blob fills most of it. Only applied to large components, since a short stubby but genuine crack can be perfectly solid |

**Area always comes from `CC_STAT_AREA`, never `cv2.contourArea`**, which
collapses to ~0 for a 1-px-wide line and would reject exactly the cracks that
matter most.

### Thresholds had to be rescaled for native tiling

Both filters are in **pixels**, and native tiling changed what a pixel is. This
is the kind of bug that leaves a setting looking untouched:

| Setting | Was | Now | Why |
|---|---|---|---|
| `min_area` | 300 | **600** | Phase 1 counted on model pixels 1/0.711 larger, so the same speck now covers 1.98× more pixels. Carried across unchanged it would reject ~half the physical area it used to |
| `max_halfwidth` | 12 px | **`--max-width-mm 15`** | 12 px meant "wider than ~15 mm" at the Phase 1 pixel scale; the same 12 at 0.294 mm native pixels means 7 mm — quietly halving the widest crack the system can report |

Expressing the thickness limit in millimetres is only possible because a pixel
now has a known size on the ground.

### What it cannot remove

Anything long and thin — because long and thin is the definition it keeps. Every
remaining false positive on this rig is long and thin: panel seams, vent
louvres, wall/ceiling joins, folded cardboard corners.

Also worth recording: on an out-of-domain surface, **raw model coverage is
8–32%, cut to 0.2–1.5% by cleaning**. The shape filter is not a refinement
there — it *is* the detector.

---

## 5. Verification filters (`accuracy-filters` branch only)

Two tests, each targeting a property a crack has and a man-made edge does not.

### Straightness — is it suspiciously perfect?

A crack is a fracture; it follows whatever path through the material is weakest,
which is never a straight line at the scale of its own width. A panel seam, door
frame or box fold is **manufactured**, and is straight to within a pixel over
hundreds of pixels.

Measured as RMS perpendicular distance of the component's pixels from its own
principal axis, in units of what a perfectly straight stroke of the same width
would give (*w*/√12):

| Shape | Score |
|---|---|
| ruled straight line | 0.95 |
| wanders ±2 px | 1.97 |
| wanders ±4 px | 3.55 |
| wanders ±16 px | 15.46 |

Default rejects below **1.12**, and only for components longer than 40 mm — a
short crack is straight for the same reason a short arc of any curve is: there
was no room to bend.

### Valleyness — is it a dark line, or just a boundary?

A crack is a **valley**: darker than the surface on *both* sides, because it is
a gap light does not return from. A shadow, fold or object edge is a **step**:
darker on one side, lighter on the other.

**The first implementation failed its own test.** It scored this with the
Hessian's largest eigenvalue. Across a blurred step the second derivative is
positive on the dark side, so a mask lying on the boundary picks up that lobe:

| | score |
|---|---|
| high-contrast step | 29.9 |
| shallow crack | 26.4 |

Ranked backwards. **Curvature measures how sharply intensity bends, not which
sides are brighter.**

What works:

```
depth = min( I(p + d·n),  I(p − d·n) ) − I(p)
```

with **n** across the feature (from the Hessian's principal direction) and *d*
about its half-width.

The `min` is an **AND over the two sides**, and a step can never satisfy it —
one sample always lands on the darker side, so the value goes negative however
strong the contrast.

| Image | Valley depth (grey levels) |
|---|---|
| 3 px dark line | 14.3 |
| 6 px dark line | 26.5 |
| shallow crack (depth 25) | 6.1 |
| step, same contrast | **0.00** |
| step, high contrast | **0.00** |

Default threshold is 2.0 grey levels. Evaluated over several scales because the
response peaks when the scale matches the feature width.

**Computed per component, not over the whole frame.** Components are sparse —
seventeen on a 1920×1080 frame — and the full-frame map cost 1.1 s of a 9.25 s
cycle to read a few thousand pixels. Locally it is free.

### Result on a real hard scene

Curtains and a cable tangle, where raw model coverage is 75%:

| | verdict | components | worst camera |
|---|---|---|---|
| without | **CRACK** | 18 | 7.4% coverage, "23.34 mm widest" |
| with | **clear** | 0 | 0.000% |

All 18 fell to the **valley** test. Straightness scored 1.45–2.29 across all of
them — curtain folds *wander* like cracks, so straightness alone would have
passed every one. The two filters are not redundant.

---

## 6. Measurement in millimetres

Coverage is an alarm, not a measurement: it conflates one wide crack with a
dozen hairlines, and it changes if the camera moves closer without the surface
changing at all. Standards grade cracks by **width**.

### Ground sample distance

```
gsd = 2 · standoff · tan(hfov/2) / width_px
```

**hfov is 70.42°, the C920's horizontal field at 16:9 — not the 78° diagonal
from the datasheet.** Using the diagonal would overstate the swath by ~15% and
every width would inherit that error.

At 0.40 m, 1920 px wide: **0.294 mm/px**, swath 1.69 m across three cameras,
smallest resolvable crack ~0.59 mm.

> **This is the weakest link.** Width scales *linearly* with standoff. An
> assumed 0.40 m when the camera is at 0.15 m reports everything 2.7× too wide.
> `--mm-per-px` overrides it with a measured figure.

### Width from the distance-transform ridge

The distance transform gives, for every interior pixel, the distance to the
nearest background pixel — the **local half-width** at that point. The ridge
running down the middle of a stroke carries its width profile.

Three bugs were found by testing against shapes of known width:

1. **The transform ran on the component's tight bounding box**, so a stroke
   filling its own crop had no background pixel inside to measure from. A 2 px
   line 300 px long measured as **302 px wide**. Fixed with a 1 px zero border.

2. **Width is 2d − 1, not 2d.** Distance is measured to the nearest zero
   *pixel*, whose centre sits a full pixel beyond the last foreground pixel,
   while the stroke's edge is only half a pixel beyond. 2d reported every 3 px
   crack as 4 px — 0.29 mm of error on a sub-millimetre measurement.

3. **The test itself was wrong.** It built shapes with `cv2.line` and assumed
   thickness 3 drew 3 rows. It draws **5**. Shapes are now analytic distance
   fields, so width is exact by construction.

After the fixes, 3/5/9/15 px strokes return exactly 3.00/5.00/9.00/15.00.

### Which statistic is the headline — chosen on evidence

| Shape | ridge median | **p95** | max | area/perimeter |
|---|---|---|---|---|
| uniform 3/5/9/15 px | exact | **exact** | exact | 2.99 / 4.96 / 8.83 / 14.47 |
| branched, true 3 px | **1.80** | **3.00** | 3.00 | 2.84 |
| branched, true 9 px | **7.79** | **9.00** | 10.59 | 8.60 |
| tapered 3→15 px | 9.00 | **15.00** | 15.00 | 8.85 |

- **median** fails on branched cracks — the junction contributes a cluster of
  short distances, a 40% underestimate on a 3 px fork
- **max** is set by one ragged junction pixel — 10.59 on a true 9.00
- **area/perimeter** is the mean over the whole component: biased low by
  branching, and describes neither end of a taper
- **p95** is exact on uniform *and* branched, and lands on a taper's wide end —
  which is the end that gets graded

`width_mm` is **p95 of the ridge**. The others are kept as `width_median_mm`,
`width_max_mm`, `width_mean_mm`.

Length is estimated as `area / mean_width`, not the bounding-box diagonal — a
meandering crack is longer than its box.

---

## 7. Overlay and composite

Per camera: red fill at 60% over detected pixels, plus a yellow 1 px contour.
Mask resizing uses **`INTER_NEAREST`** — a mask is a label image, and
interpolating between 0 and 255 would invent partial-membership pixels that mean
nothing.

The composite is `hstack` of the three overlaid frames, downscaled for display,
captioned on the image itself. The caption is not decoration: someone looking at
three frames butted together will reasonably assume they were stitched and fed to
the model as one image — which is exactly the design this project rejected.

---

## 8. Validation

| Test | Checks |
|---|---|
| `test_tiler.py` (23) | Coverage, bounds, byte-identical sub-arrays, round-trip, union-on-overlap, connectivity across seams, blend window shape, blend rejects a seam spike the union keeps |
| `test_measure.py` (13) | Width against analytic shapes: straight, diagonal, branched, tapered |
| `test_verify.py` | Valley vs step separation; straightness monotonic in wander |
| `test_ae_sim.py` (7) | Exposure convergence from both extremes, including the stuck state |

Diagnostics: `analyze_tile_edges.py` (border artefact rate), `explain.py` (why
each detection was kept or dropped), `inject_crack.py` (plausible crack drawn
into a real captured frame).
