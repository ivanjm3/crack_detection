# 10 — Reproduce and extend

How to regenerate every number and figure, how to check the paper's claims
yourself, and what experiments to run next (with protocols).

---

## 1. What can be reproduced where

| Need | Can run on | Needs |
|---|---|---|
| Model accuracy, threshold sweep, METU histogram, training curve | **any PC** | `crack_outputs/*.npy,*.csv` (in the repo) |
| Tiler, width estimator, valley, straightness (+ their tests) | **any PC** | Python, NumPy, OpenCV, SciPy |
| Every figure | **any PC** | the above + `matplotlib`, `git` |
| Reference verification | any PC with internet | Crossref / DataCite |
| FP16 latency, parity, capture timings, exposure behaviour, USB probe | **Jetson only** | the board, cameras, engine |

So the paper's *data-derived* and *code-derived* numbers are all reproducible on
a laptop; its *hardware-derived* numbers (Tables V, VI, VII-hardware, XII;
Figs. 15, 16, 17) are recorded measurements that need the board.

---

## 2. Set up a PC

```powershell
python -m pip install numpy opencv-python-headless scipy matplotlib
git fetch origin accuracy-filters          # verify.py and the 23-check tiler live here
```

---

## 3. Regenerate every figure

```powershell
python docs\documentation\figures\make_figures.py
```

What it does:

1. Creates `docs/documentation/figures/_code/` and fills it with
   `git show accuracy-filters:jetson/{tiler,measure,verify,test_measure,test_verify}.py`.
   (Git-ignored; `verify.py` exists only on that branch.)
2. Writes 18 PNGs beside the script and `computed_values.json` (the measured
   widths, valley depths and straightness scores).

Each function's docstring states the figure's provenance (DATA / CODE / TABLE /
DERIVED / SCHEMATIC). Nothing is drawn from a number typed into a plot that is
not either in a data file, produced by the shipped code, or labelled as a
recorded measurement.

---

## 4. Re-run the project's own tests

```powershell
cd docs\documentation\figures\_code
$env:PYTHONPATH = "."
python test_measure.py      # width vs analytic shapes
python test_verify.py       # valley vs step; straightness
```

Expected tail of the first: `worst error on the reported width: 1.00 px` and
`PASS`. For the tiler:

```powershell
git show accuracy-filters:jetson/test_tiler.py > test_tiler.py
git show accuracy-filters:jetson/tiler.py      > tiler.py
python test_tiler.py        # 23/23 passed
```

> Use `PYTHONPATH=.`, **not** `python -I`: isolated mode drops the script's
> directory from `sys.path` and the imports fail.

---

## 5. Recompute the headline numbers from data

```python
import numpy as np, csv

# ---- accuracy (Table III / IV, Fig. 13) ----
t = np.load("crack_outputs/test_sweep.npy")        # rows: thr, P, R, Dice, IoU
thr, P, R, D, I = t
k = int(np.argmin(abs(thr - 0.55)))
print("test @0.55:", P[k], R[k], D[k], I[k])        # 0.7609 0.8206 0.7896 0.6524
assert np.allclose(D, 2*P*R/(P+R), atol=1e-3)       # row order check
assert np.allclose(I, D/(2-D),    atol=1e-3)

# ---- training (Fig. 12) ----
rows = list(csv.DictReader(open("crack_outputs/train_log.csv")))
iou = np.array([float(r["val_iou"]) for r in rows])
print("best IoU %.4f at epoch %d; last-20 std %.4f" % (iou.max(), iou.argmax(), iou[40:].std()))

# ---- false positives (Fig. 14) ----
m = np.load("crack_outputs/metu_negative_pixels.npy")
print("n", m.size, "median", np.median(m), "mean %.0f" % m.mean(), "max", m.max())

# ---- geometry (Eq. 1, Fig. 21, Table XI) ----
import math
t = math.tan(math.radians(70.42/2))
gsd = lambda d, W=1920: 2*d*t/W*1000
print("GSD 0.4 m: %.3f  2px: %.3f   4.5 m: %.3f  2px: %.3f" %
      (gsd(.4), 2*gsd(.4), gsd(4.5), 2*gsd(4.5)))
```

---

## 6. Rebuild or audit the references

The bibliography is generated from `refs.json`, which is produced by
`verify_refs.py`: each candidate is an (identifier, expected title keywords)
pair, looked up in **Crossref** (DOIs) or **DataCite** (arXiv DOIs). A candidate
is kept only if the registry's title contains every keyword — a DOI that resolves
to a different paper is rejected rather than cited.

Three candidates failed and show why this exists:

| DOI tried | Resolved to |
|---|---|
| `10.1016/j.isprsjprs.2018.07.003` | a rubber-plantation stand-age paper |
| `10.1109/ICRA.2017.7989443` | *PennCOSYVIO*, a visual-inertial odometry benchmark |
| `10.1109/IROS.2004.1389727` | valid, but no date field (dropped when simulation left the paper) |

Notes: Crossref's `issued` is the *online* date; IEEE style wants the volume's
print year (`published-print`), which the script prefers. `export.arxiv.org`
fails certificate verification from the lab PC, so arXiv is queried through
DataCite with verification left **on**.

---

## 7. The paper build (local)

The Word paper is built by a Python script that holds all the prose, tables,
figure placement and the generated reference list. It lives in `paper/` — which
is **git-ignored on purpose** (the unpublished manuscript is not pushed). If you
have that folder:

```powershell
python paper\refs\verify_refs.py       # -> refs/refs.json
python paper\figs\make_figures.py      # -> figs/out/*.png
python paper\build_paper.py            # -> crack_inspection_paper.docx
```

(If Word has the `.docx` open, the build writes `..._rebuilt.docx` beside it.)

**Cross-references are hard-coded text** ("Fig. 10", "Table XI"). If you add or
reorder a figure/table, re-check every one.

---

## 8. What to run next — with protocols

These are ordered by how much each would strengthen the paper.

### 8.1 Calibrate the scale *(highest value; closes the biggest threat)*

**Goal.** Replace the assumed 0.40 m with a measured mm/px.

1. Fix the camera at its working distance. Place a **flat object of known
   width** in frame, parallel to the sensor — e.g. a bank card (85.60 mm), or a
   printed ruler.
2. Capture a frame; measure the object's width in pixels (`cv2.selectROI`, or
   threshold + `connectedComponentsWithStats`).
3. `mm_per_px = known_mm / measured_px`.
4. Run with `python3 scanner.py --mm-per-px <value>` (overrides `--standoff`).
5. **Validate**: place a *second* known object, report its measured width and the
   error in mm. *A calibration nobody checked is a calibration that is wrong.*

For lens distortion, use a checkerboard and Zhang's method
(`cv2.calibrateCamera`): the C920 has visible barrel distortion at the frame
edges, which bends straight cracks.

### 8.2 Verify 1080p uses the whole sensor

```bash
python3 check_fov_parity.py --cam 0 --frames 8
```

Point at a static, **textured** wall. It captures 720p and 1080p, then searches
for the crop factor `f` that best aligns the central `f` of the 1080p frame to
the 720p frame. `f ≈ 1.00` ⇒ same field of view; `f < 1` ⇒ 1080p is cropped and
every GSD in the paper must be rescaled (too large by `1/f`).

### 8.3 Fix focus

```bash
python3 focus.py sweep       # (branch) variance of the Laplacian vs focus_absolute
python3 focus.py apply       # writes focus.json
```

Needs a **static, textured target at the working distance**; a sweep against a
blank wall returns noise. Report the sharpness curve (it would make a good
figure) and the chosen value.

### 8.4 Genuine-crack validation *(the other big gap)*

1. Find real cracks in a continuous surface (concrete wall, pavement) and measure
   a few widths with a **crack-width comparator card or calibrated microscope**.
2. Photograph them at the working standoff, in focus, with the calibrated scale.
3. Run `explain.py` (branch) to see, per component, what each filter measured
   and what it decided — a real crack should be kept **with margin**, not at
   2.1 against a threshold of 2.0.
4. Report: detected / missed, measured width vs reference width (mm error),
   and the valley/straightness scores of each.

If no real cracks are available, `inject_crack.py` draws a plausible one into a
real captured frame — weaker evidence, but it exercises the filters on a real
background.

### 8.5 Ablations that would be cheap and informative

| Experiment | How | What it shows |
|---|---|---|
| Union vs blend recall | Inject known cracks across tile seams; compare `main` vs branch | whether blending *costs* recall |
| Each filter on/off | `--no-shape-filter`, `--no-verify`, `--min-valleyness 0`, `--max-straightness 0` | marginal value of each |
| Averaging N | `--average 1,4,16` | noise vs time |
| Threshold sweep on the *deployment* domain | with injected cracks + crack-free frames | an operating point for tunnels, not pavement |

### 8.6 Close-range pass

Measure detectability vs standoff with a printed crack target of known width
(e.g. widths 0.2, 0.3, 0.5, 1 mm) at 0.25, 0.4, 0.6, 1.0 m: that is the
empirical version of Fig. 21.

---

## 9. Checklist before quoting a number

- [ ] Is it in the fact sheet (`context.md` §4) with a source?
- [ ] Was the GPU clock locked when it was timed?
- [ ] Does it depend on the assumed 0.40 m standoff? If so, say "assuming…".
- [ ] Is it CRACK500 (pavement) or the deployment domain?
- [ ] Is it one scene or several?
- [ ] Worst case reported, not the pass bar?

---

## 10. Troubleshooting

| Problem | Likely cause |
|---|---|
| `make_figures.py`: "cannot read accuracy-filters:jetson/verify.py" | branch not fetched: `git fetch origin accuracy-filters` |
| `ModuleNotFoundError: measure` running a test | use `PYTHONPATH=.`, not `python -I` |
| `UnicodeEncodeError` printing titles on Windows | `sys.stdout.reconfigure(encoding="utf-8")` |
| Crossref HTTP 429 | searching too fast; the `/works/<doi>` endpoint was fine |
| Figures look different from the docs | check the matplotlib font (Times New Roman → DejaVu Serif fallback) |
| Jetson unreachable | `cracknet link`; join `CrackNet`; USB `192.168.55.1` is the fallback |
