# context.md — project and paper handoff

Written 2026-10-10. Purpose: let a fresh session (or a person) **(a)** update the
existing paper, or **(b)** write a different paper, using only this file plus the
repo. Everything here was checked against the repo or a measured run; where
something is unknown it says so.

If you read nothing else, read **§3 (what is and is not built)** and **§11
(rules for writing)**. The old paper's biggest problem was claiming things that
were never built.

---

## 1. The project in one paragraph

CrackNet: a crack-segmentation model (U-Net, MobileNetV3-Large encoder, trained
on CRACK500) deployed on an **NVIDIA Jetson Orin Nano 8 GB** with **three
Logitech C920** cameras (left / top / right) on a **stationary** rig. Cameras are
read **one at a time** (USB 2.0 bandwidth limit), each 1080p frame is **tiled at
native resolution**, inferred with a **TensorRT FP16** engine (8.77 ms/tile),
tile logits are **blended**, components are **verified** (shape, valley,
straightness) and **measured in millimetres** from the distance-transform ridge.
An HTTP **console** shows results; the Jetson hosts its own Wi-Fi AP and a
Windows batch launcher drives it. Target use: tunnel-lining inspection.

Team (from the paper's author block, unchanged from the original): Dr. Prem
Shankar, Vishnu CV, Ivan J Madathil, Akash VP — School of Computer Science and
Engineering, Vandalur-Kelambakkam road, Chennai.

---

## 2. Where everything lives

| Need | File |
|---|---|
| Phase 2 overview + chronology | `docs/phase2/README.md` |
| Camera / USB / exposure / capture | `docs/phase2/01-camera-weaving.md` |
| Perception: tiling, filters, measurement | `docs/phase2/02-perception.md` |
| What every console number means | `docs/phase2/03-ui.md` |
| Phase 1 work log (first deployment, AE, false positives) | `docs/WORKLOG-2026-09-16.md` |
| Original design reasoning, USB research, 4A measurements | `docs/multicam-plan.md` |
| Phase 1 pipeline reference with diagrams | `docs/perception.html` |
| File index + where `setup.md.txt` is wrong | `jetson/README.md` |
| Network (why the Jetson hosts an AP) | `docs/network-link.md` |
| Copy-paste operating commands | `RUN.md`, `docs/commands.md` |
| Training log, sweeps, METU blob sizes | `crack_outputs/` |
| Unbuilt Gazebo plan (excluded from the paper by the user) | `docs/phase3/gazebo-plan.md` |
| The paper (build system) | `paper/` — see §10 |

Code (Jetson-side, `jetson/`): `capture.py`, `camera_ctl.py`, `camera_setup.sh`,
`tiler.py`, `trt_infer.py`, `live.py` (has `clean_mask`), `measure.py`,
`scanner.py`, `serve_scan.py`, `cracknet.sh`, `netlink.sh`. Host-side (`tools/`):
`jssh.py`, `jlink.py`, `jproxy.py`, `demo_server.py`. Launcher: `cracknet.bat`.
Console: `viewer/dashboard.html`.

---

## 3. What is and is not built — the truth table

| Claim | Status | Evidence |
|---|---|---|
| Model trained on CRACK500, leak-free split | **Built, measured** | `crack_outputs/train_log.csv`, `*_sweep.npy`, `setup.md.txt` |
| TensorRT FP16 on-device, parity checked | **Built, measured** | `docs/WORKLOG-2026-09-16.md` §4 |
| Three-camera sequential capture | **Built, measured** | `docs/phase2/01-camera-weaving.md` |
| Exposure controller (ladder-aware) | **Built; sim 7/7; real cameras settled** | §4 below |
| Native tiling + union reassembly | **Built, tested (13 checks on `main`)** | `jetson/test_tiler.py` |
| Logit blending, valley + straightness filters | **Built, on branch `accuracy-filters` only** | 23/23 tiler checks, `test_verify.py` |
| Width in mm from distance-transform ridge | **Built, validated on analytic shapes** | `jetson/test_measure.py` |
| Console, live + scan modes | **Built** | `docs/phase2/03-ui.md` |
| Jetson-hosted Wi-Fi AP, batch launcher | **Built, reboot-tested** | `docs/network-link.md` |
| **Calibrated millimetres** | **NOT done** — assumes 0.40 m standoff | open item |
| **Recall on a genuine crack in a continuous surface** | **NOT measured** | open item |
| **Focus** | **Unresolved** — pinned at 30 (~infinity) | `01-camera-weaving.md` §10 |
| **ArUco / fiducial wall-frame mapping** | **NOT built** (was in the old paper) | no code anywhere |
| **Persistent crack IDs, repeat-pass change detection** | **NOT built** (old paper) | no code anywhere |
| **EV3 rover / rail trials** | **NOT built** (old paper) | no code anywhere |
| **Pi 5 CPU-only comparison** | **NOT done** (old paper) | no code anywhere |
| **Moving carriage** | **NOT built** — every result is stationary | — |
| **Gazebo simulation** | **NOT built**; user said to keep it out of the paper | `docs/phase3/` |
| **INT8 engine** | **NOT done** | `setup.md.txt` mentions it as a possibility only |

---

## 4. Fact sheet (numbers safe to quote) and where each came from

Prefer a number from this table over one remembered. "Re-run" means it was
reproduced locally on this PC on 2026-10-10 from the repo's own code.

### Model and training
| Fact | Value | Source |
|---|---|---|
| Architecture | `smp.Unet`, `timm-mobilenetv3_large_100`, decoder (256,128,64,32,16), 1 logit | `setup.md.txt` |
| Data | CRACK500, 2333 / 526 / 509 crops, rebuilt by parent photo (70/15/15) | `setup.md.txt` |
| Why rebuilt | Official splits leak 5 parent photos train↔test and 1 val↔test | `setup.md.txt` |
| Class balance | ~6.3% crack pixels, median 5%, no empty masks | `setup.md.txt` |
| Loss | Dice + focal (weights **not recorded**) | `setup.md.txt` |
| Augmentation | RandomResizedCrop 448, flips, rot90, brightness/contrast, noise, motion blur, JPEG | `setup.md.txt` |
| Epochs / time | 60 epochs, ~1.2 h | `train_log.csv` |
| LR | warm-up 3 epochs to 3e-4, cosine to ~0 | `train_log.csv` |
| Best val IoU / Dice | 0.6505 / 0.7883 at epoch 34 | `train_log.csv` |
| Final val IoU / Dice | 0.6434 / 0.7830 (epoch 59) | `train_log.csv` |
| Val @ T=0.55 | P 0.7752, R 0.8101, Dice 0.7923, IoU 0.6560 | `val_sweep.npy` |
| **Test @ T=0.55** | **P 0.7609, R 0.8206, Dice 0.7896, IoU 0.6524** | `test_sweep.npy` |
| Sweep array layout | rows = threshold, precision, recall, Dice, IoU; 17 thresholds 0.10–0.90 | verified: F1=2PR/(P+R), IoU=D/(2−D) reproduce |
| METU negatives | 10.85% of 20 000 crack-free images fire ≥1 px at 0.55; median blob 165 px, mean 734 | `setup.md.txt`, `metu_negative_pixels.npy` (n=2169) |
| **Not recorded anywhere** | optimiser, batch size, training hardware, loss weights | do not invent |

Note: sweep IoU/Dice (0.656/0.792) differ slightly from the per-epoch log
(0.650/0.788). They come from different scripts; the paper says so.

### Deployment (Jetson Orin Nano 8 GB, JetPack 6.2.3, TensorRT 10.3.0, CUDA 12.6)
| Fact | Value | Source |
|---|---|---|
| FP16 / FP32 engine | 8.77 ms (113.7 qps) / 18.44 ms; 2.1× | WORKLOG §4 |
| Parity TRT vs ONNX | 0.0552% mask disagreement, bar 0.1%; 38% of pixels cleared thr. | WORKLOG §4 |
| Power | `nvpmodel -m 2` = MAXN_SUPER (**mode 0 is 15 W**, not max); `jetson_clocks` GPU 1020 MHz, CPU 1.73 GHz | WORKLOG §3 |
| `jetson_clocks` does not survive reboot | 306 MHz idle → 2.6× slower, silently | `multicam-plan.md` §4A.3 |
| Phase 1 live | ~15–19 fps, 18.4 ms inference, capture-bound ~22 fps | `jetson/README.md` |
| 1080p, 15 tiles, per camera | 221.1 ms = 89.0 pre + 130.3 inf + 1.7 tile/stitch | `bench_tiling.py` |
| 720p, 6 tiles, per camera | 88.2 ms = 35.4 + 52.1 + 0.6 | `bench_tiling.py` |
| Logit-space threshold | pixel-exact (0 / 3,932,160 differ); 169 → 130 ms | `multicam-plan.md` §4A.3 |

### Capture and exposure
| Fact | Value | Source |
|---|---|---|
| Bus | every Type-A port behind one Realtek USB 2.0 hub, 480 Mbit/s | `multicam-plan.md` §3 |
| 640×480 MJPG 15 | 3 of 3 stream, 15.0 fps each | Phase 0 |
| 1280×720 MJPG 30 | **2 of 3**; third opens, never delivers | Phase 0 |
| uvcvideo quirks | `0xFFFFFFFF`, already all set | WORKLOG / 01 §3 |
| YUYV 720p30 vs MJPG | 442 Mbit/s vs ~45; practical payload ~320 | `perception.html` §1 (computed 1280·720·16·30) |
| Per camera | first frame after STREAMON 675 ms; release 238; flush 152; open 55; setup 30 | `01-camera-weaving.md` §4 |
| Cycle | open 0.99 + confirm 0.53 + grab 0.22 + **release 0.24** = 1.98 s/cam; ×3 = **5.93 s**; + 0.66 s inference = **~6.6 s**; capture ≈ 90% | note: the docs' table omits the release row, so its parts sum to 5.22 not 5.93 |
| **Exposure ladder** | while streaming 1080p only **38, 77, 156, 312, 624** accepted; idle driver accepts all 747 of 3–2047 | `01-camera-weaving.md` §6 |
| Luma consequence | 99 or 148, never 120 | same |
| Real-camera result | exp=156, gain=0, luma 94–99, 0.2% clipped, converged every pass | same |
| AE sim | 7/7 scenarios | `test_ae_sim.py`, WORKLOG §8 (first control law, exp cap 500) |
| Failed designs | p99=235 limit-cycled 311↔532; anti-windup ceiling drove exp=3, gain=255 | WORKLOG §8 |
| Transients | at fixed exp=500 gain=0 consecutive medians were 4 and 241 | WORKLOG §8 |

### Perception
| Fact | Value | Source |
|---|---|---|
| hfov | **70.42°** horizontal at 16:9 (78° is the diagonal; using it overstates swath ~15%: tan 39° / tan 35.21° = 1.148) | `02-perception.md` §6 |
| GSD @ 0.40 m, 1920 px | 0.294 mm/px; swath 0.56 m/camera; 2-px crack ≈ 0.59 mm | formula `2·d·tan(hfov/2)/W` |
| Stitching rejected | 5760 px → 512 is ×0.089: 2-px crack → 0.18 px (0.31 px was for the earlier 3328-px mosaic plan); seams are crack-shaped | `01-camera-weaving.md` §1 |
| Tiling | 512², 15% overlap, 15 tiles @1080p, 6 @720p; edge tiles clamped not padded | `tiler.plan_tiles` (re-run) |
| vs Phase 1 | 1.8× swath, 2.1× finer, simultaneously | `02-perception.md` §1 |
| Tile-border artefact | 2.18× raw, **11.33× after clean_mask**; 3 frames × 15 tiles | `analyze_tile_edges.py` |
| Blend effect on coverage | 75.2→63.7, 39.6→24.0, 18.0→8.5 (%) | `02-perception.md` §3 |
| Shape filter | person: 38% → 1.44% (96.2% removed); 51 components | WORKLOG §7 |
| Component stats | thin synthetic crack half-width 1.94 px, solidity 0.041; false blobs median 11.2 px (max 43), 0.715 (max 1.09) | WORKLOG §7 |
| Out-of-domain | raw 8–32% → 0.2–1.5% after cleaning | `02-perception.md` §4 |
| min_area / max_halfwidth | 600 px / 15 mm (were 300 px / 12 px; rescaled for native tiling) | `02-perception.md` §4 |
| Curtain + cables | main: CRACK, 18 comps, 7.4%, "23.34 mm"; branch: clear, 0; all 18 fell to valley; straightness 1.45–2.29 | `02-perception.md` §5 |
| Valley (re-run) | 14.28, 26.49, 6.06 (cracks); 0.00, 0.00 (steps). First Hessian-eigenvalue version ranked step 29.9 over crack 26.4 | `test_verify.py` |
| Straightness (re-run) | 0.946, 1.965, 3.546, 7.105, 15.456; reject <1.12 and >40 mm | `test_verify.py` |
| Width (re-run) | uniform 3/5/9/15 exact; branched within 0.75 px; taper wide end ≤1.0 px; **worst 1.00 px** | `test_measure.py` |
| Width estimator | headline = ridge **p95**; width = 2d−1 | `02-perception.md` §6 |
| Width bugs found | 302-px reading of a 2-px line (no bg in tight bbox); 2d vs 2d−1; `cv2.line` thickness 3 draws 5 rows | same |
| Tests (re-run) | tiler 23/23 (branch), measure PASS, verify PASS | 2026-10-10 |
| Standoff table | 0.25 m→0.36 mm … 4.5 m→6.62 mm, 5.5 m→8.09 mm | `gazebo-plan.md` §2 (analytic) |
| Scan vs live | 223 ms vs 15 ms /camera; 0.29 vs 0.83 mm/px; 0.59 vs 1.65 mm; 10 fps measured | `01-camera-weaving.md` §9 |

### The cardboard case (use carefully)
User pointed the rig at a cardboard sample with a visible crack. The `main`
configuration (before blending + verification) flagged it; the branch did not.
Diagnosis: **not a filter fault**. Raw model output 0.27%, all four components on
the bottom frame edge; the feature was a **torn-edge silhouette (a step)**; the
frame was **defocused** (variance of Laplacian 16.6, sharp frames are hundreds).
`main` was kept for the teacher demo for this reason; `accuracy-filters` holds
the rest.

---

## 5. Repo and branch state

- `main` (head `7b22ee9`): union reassembly, shape filter only. This is the
  **demo** configuration. Includes the network/AP work and `cracknet.bat`.
- `accuracy-filters` (head `7fb7b7f`): blending, `verify.py`, `explain.py`,
  `focus.py`, `inject_crack.py`, 23-check `test_tiler.py`, `test_verify.py`.
  **Not merged**; it also does not contain `main`'s later network commits.
- `paper/` is **untracked**. Nothing in the paper work has been committed or
  pushed. `tools/.jetson.env` (credentials) is gitignored and must stay so.
- Merging `accuracy-filters` after the demo is a pending task.

---

## 6. Negative results and lessons (these make good paper material)

1. **Stitch-then-infer** rejected on measured resolution collapse and seam artefacts.
2. **Simultaneous 720p** impossible on one USB 2.0 hub; every driver quirk already set.
3. **Exposure actuator is a 1-stop ladder** — the AE loop could not converge until
   it was designed around it; "drifted 156→312" was one rung, not drift.
4. **Shape filter amplifies tile-border artefacts** (2.18× → 11.33×) because a
   border line is the geometry it keeps.
5. **Valley test v1 failed its own test** (Hessian eigenvalue ranks steps above cracks).
6. **Width estimator** wrong three ways before it was right (§4).
7. **Fixed focus** at an absolute value near infinity silently defocuses close work.
8. **Silent failures recur**: FOURCC after resolution → 10 fps; exposure lock
   discarded at stream start; `jetson_clocks` lost on reboot; ICMP and NTP
   blocked on campus Wi-Fi; Windows not re-joining an SSID after a Jetson reboot.

---

## 7. Open items (what blocks stronger claims)

1. **Calibrate scale** (`--mm-per-px`); until then every mm figure is conditional on 0.40 m.
2. **Fix focus**; needs a static, textured target for `focus.py`.
3. **Validate recall on a real crack** in a continuous surface with measured width.
4. **Decide the smallest crack that must be detected** — fixes standoff and swath.
5. **Merge `accuracy-filters`** into `main` after the demo.
6. Console is **unauthenticated** and binds `0.0.0.0`.
7. Server does not survive reboot (`nohup`, not a service).
8. Training hyper-parameters not recorded (§4).
9. **1080p field of view is unverified.** `jetson/check_fov_parity.py` was written to
   test that 1080p reads the whole sensor (not a cropped window) but no result is
   recorded anywhere. If 1080p is cropped, every GSD is too large and every width
   overstated by the same factor. The paper lists this under threats to validity.

---

## 8. The paper as it stands

`paper/crack_inspection_paper.docx` — 21 figures (3 hardware-photo
**placeholders**), 12 tables, 5 equations, **42 references, all verified**,
single-column Times New Roman, 19 pages. Formatting is intentionally plain; a
two-column IEEE layout is a later pass.

**Title (changed from the original):** *Stationary Multi-Camera Crack Detection
and Metric Width Measurement on Edge Hardware for Tunnel-Lining Inspection.*
The original, *Motion-Aware Crack Localization and Mapping for Structural
Inspection on Edge Hardware*, described unbuilt work.

**Sections:** Abstract · Keywords · I Introduction · II Related Work (A–F) ·
III Methodology (A–K) · IV Results (A–J) · V Discussion · VI Conclusion ·
VII Future Work · References.

**Figures (Fig. N in the document):** 1 architecture · 2–4 photo placeholders ·
5 rig geometry · 6 USB bandwidth · 7 exposure ladder · 8 tile plan · 9 blend ·
10 pipeline · 11 console · 12 training · 13 threshold sweep · 14 METU blobs ·
15 latency · 16 capture timeline · 17 false-positive reduction · 18 valley ·
19 straightness · 20 width estimators · 21 resolution vs standoff.

**Provenance of figures** (also in the docstrings of `make_figures.py`):
DATA = read from `crack_outputs/` (12, 13, 14). CODE = produced by running the
shipped `tiler`/`measure`/`verify` (8, 9, 18, 19, 20). TABLE = measured value from
a doc (15, 16, 17, 7). DERIVED = from the GSD formula (6, 21). SCHEMA = diagrams
with no data (1, 5, 10, 11). Fig. 5 and 11 are schematics and say so.

**Decisions already made — do not re-litigate:**
- **No simulation / Gazebo** anywhere in the paper (user instruction).
- The unbuilt old-paper contributions moved to **Future Work**, not claimed.
- **0.3 mm** is described as the project's *working target*, not a cited standard.
- Hardware photos are the **only** placeholders; everything else is produced.
- Author block unchanged.

**Tables:** I positioning vs prior work · II platform · III CRACK500 accuracy ·
IV threshold effect · V latency · VI capture modes · VII exposure sim · VIII valley
· IX straightness · X width estimators · XI GSD vs standoff · XII scan vs live.
In-text cross-references were hand-checked against this numbering; if you add or
reorder a figure or table, **re-check every "Fig. N" / "Table N" in
`build_paper.py`** — they are hard-coded text, not fields.

---

## 9. How to rebuild or update the paper

```powershell
cd C:\Users\student\Desktop\rp_proj

# 1. snapshot the shipped modules the figures are computed from
git show accuracy-filters:jetson/tiler.py        > paper/_code/tiler.py
git show accuracy-filters:jetson/measure.py      > paper/_code/measure.py
git show accuracy-filters:jetson/verify.py       > paper/_code/verify.py
git show accuracy-filters:jetson/test_measure.py > paper/_code/test_measure.py
git show accuracy-filters:jetson/test_verify.py  > paper/_code/test_verify.py

python paper/refs/verify_refs.py      # re-verify every reference -> refs/refs.json
python paper/figs/make_figures.py     # 18 PNGs -> paper/figs/out/
python paper/build_paper.py           # -> paper/crack_inspection_paper.docx
```

Dependencies: `python-docx`, `matplotlib`, `numpy`, `opencv-python-headless`,
`scipy`, `pymupdf` (only for inspecting the old PDF), `pdf2docx` (baseline only).

| File | Role |
|---|---|
| `paper/build_paper.py` | **All paper text** + tables + figure placement + bibliography. Edit prose here. |
| `paper/figs/make_figures.py` | Every figure; data provenance in docstrings. |
| `paper/refs/verify_refs.py` | Candidate DOIs + expected-title keywords; verifies against Crossref/DataCite. |
| `paper/refs/refs.json` | Verified registry output. **Authors/venue/pages come from here.** |
| `paper/_code/` | Snapshot of branch modules used by the figure script. |
| `paper/_old/`, `paper/_render/` | Old paper text/pages and render outputs; disposable. |
| `paper/figs/_patch_*.py` | One-off patches already applied; kept only as history. |

**Adding a citation:** add `("key", "10.xxxx/...", ["title","keywords"])` to
`CANDIDATES` in `verify_refs.py`, run it, and only use the key if it is
reported `ok`. Cite in prose as `[[key]]` or `[[k1,k2]]`; numbering is by order
of first appearance and the reference list is generated.

**Render check:** the Word file was visually checked by converting with
LibreOffice (`soffice --headless --convert-to pdf`). LibreOffice was **extracted,
not installed**, into the session scratch folder
(`winget download TheDocumentFoundation.LibreOffice -d <dir>` then
`msiexec /a <msi> /qn TARGETDIR=<dir>`); it will be gone in a new session.

**Not yet done for the paper:** two-column IEEE layout; the original paper
reproduced in clean Word (only the rough `pdf2docx` baseline exists); real
photos; real console screenshots; equations are plain Unicode text, not Word
equation objects.

---

## 10. Writing a different paper from this repo

All of these are supportable by evidence already in the repo. Each lists what is
missing.

| Angle | Core claim | Evidence in repo | Missing |
|---|---|---|---|
| **A. Sequential capture on a shared USB 2.0 hub** | Stationary rigs turn a bandwidth wall into a ~6 s time cost | Phase 0 table, cycle breakdown, YUYV/MJPG arithmetic | A second camera model; a moving-rig comparison |
| **B. Actuator quantisation breaks AE loops** | A 1-stop exposure ladder makes median-luma setpoints unreachable; design around it | Ladder measurement, 3 failed controllers, 7/7 sim, real-camera settle | Other camera models; a closed-loop ablation on real scenes |
| **C. Tiling amplifies seams under shape filtering** | Border artefacts 2.18× → 11.33×; logit blending removes them | `analyze_tile_edges.py`, coverage drops, `test_tiler.py` | A labelled set to measure the effect on *recall* |
| **D. Valley vs step verification** | Two-sided minimum separates dark lines from steps; straightness removes seams | `test_verify.py`, curtain scene 18→0, v1 failure story | Real cracks to show it does not reject them; ROC |
| **E. Resolution budget for COTS tunnel inspection** | At 4.5 m a C920 resolves ~6.6 mm, >20× the 0.3 mm target | GSD formula, standoff table | A calibrated field measurement; a lens/standoff comparison |
| **F. Systems report** (what the current paper is) | End-to-end edge pipeline with honest validity | everything | Calibration, genuine-crack recall |

For any of them: compute from the shipped code (as `make_figures.py` does)
rather than quoting a doc, and put anything unmeasured in a limitations list.

---

## 11. Rules for writing (learned the hard way)

1. **Do not claim anything in the §3 "NOT built" rows.** Put it in Future Work.
2. **Every reference must pass `verify_refs.py`.** Three DOIs recalled from memory
   were wrong (one resolved to a rubber-plantation paper; one to a VO benchmark;
   one had no registry record). The original paper's ref [6] was mis-titled:
   `arXiv:2412.07205` is *CrackESS*, not "Crack-EdgeSAM".
3. **Label anything illustrative.** Schematics and synthetic demonstrations say so
   in the caption. No invented sample values in diagrams.
4. **Use 70.42°, not 78°**, and say mm figures assume the standoff.
5. **Report the worst case, not the pass criterion** (width worst error is
   1.00 px even though the test's bar for uniform strokes is 0.75 px).
6. **Do not quote a timing without the clock state.** 306 MHz makes everything 2.6×.
7. **State the sweep-vs-log IoU difference** rather than silently mixing them.
8. **Do not put credentials, IPs of a private network, or the AP password in a paper.**

---

## 12. Tooling gotchas (will waste time otherwise)

- **The Bash tool mangles backslashes in inline heredocs** (`\n` becomes a real
  newline inside string literals). Write patch scripts with the **Write tool**
  and run the file. This bit three times during the paper work.
- In patch scripts, the figure file holds **literal** `·`, `×`, `≈`, not
  `\u00b7`; match on the literal characters or the anchor will not be found.
  Patches here assert each anchor matches exactly once so a miss fails loudly.
- `python -I` drops the script directory from `sys.path`; run the project's tests
  with `PYTHONPATH=.` instead.
- `export.arxiv.org` fails certificate verification from this PC; arXiv DOIs are
  looked up through **DataCite** instead (verification stays on). Crossref returns
  HTTP 429 if searched rapidly; the `/works/<doi>` endpoint was fine.
- Crossref `issued` is the *online* date; the volume's year is
  `published-print`. IEEE style wants the latter.
- Windows console is cp1252: reconfigure stdout to UTF-8 before printing titles.
- `pdf2docx` output is not usable as a working file (16 sections, every paragraph
  `Normal`, 10 pages for a 5-page paper); rebuild with real styles instead.

---

## 13. Access to the Jetson (no secrets here)

Credentials live only in `tools/.jetson.env` (gitignored, never committed).
Routes are tried in order: `10.42.0.1` (the Jetson's own `CrackNet` AP) then
`192.168.55.1` (USB). Campus Wi-Fi cannot carry PC↔Jetson traffic (client
isolation, measured) and the wired jack does not lease to the Jetson's MAC.
`cracknet.bat live|scan|stop|status|logs|check|join|link|net` from the project
root; see `RUN.md`. At last check the board was **not reachable** (no USB
adapter, no AP) — it may have been powered off.

---

## 14. Pending questions for the user

1. Real hardware photos for Figs. 2–4, and real console screenshots?
2. Training hyper-parameters (optimiser, batch size, hardware, loss weights)?
3. Was any ArUco / change-detection / rover work done outside this repo?
4. The smallest crack width the survey must detect (fixes the standoff).
5. Venue and template (IEEE two-column? page limit?) for the formatting pass.

---

## 15. Reference registry

All entries below were retrieved from Crossref or DataCite and title-matched by
`paper/refs/verify_refs.py`. Authors, venue, volume, issue and pages are the
registry's, not typed. "Used for" says what the paper cites each one **for**; do
not cite it for more than that.

| # | Key | Citation | DOI | Used for |
|---|---|---|---|---|
| [1] | `spencer2019` | B. F. Spencer, V. Hoskere, and Y. Narazaki, “Advances in Computer Vision-Based Civil Infrastructure Inspection and Monitoring,” Engineering, vol. 5, no. 2, pp. 199–222, 2019. | https://doi.org/10.1016/j.eng.2018.11.030 | Computer-vision civil-infrastructure inspection overview |
| [2] | `mohan2018` | A. Mohan and S. Poobal, “Crack detection using image processing: A critical review and analysis,” Alexandria Engineering Journal, vol. 57, no. 2, pp. 787–798, 2018. | https://doi.org/10.1016/j.aej.2017.01.020 | Crack detection by image processing, critical review |
| [3] | `montero2015` | R. Montero, J. Victores, S. Martínez, A. Jardón, and C. Balaguer, “Past, present and future of robotic tunnel inspection,” Automation in Construction, vol. 59, pp. 99–112, 2015. | https://doi.org/10.1016/j.autcon.2015.02.003 | Robotic tunnel inspection, past/present/future |
| [4] | `attard2018` | L. Attard, C. J. Debono, G. Valentino, and M. Di Castro, “Tunnel inspection using photogrammetric techniques and image processing: A review,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 144, pp. 180–188, 2018. | https://doi.org/10.1016/j.isprsjprs.2018.07.010 | Tunnel inspection by photogrammetry + image processing, review |
| [5] | `dung2019` | C. V. Dung and L. D. Anh, “Autonomous concrete crack detection using deep fully convolutional neural network,” Automation in Construction, vol. 99, pp. 52–58, 2019. | https://doi.org/10.1016/j.autcon.2018.11.028 | FCN pixel-level crack segmentation; masks over patch classification |
| [6] | `liu2019` | Z. Liu, Y. Cao, Y. Wang, and W. Wang, “Computer vision-based concrete crack detection using U-net fully convolutional networks,” Automation in Construction, vol. 104, pp. 129–139, 2019. | https://doi.org/10.1016/j.autcon.2019.04.005 | U-Net concrete crack detection |
| [7] | `yang2020` | F. Yang, L. Zhang, S. Yu, D. Prokhorov, X. Mei, and H. Ling, “Feature Pyramid and Hierarchical Boosting Network for Pavement Crack Detection,” IEEE Transactions on Intelligent Transportation Systems, vol. 21, no. 4, pp. 1525–1535, 2020. | https://doi.org/10.1109/tits.2019.2910595 | CRACK500 dataset / FPHBN |
| [8] | `hendrycks17` | D. Hendrycks and K. Gimpel, “A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks,” arXiv preprint arXiv:1610.02136, 2016. | https://doi.org/10.48550/arXiv.1610.02136 | Out-of-distribution baseline (model cannot abstain) |
| [9] | `fujita2011` | Y. Fujita and Y. Hamamoto, “A robust automatic crack detection method from noisy concrete surfaces,” Machine Vision and Applications, vol. 22, no. 2, pp. 245–254, 2011. | https://doi.org/10.1007/s00138-009-0244-5 | Hand-crafted crack detection under noisy concrete surfaces |
| [10] | `ronneberger` | O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” Lecture Notes in Computer Science Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, pp. 234–241, 2015. | https://doi.org/10.1007/978-3-319-24574-4_28 | U-Net architecture |
| [11] | `zhang2021` | L. Zhang, J. Shen, and B. Zhu, “A research on an improved Unet-based concrete crack detection algorithm,” Structural Health Monitoring, vol. 20, no. 4, pp. 1864–1879, 2021. | https://doi.org/10.1177/1475921720940068 | U-Net adapted to thin, low-contrast crack pixels |
| [12] | `cha2017` | Y. Cha, W. Choi, and O. Büyüköztürk, “Deep Learning‐Based Crack Damage Detection Using Convolutional Neural Networks,” Computer-Aided Civil and Infrastructure Engineering, vol. 32, no. 5, pp. 361–378, 2017. | https://doi.org/10.1111/mice.12263 | CNN crack-damage classification in image patches |
| [13] | `zou2019` | Q. Zou, Z. Zhang, Q. Li, X. Qi, Q. Wang, and S. Wang, “DeepCrack: Learning Hierarchical Convolutional Features for Crack Detection,” IEEE Transactions on Image Processing, vol. 28, no. 3, pp. 1498–1512, 2019. | https://doi.org/10.1109/tip.2018.2878966 | DeepCrack benchmark |
| [14] | `ozgenel2018` | Ç. F. Özgenel and A. G. Sorguç, “Performance Comparison of Pretrained Convolutional Neural Networks on Crack Detection in Buildings,” Proceedings of the International Symposium on Automation and Robotics in Construction (IAARC) Proceedings of the 35th International Symposium on Automation and Robotics in Construction (ISARC), 2018. | https://doi.org/10.22260/isarc2018/0094 | METU building-crack dataset (used for crack-free negatives) |
| [15] | `hui2025` | L. Hui, A. Ibrahim, and R. Hindi, “Computer Vision-Based Concrete Crack Identification Using MobileNetV2 Neural Network and Adaptive Thresholding,” Infrastructures, vol. 10, no. 2, pp. 42, 2025. | https://doi.org/10.3390/infrastructures10020042 | MobileNetV2 + adaptive thresholding on embedded hardware |
| [16] | `dong2024` | X. Dong, Y. Liu, and J. Dai, “Concrete Surface Crack Detection Algorithm Based on Improved YOLOv8,” Sensors, vol. 24, no. 16, pp. 5252, 2024. | https://doi.org/10.3390/s24165252 | Lightweight YOLOv8 variant for crack detection |
| [17] | `dual2025` | M. Feng and J. Xu, “Lightweight Dual-Attention Network for Concrete Crack Segmentation,” Sensors, vol. 25, no. 14, pp. 4436, 2025. | https://doi.org/10.3390/s25144436 | Lightweight dual-attention crack segmentation network |
| [18] | `edgesam2024` | Y. Wang, J. He, and S. Yu, “CrackESS: A Self-Prompting Crack Segmentation System for Edge Devices,” arXiv preprint arXiv:2412.07205, 2024. | https://doi.org/10.48550/arXiv.2412.07205 | Self-prompting crack segmentation for edge devices (CrackESS; NOT 'Crack-EdgeSAM') |
| [19] | `energy2026` | M. Tschope, M. Moursi, V. Rybalkin, B. Zhou, N. Wehn, and P. Lukowicz, “A Case Study on Energy-Efficient Edge AI Crack Segmentation,” arXiv preprint arXiv:2604.13933, 2026. | https://doi.org/10.48550/arXiv.2604.13933 | Energy / hardware case study of edge crack segmentation |
| [20] | `mittal2019` | S. Mittal, “A Survey on optimized implementation of deep learning models on the NVIDIA Jetson platform,” Journal of Systems Architecture, vol. 97, pp. 428–442, 2019. | https://doi.org/10.1016/j.sysarc.2019.01.011 | Optimised deep-learning deployment on NVIDIA Jetson, survey |
| [21] | `review2025` | R. Dai, R. Wang, C. Shu, J. Li, and Z. Wei, “Crack Detection in Civil Infrastructure Using Autonomous Robotic Systems: A Synergistic Review of Platforms, Cognition, and Autonomous Action,” Sensors, vol. 25, no. 15, pp. 4631, 2025. | https://doi.org/10.3390/s25154631 | Review of autonomous robotic crack detection; spatial grounding via SLAM |
| [22] | `sam3d2025` | P. Deng et al., “3D Modeling and Automated Measurement of Concrete Cracks via Segment Anything Refinement and Visual Inertial LiDAR Fusion,” arXiv preprint arXiv:2501.09203, 2025. | https://doi.org/10.48550/arXiv.2501.09203 | 3-D crack geometry via SAM refinement + visual-inertial-LiDAR fusion |
| [23] | `ge2025` | L. Ge and A. Sadhu, “Deep learning-enhanced smart ground robotic system for automated structural damage inspection and mapping,” Automation in Construction, vol. 170, pp. 105951, 2025. | https://doi.org/10.1016/j.autcon.2024.105951 | Ground robotic system for structural damage inspection and mapping |
| [24] | `huang2018` | H. w. Huang, Q. t. Li, and D. m. Zhang, “Deep learning based image recognition for crack and leakage defects of metro shield tunnel,” Tunnelling and Underground Space Technology, vol. 77, pp. 166–176, 2018. | https://doi.org/10.1016/j.tust.2018.04.002 | Crack and leakage recognition in metro shield tunnels |
| [25] | `attard2018b` | L. Attard, C. J. Debono, G. Valentino, and M. Di Castro, “Vision-based change detection for inspection of tunnel liners,” Automation in Construction, vol. 91, pp. 142–154, 2018. | https://doi.org/10.1016/j.autcon.2018.03.020 | Vision-based change detection for tunnel liners (Future Work) |
| [26] | `brown2007` | M. Brown and D. G. Lowe, “Automatic Panoramic Image Stitching using Invariant Features,” International Journal of Computer Vision, vol. 74, no. 1, pp. 59–73, 2007. | https://doi.org/10.1007/s11263-006-0002-3 | Feature-based panoramic stitching (why stitching was rejected) |
| [27] | `szeliski2006` | R. Szeliski, “Image Alignment and Stitching: A Tutorial,” Foundations and Trends® in Computer Graphics and Vision, vol. 2, no. 1, pp. 1–104, 2007. | https://doi.org/10.1561/0600000009 | Image alignment and stitching tutorial |
| [28] | `unel2019` | F. O. Unel, B. O. Ozkalayci, and C. Cigla, “The Power of Tiling for Small Object Detection,” 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 582–591, 2019. | https://doi.org/10.1109/cvprw.2019.00084 | Tiling for small-object detection |
| [29] | `akyon2022` | F. C. Akyon, S. Onur Altinuc, and A. Temizel, “Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection,” 2022 IEEE International Conference on Image Processing (ICIP), pp. 966–970, 2022. | https://doi.org/10.1109/icip46576.2022.9897990 | Slicing-aided inference (SAHI) |
| [30] | `rosenfeld` | A. Rosenfeld and J. L. Pfaltz, “Sequential Operations in Digital Picture Processing,” Journal of the ACM, vol. 13, no. 4, pp. 471–494, 1966. | https://doi.org/10.1145/321356.321357 | Distance transform (origin) |
| [31] | `borgefors` | G. Borgefors, “Distance transformations in digital images,” Computer Vision, Graphics, and Image Processing, vol. 34, no. 3, pp. 344–371, 1986. | https://doi.org/10.1016/s0734-189x(86)80047-0 | Distance transforms in digital images |
| [32] | `steger1998` | C. Steger, “An unbiased detector of curvilinear structures,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 20, no. 2, pp. 113–125, 1998. | https://doi.org/10.1109/34.659930 | Curvilinear-structure detection (motivates the valley test) |
| [33] | `frangi1998` | A. F. Frangi, W. J. Niessen, K. L. Vincken, and M. A. Viergever, “Multiscale vessel enhancement filtering,” Lecture Notes in Computer Science Medical Image Computing and Computer-Assisted Intervention — MICCAI’98, pp. 130–137, 1998. | https://doi.org/10.1007/bfb0056195 | Hessian-based vesselness (motivates the valley test) |
| [34] | `garrido2014` | S. Garrido-Jurado, R. Muñoz-Salinas, F. Madrid-Cuevas, and M. Marín-Jiménez, “Automatic generation and detection of highly reliable fiducial markers under occlusion,” Pattern Recognition, vol. 47, no. 6, pp. 2280–2292, 2014. | https://doi.org/10.1016/j.patcog.2014.01.005 | ArUco planar fiducial markers (Future Work: wall-frame pose) |
| [35] | `hinderer2025` | S. Hinderer, M. Scheffler, and B. Yang, “Investigation of ArUco Marker Placement for Planar Indoor Localization,” arXiv preprint arXiv:2509.17345, 2025. | https://doi.org/10.48550/arXiv.2509.17345 | ArUco marker placement for planar indoor localisation |
| [36] | `zhang2000` | Z. Zhang, “A flexible new technique for camera calibration,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 22, no. 11, pp. 1330–1334, 2000. | https://doi.org/10.1109/34.888718 | Camera calibration (prerequisite for metric claims; Future Work) |
| [37] | `howard2019` | A. Howard et al., “Searching for MobileNetV3,” 2019 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 1314–1324, 2019. | https://doi.org/10.1109/iccv.2019.00140 | MobileNetV3 encoder |
| [38] | `milletari` | F. Milletari, N. Navab, and S. A. Ahmadi, “V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation,” 2016 Fourth International Conference on 3D Vision (3DV), pp. 565–571, 2016. | https://doi.org/10.1109/3dv.2016.79 | Dice loss |
| [39] | `lin2017` | T. Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, “Focal Loss for Dense Object Detection,” 2017 IEEE International Conference on Computer Vision (ICCV), pp. 2999–3007, 2017. | https://doi.org/10.1109/iccv.2017.324 | Focal loss |
| [40] | `hendrycks19` | D. Hendrycks and T. Dietterich, “Benchmarking Neural Network Robustness to Common Corruptions and Perturbations,” arXiv preprint arXiv:1903.12261, 2019. | https://doi.org/10.48550/arXiv.1903.12261 | Robustness to common corruptions |
| [41] | `omni2026` | A. Zacharia, M. Dharmadhikari, M. Singh, and K. Alexis, “OmniPlanner: Universal Exploration and Inspection Path Planning Across Robot Morphologies,” arXiv preprint arXiv:2603.04284, 2026. | https://doi.org/10.48550/arXiv.2603.04284 | Planning across robot morphologies (platform independence, Future Work) |
| [42] | `uav2023` | R. Zhang, G. Hao, K. Zhang, and Z. Li, “Unmanned aerial vehicle navigation in underground structure inspection: A review,” Geological Journal, vol. 58, no. 6, pp. 2454–2472, 2023. | https://doi.org/10.1002/gj.4763 | UAV navigation in underground structure inspection (Future Work) |

**Cited in the paper:** 42 of 42 verified. None are left uncited.

**Candidates that failed verification and must not be used:**

| DOI tried | What the registry returned |
|---|---|
| `10.1016/j.isprsjprs.2018.07.003` | A rubber-plantation stand-age paper (wrong suffix; the tunnel review is `.07.010`) |
| `10.1109/ICRA.2017.7989443` | *PennCOSYVIO*, a visual-inertial odometry benchmark, not exposure control |
| `10.1109/IROS.2004.1389727` | Gazebo paper, valid record but no date field; dropped when the user removed simulation from the paper |
