#!/usr/bin/env python3
"""Generate every figure in the paper.

Provenance of each figure's data is stated in its docstring, because a figure
that looks measured but is illustrative is worse than no figure:

    DATA     read from crack_outputs/ (training log, sweeps, METU blob sizes)
    CODE     computed by running the project's own measure/verify/tiler code
    TABLE    a measured value recorded in docs/ from an on-device run
    DERIVED  analytic, from the GSD formula and the measured field of view
    SCHEMA   a diagram; carries no data

    python docs/documentation/figures/make_figures.py
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon, Circle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = HERE                                  # PNGs land beside this script
CODE = os.environ.get("CRACKNET_CODE", os.path.join(HERE, "_code"))


def _bootstrap_code():
    """Fetch the shipped modules the figures are computed from.

    verify.py and the 23-check tiler only exist on the accuracy-filters branch,
    so they are read out of git rather than assumed to be in the working tree.
    """
    import subprocess
    os.makedirs(CODE, exist_ok=True)
    for name in ("tiler", "measure", "verify", "test_measure", "test_verify"):
        dst = os.path.join(CODE, name + ".py")
        if os.path.exists(dst):
            continue
        out = subprocess.run(
            ["git", "show", "accuracy-filters:jetson/%s.py" % name],
            cwd=ROOT, capture_output=True)
        if out.returncode != 0:
            sys.exit("cannot read accuracy-filters:jetson/%s.py - is the branch "
                     "fetched? (git fetch origin accuracy-filters)" % name)
        with open(dst, "wb") as f:
            f.write(out.stdout)


_bootstrap_code()
sys.path.insert(0, CODE)

W1, W2 = 3.5, 7.16           # IEEE column and page width, inches
BLUE, DBLUE = "#1F5C99", "#0B2F57"
ORANGE, RED, GREEN, GREY = "#D9822B", "#B03A2E", "#2E7D4F", "#6B7280"
LIGHT = "#E8F0F8"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.linewidth": 0.6, "lines.linewidth": 1.1, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
    "mathtext.fontset": "stix",
})


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("  wrote", name)


def box(ax, x, y, w, h, text, fc=LIGHT, ec=BLUE, fs=7, bold=False, tc="black", lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.04",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc, linespacing=1.15)


def arrow(ax, p0, p1, color=DBLUE, lw=0.9, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=7,
                                 color=color, lw=lw, linestyle=ls,
                                 shrinkA=0, shrinkB=0))


# --------------------------------------------------------------------------- #
# SCHEMA - system architecture
# --------------------------------------------------------------------------- #
def fig_arch():
    fig, ax = plt.subplots(figsize=(W2, 3.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")

    ax.add_patch(Rectangle((1, 2), 71, 41, fc="#F7F9FC", ec=GREY, lw=0.7, ls="--"))
    ax.text(2.2, 40.2, "Jetson Orin Nano 8 GB   (MAXN_SUPER, GPU 1020 MHz)", fontsize=7.2,
            color=DBLUE, fontweight="bold")
    ax.add_patch(Rectangle((82, 2), 17, 41, fc="#FBF7F1", ec=GREY, lw=0.7, ls="--"))
    ax.text(90.5, 40.2, "Operator PC", fontsize=7.2, color=DBLUE, fontweight="bold", ha="center")

    # three cameras, stacked
    ys = (29.0, 20.0, 11.0)
    for y, nm in zip(ys, ("LEFT", "TOP", "RIGHT")):
        box(ax, 3, y, 9.5, 6.6, "C920  %s\n1080p MJPG" % nm, fc="#FFFFFF", fs=6.2)
    ax.text(7.75, 7.0, "one USB 2.0 hub\n480 Mbit/s shared", ha="center", va="center",
            fontsize=5.9, color=GREY)

    box(ax, 16.5, 17.5, 12.5, 14.0,
        "Sequential\ncapture\n\nopen → settle\n→ average →\nrelease", fs=6.3)
    box(ax, 16.5, 5.0, 12.5, 7.0, "Exposure control\n(1-stop ladder)", fs=6.1, fc="#FFFFFF")
    box(ax, 33.5, 17.5, 11.5, 14.0,
        "Native tiler\n\n512×512\n15% overlap\n15 tiles/frame", fs=6.3)
    box(ax, 49.5, 17.5, 11.5, 14.0,
        "TensorRT\nFP16 engine\n\nU-Net,\nMobileNetV3-L\n8.77 ms/tile", fs=6.1, fc="#DCEBF7")

    # right-hand chain, top to bottom
    cx, cw, ch = 62.2, 9.4, 7.6
    box(ax, cx, 31.5, cw, ch, "Blend\nlogits", fs=6.2)
    box(ax, cx, 22.0, cw, ch, "Shape, valley,\nstraightness\nfilters", fs=5.4)
    box(ax, cx, 12.5, cw, ch, "Measure\nwidth, length\n(mm)", fs=5.6, fc="#E6F2E9", ec=GREEN)
    box(ax, cx, 3.0, cw, 7.0, "Scan loop\nHTTP :8081", fs=5.6, fc="#FFFFFF", ec=ORANGE)

    for i, y in enumerate(ys):
        arrow(ax, (12.5, y + 3.3), (16.5, 24.5 + (1 - i) * 3.0))
    arrow(ax, (29.0, 24.5), (33.5, 24.5))
    arrow(ax, (45.0, 24.5), (49.5, 24.5))
    arrow(ax, (61.0, 27.0), (62.2, 33.5))
    arrow(ax, (66.9, 31.5), (66.9, 29.6))
    arrow(ax, (66.9, 22.0), (66.9, 20.1))
    arrow(ax, (66.9, 12.5), (66.9, 10.0))
    arrow(ax, (22.75, 12.0), (22.75, 17.5), ls=":")

    # operator side
    box(ax, 84, 22, 14, 9, "Dashboard\n(browser)\nMJPEG · JSON", fs=6.4, fc="#FFFFFF", ec=ORANGE)
    box(ax, 84, 8, 14, 9, "cracknet.bat\nstart · stop\nstatus · logs", fs=6.2, fc="#FFFFFF", ec=ORANGE)
    arrow(ax, (71.6, 6.5), (76.5, 6.5), color=ORANGE, style="-")
    arrow(ax, (76.5, 6.5), (76.5, 26.5), color=ORANGE, style="-")
    arrow(ax, (76.5, 26.5), (84.0, 26.5), color=ORANGE)
    arrow(ax, (84.0, 12.5), (72.6, 12.5), color=ORANGE, ls="--")
    ax.text(77.2, 18.5, "Wi-Fi\nAP", fontsize=6.0, color=ORANGE, ha="left", va="center")
    ax.text(77.2, 14.0, "SSH", fontsize=6.0, color=ORANGE, ha="left")
    save(fig, "fig_arch.png")


# --------------------------------------------------------------------------- #
# SCHEMA - rig geometry
# --------------------------------------------------------------------------- #
def fig_geometry():
    """Field-of-view cones are ray-cast against the lining, so they stop at the
    wall, crown and floor instead of leaking through them."""
    HALF_W, SPRING, RISE = 4.6, 3.2, 3.4

    def inside(x, y):
        if y < 0 or abs(x) > HALF_W:
            return False
        return y <= SPRING or (x / HALF_W) ** 2 + ((y - SPRING) / RISE) ** 2 <= 1.0

    def cone(cx, cy, axis_deg, half_deg):
        pts = [(cx, cy)]
        for a in np.radians(np.linspace(axis_deg - half_deg, axis_deg + half_deg, 60)):
            r = 0.0
            while inside(cx + r * np.cos(a), cy + r * np.sin(a)) and r < 12:
                r += 0.01
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
        return pts

    fig, ax = plt.subplots(figsize=(W1, 2.95))
    ax.set_xlim(-9.6, 9.6); ax.set_ylim(-1.4, 8.4); ax.set_aspect("equal"); ax.axis("off")
    th = np.linspace(0, np.pi, 100)
    ax.plot(np.r_[-HALF_W, -HALF_W, HALF_W * np.cos(th), HALF_W, HALF_W],
            np.r_[0, SPRING, SPRING + RISE * np.sin(th), SPRING, 0], color=DBLUE, lw=1.6)
    ax.plot([-HALF_W, HALF_W], [0, 0], color=GREY, lw=1.2)

    half = 70.42 / 2
    cams = {"LEFT": (-0.55, 0.95, 180.0), "TOP": (0.0, 1.05, 90.0), "RIGHT": (0.55, 0.95, 0.0)}
    for nm, (cx, cy, ang) in cams.items():
        ax.add_patch(Polygon(cone(cx, cy, ang, half), closed=True, fc=ORANGE,
                             alpha=0.20, ec=ORANGE, lw=0.6))
    ax.add_patch(Rectangle((-0.8, 0.02), 1.6, 0.62, fc=BLUE, ec=DBLUE, zorder=4))
    for xc in (-0.5, 0.5):
        ax.add_patch(Circle((xc, 0.03), 0.13, fc="white", ec=DBLUE, zorder=5))
    for nm, (cx, cy, ang) in cams.items():
        ax.plot(cx, cy - 0.05, "s", color=DBLUE, ms=3.2, zorder=6)

    ax.text(-7.7, 3.0, "LEFT\nwall", ha="center", fontsize=7, color=DBLUE, fontweight="bold")
    ax.text(7.7, 3.0, "RIGHT\nwall", ha="center", fontsize=7, color=DBLUE, fontweight="bold")
    ax.text(0, 7.55, "TOP  (crown)", ha="center", fontsize=7, color=DBLUE, fontweight="bold")
    ax.text(0, -0.95, "inspection carriage", ha="center", fontsize=6.6, color=GREY)
    ax.annotate("", xy=(-HALF_W, 0.3), xytext=(-0.8, 0.3),
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=0.7))
    ax.text(-2.7, 0.42, "standoff $d$", ha="center", fontsize=6.4, color=GREY)
    ax.text(0, 4.9, "HFOV 70.42°\nper camera", ha="center", fontsize=6.4, color="#9A5A12")
    save(fig, "fig_geometry.png")


# --------------------------------------------------------------------------- #
# SCHEMA + TABLE - perception pipeline with measured per-stage cost
# --------------------------------------------------------------------------- #
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(W1, 4.7))
    ax.set_xlim(0, 20); ax.set_ylim(0, 46); ax.axis("off")
    stages = [
        ("Frame  1920×1080 BGR", "average of 4 independent frames", "capture 1.98 s / camera", LIGHT),
        ("Tile  512×512 at native scale", "15 tiles, 15% overlap, edges clamped", "0.6–1.7 ms", LIGHT),
        ("Preprocess", "BGR→RGB, ImageNet mean/std, NCHW", "89.0 ms", LIGHT),
        ("TensorRT FP16 inference", "logits, no sigmoid (logit-space threshold)", "130.3 ms", "#DCEBF7"),
        ("Blend overlapping logits", "raised-cosine weights, threshold once", "per frame", LIGHT),
        ("Shape filter", "area/perimeter, solidity, area floor", "per component", "#FBEFE0"),
        ("Valley + straightness", "reject steps and ruled lines", "per component", "#FBEFE0"),
        ("Measure in millimetres", "GSD × distance-transform ridge p95", "per component", "#E6F2E9"),
    ]
    y = 44.0
    for i, (t, s, c, fc) in enumerate(stages):
        box(ax, 0.3, y - 4.4, 13.2, 4.4, "", fc=fc, ec=BLUE, lw=0.7)
        ax.text(0.9, y - 1.55, t, fontsize=7.0, fontweight="bold", va="center")
        ax.text(0.9, y - 3.2, s, fontsize=6.1, color="#333333", va="center")
        ax.text(19.8, y - 2.2, c, fontsize=6.3, ha="right", va="center", color=DBLUE)
        if i < len(stages) - 1:
            arrow(ax, (6.9, y - 4.4), (6.9, y - 5.1), lw=0.8)
        y -= 5.5
    save(fig, "fig_pipeline.png")


# --------------------------------------------------------------------------- #
# TABLE - capture timeline for one scan position
# --------------------------------------------------------------------------- #
def fig_capture_timeline():
    """docs/phase2/01-camera-weaving.md: per-camera open 0.99, confirm 0.53,
    grab+average 0.22, release 0.24 (x3 = 5.93 s), then inference 0.22 x3."""
    fig, ax = plt.subplots(figsize=(W2, 1.75))
    seg = [("open + STREAMON", 0.99, BLUE), ("exposure confirm", 0.53, ORANGE),
           ("grab + average 4", 0.22, GREEN), ("release", 0.24, GREY)]
    t = 0.0
    for cam in ("LEFT", "TOP", "RIGHT"):
        for j, (nm, d, c) in enumerate(seg):
            ax.barh(0, d, left=t, color=c, edgecolor="white", lw=0.5, height=0.55,
                    label=nm if cam == "LEFT" else None)
            t += d
        ax.text(t - 0.99, 0.43, cam, fontsize=6.6, ha="center", color=DBLUE, fontweight="bold")
    cap_end = t
    for k in range(3):
        ax.barh(0, 0.22, left=t, color=RED, edgecolor="white", lw=0.5, height=0.55,
                label="inference (15 tiles)" if k == 0 else None)
        t += 0.22
    ax.axvline(cap_end, color="k", ls=":", lw=0.7)
    ax.text(cap_end / 2, -0.62, "capture ≈ %.1f s  (%.0f%% of the cycle)" % (cap_end, 100 * cap_end / t),
            ha="center", fontsize=7)
    ax.text(cap_end + (t - cap_end) / 2, -0.62, "infer ≈ %.1f s" % (t - cap_end), ha="center", fontsize=7)
    ax.set_xlim(0, t + 0.1); ax.set_ylim(-0.9, 0.85)
    ax.set_yticks([]); ax.set_xlabel("time within one scan position (s)")
    ax.spines["left"].set_visible(False)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.28), frameon=False)
    save(fig, "fig_capture_timeline.png")


# --------------------------------------------------------------------------- #
# TABLE - exposure ladder
# --------------------------------------------------------------------------- #
def fig_exposure_ladder():
    """Measured: while streaming at 1080p only 38,77,156,312,624 are applied.
    Plotted with nearest-rung snapping in log space, as camera_ctl.rung_index."""
    rungs = np.array([38, 77, 156, 312, 624])
    req = np.arange(3, 1000)
    idx = np.abs(np.log(req[:, None] / rungs[None, :])).argmin(axis=1)
    fig, ax = plt.subplots(figsize=(W1, 2.35))
    ax.plot(req, req, color=GREY, lw=0.8, ls="--", label="requested (idle driver accepts all 747 values)")
    ax.step(req, rungs[idx], where="mid", color=BLUE, lw=1.4, label="applied while streaming")
    ax.plot(rungs, rungs, "o", color=RED, ms=3.5)
    for r in rungs:
        ax.annotate(str(r), (r, r), textcoords="offset points",
                    xytext=(7, 3) if r == 38 else (5, -11), fontsize=6.3, color=RED)
    ax.set_xlabel("requested exposure_time_absolute (100 µs units)")
    ax.set_ylabel("applied exposure")
    ax.set_xlim(0, 800); ax.set_ylim(-15, 700)
    ax.legend(loc="upper left", frameon=False)
    save(fig, "fig_exposure_ladder.png")


# --------------------------------------------------------------------------- #
# DERIVED - USB bandwidth
# --------------------------------------------------------------------------- #
def fig_usb():
    yuyv = 1280 * 720 * 16 * 30 / 1e6          # 442.4 Mbit/s
    fig, ax = plt.subplots(figsize=(W1, 1.85))
    ax.barh(["MJPG 720p30", "YUYV 720p30"], [45, yuyv], color=[GREEN, RED], height=0.5)
    ax.axvline(320, color="k", ls="--", lw=0.8)
    ax.text(324, 0.58, "practical isochronous\nvideo payload ≈ 320", fontsize=6.3, va="center")
    ax.axvline(480, color=GREY, ls=":", lw=0.8)
    ax.text(478, 0.2, "USB 2.0\nsignalling 480", fontsize=6.3, ha="right", va="center", color=GREY)
    ax.text(45 + 6, 0, "≈ 45", va="center", fontsize=7)
    ax.text(yuyv - 8, 1, "%.0f" % yuyv, va="center", ha="right", fontsize=7, color="white")
    ax.set_xlim(0, 500); ax.set_xlabel("required bandwidth (Mbit/s)")
    save(fig, "fig_usb_bw.png")


# --------------------------------------------------------------------------- #
# CODE - real tile plan
# --------------------------------------------------------------------------- #
def fig_tiles():
    import tiler
    tiles = tiler.plan_tiles(1920, 1080)
    fig, ax = plt.subplots(figsize=(W1, 2.15))
    ax.add_patch(Rectangle((0, 0), 1920, 1080, fc="#F4F6F9", ec=DBLUE, lw=1.0))
    cols = [BLUE, ORANGE, GREEN]
    for i, (x, y) in enumerate(tiles):
        ax.add_patch(Rectangle((x, y), 512, 512, fc=cols[i % 3], alpha=0.10, ec=cols[i % 3], lw=0.6))
        ax.text(x + 256, y + 256, str(i + 1), ha="center", va="center", fontsize=6.2, color=DBLUE)
    ax.set_xlim(-30, 1950); ax.set_ylim(1110, -30); ax.set_aspect("equal")
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    ax.set_title("%d tiles of 512×512 over a 1920×1080 frame (tiler.plan_tiles)" % len(tiles), fontsize=7)
    save(fig, "fig_tiles.png")


# --------------------------------------------------------------------------- #
# CODE - blending removes a seam artefact the union keeps
# --------------------------------------------------------------------------- #
def fig_blend():
    import tiler
    w, h = 768, 512
    tiles = tiler.plan_tiles(w, h)
    lg = []
    for (x, y) in tiles:
        t = np.full((512, 512), -4.0, np.float32)
        # each tile hallucinates a response on the border that faces its neighbour
        if x == 0:
            t[:, 509:512] = 6.0
        else:
            t[:, 0:3] = 6.0
        lg.append(t)
    blended, _ = tiler.blend_logits(lg, tiles, w, h)
    union = np.zeros((h, w), bool)
    for (x, y), t in zip(tiles, lg):
        union[y:y + 512, x:x + 512] |= (t > 0)
    row = 256
    xs = np.arange(w)
    win = tiler.tile_window(512, 0.25)

    fig, axs = plt.subplots(2, 1, figsize=(W1, 3.3), sharex=False,
                            gridspec_kw={"height_ratios": [1, 1.35]})
    axs[0].plot(np.arange(512), win[256, :], color=BLUE)
    axs[0].set_ylabel("weight"); axs[0].set_xlabel("pixel within tile")
    axs[0].set_title("raised-cosine tile window (taper 0.25)", fontsize=7)
    for (x0, _), t, c, nm in zip(tiles, lg, (ORANGE, GREEN), ("tile 1 own logit", "tile 2 own logit")):
        axs[1].plot(np.arange(x0, x0 + 512), t[row], color=c, lw=0.9, label=nm)
    axs[1].fill_between(xs, -5, np.where(union[row], 7, -5), step="mid", color=RED, alpha=0.30,
                        label="union keeps these pixels")
    axs[1].plot(xs, blended[row], color=DBLUE, lw=1.6, label="blended logit")
    axs[1].axhline(0, color=GREY, lw=0.6, ls=":")
    axs[1].set_ylim(-5, 7.5)
    axs[1].set_ylabel("logit"); axs[1].set_xlabel("x across the 768-px frame (px)")
    axs[1].legend(loc="upper center", frameon=False, ncol=2, fontsize=6.2,
                  bbox_to_anchor=(0.5, 1.02), columnspacing=0.9, handlelength=1.4)
    axs[1].set_title("border artefact: kept by union, removed by blending", fontsize=7)
    fig.tight_layout(h_pad=0.8)
    save(fig, "fig_blend.png")
    print("    union detections:", int(union.sum()), "| blended > 0:", int((blended > 0).sum()))


# --------------------------------------------------------------------------- #
# DATA - training curves
# --------------------------------------------------------------------------- #
def fig_training():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "crack_outputs", "train_log.csv"))))
    ep = np.array([int(r["epoch"]) for r in rows])
    loss = np.array([float(r["train_loss"]) for r in rows])
    iou = np.array([float(r["val_iou"]) for r in rows])
    dice = np.array([float(r["val_dice"]) for r in rows])
    best = int(iou.argmax())
    fig, ax = plt.subplots(figsize=(W1, 2.55))
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax.plot(ep, loss, color=GREY, label="train loss (Dice+focal)")
    ax2.plot(ep, iou, color=BLUE, label="val IoU")
    ax2.plot(ep, dice, color=ORANGE, label="val Dice")
    ax2.plot(best, iou[best], "o", color=RED, ms=4)
    ax2.annotate("best IoU %.3f\n(epoch %d)" % (iou[best], best), (best, iou[best]),
                 xytext=(best - 3, iou[best] - 0.30), fontsize=6.5, ha="center",
                 arrowprops=dict(arrowstyle="-", lw=0.5, color=RED))
    ax.set_xlabel("epoch"); ax.set_ylabel("training loss"); ax2.set_ylabel("validation score")
    ax.set_ylim(0, 1.05); ax2.set_ylim(0.15, 0.85)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", frameon=False, ncol=3,
              bbox_to_anchor=(0.5, -0.20), columnspacing=1.0, handlelength=1.4)
    save(fig, "fig_training.png")


# --------------------------------------------------------------------------- #
# DATA - threshold sweep
# --------------------------------------------------------------------------- #
def fig_sweep():
    v = np.load(os.path.join(ROOT, "crack_outputs", "val_sweep.npy"))
    t = np.load(os.path.join(ROOT, "crack_outputs", "test_sweep.npy"))
    thr = v[0]
    fig, axs = plt.subplots(1, 2, figsize=(W1, 2.3), sharey=True)
    for ax, d, nm in ((axs[0], v, "validation"), (axs[1], t, "test")):
        ax.plot(thr, d[1], color=BLUE, label="precision")
        ax.plot(thr, d[2], color=ORANGE, label="recall")
        ax.plot(thr, d[3], color=GREEN, label="Dice")
        ax.plot(thr, d[4], color=GREY, label="IoU")
        ax.axvline(0.55, color=RED, ls="--", lw=0.8)
        ax.set_title(nm, fontsize=7.5); ax.set_xlabel("threshold")
    axs[0].set_ylabel("score")
    for ax in axs:
        ax.text(0.57, 0.845, "T = 0.55", color=RED, fontsize=6.5)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03),
               columnspacing=1.0, handlelength=1.4)
    fig.tight_layout(w_pad=0.6, rect=(0, 0.07, 1, 1))
    save(fig, "fig_sweep.png")


# --------------------------------------------------------------------------- #
# DATA - METU negative blobs
# --------------------------------------------------------------------------- #
def fig_metu():
    a = np.load(os.path.join(ROOT, "crack_outputs", "metu_negative_pixels.npy")).astype(float)
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    bins = np.logspace(0, np.log10(a.max() * 1.05), 36)
    ax.hist(a, bins=bins, color=BLUE, edgecolor="white", lw=0.3)
    ax.set_xscale("log")
    ax.axvline(np.median(a), color=RED, ls="--", lw=0.9)
    ax.axvline(a.mean(), color=ORANGE, ls="--", lw=0.9)
    ax.text(np.median(a) * 0.9, ax.get_ylim()[1] * 0.92, "median %d px" % np.median(a),
            color=RED, fontsize=6.6, ha="right")
    ax.text(a.mean() * 1.1, ax.get_ylim()[1] * 0.78, "mean %d px" % a.mean(), color=ORANGE, fontsize=6.6)
    ax.set_xlabel("false-positive pixels per crack-free image (log scale)")
    ax.set_ylabel("images")
    ax.set_title("n = %d of 20 000 METU crack-free images fired at T = 0.55" % a.size, fontsize=7)
    save(fig, "fig_metu.png")


# --------------------------------------------------------------------------- #
# TABLE - latency
# --------------------------------------------------------------------------- #
def fig_latency():
    """bench_tiling.py: 720p 6 tiles = 35.4 + 52.1 + 0.6; 1080p 15 tiles = 89.0 + 130.3 + 1.7.
    trtexec compute: FP16 8.77 ms, FP32 18.44 ms."""
    fig, axs = plt.subplots(1, 2, figsize=(W1, 2.3), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axs[0]
    labs = ["720p\n6 tiles", "1080p\n15 tiles"]
    pre = np.array([35.4, 89.0]); inf = np.array([52.1, 130.3]); til = np.array([0.6, 1.7])
    ax.bar(labs, pre, color=ORANGE, label="preprocess")
    ax.bar(labs, inf, bottom=pre, color=BLUE, label="inference")
    ax.bar(labs, til, bottom=pre + inf, color=GREY, label="tile + stitch")
    for i, tot in enumerate(pre + inf + til):
        ax.text(i, tot + 4, "%.0f ms" % tot, ha="center", fontsize=7)
    ax.set_ylabel("ms per camera"); ax.set_ylim(0, 260)
    ax.legend(frameon=False, loc="upper left", fontsize=6.3)
    ax = axs[1]
    ax.bar(["FP32", "FP16"], [18.44, 8.77], color=[GREY, GREEN], width=0.6)
    for i, v in enumerate([18.44, 8.77]):
        ax.text(i, v + 0.5, "%.2f" % v, ha="center", fontsize=7)
    ax.set_ylabel("ms per 512² tile"); ax.set_ylim(0, 23)
    ax.set_title("FP16 is 2.1× faster", fontsize=7)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_latency.png")


# --------------------------------------------------------------------------- #
# DERIVED - resolution vs standoff
# --------------------------------------------------------------------------- #
def fig_resolution():
    hfov = np.radians(70.42)
    d = np.linspace(0.2, 6.0, 300)

    def gsd(px):
        return 2 * d * np.tan(hfov / 2) / px * 1000.0     # mm / px

    fig, ax = plt.subplots(figsize=(W1, 2.75))
    ax.plot(d, 2 * gsd(1920), color=BLUE, label="1080p native (1920 px): 2-px crack")
    ax.plot(d, 2 * gsd(1280), color=ORANGE, label="720p native (1280 px): 2-px crack")
    ax.axhline(0.3, color=RED, ls="--", lw=0.9)
    ax.text(5.98, 0.42, "0.3 mm working target", color=RED, ha="right", va="bottom", fontsize=6.6)
    for dm, lab, tx in ((0.4, "0.4 m: 0.59 mm\n(bench scan)", (0.9, 4.6)),
                        (4.5, "4.5 m: 6.6 mm\n(road-tunnel wall)", (2.9, 9.0))):
        y = 2 * 2 * dm * np.tan(hfov / 2) / 1920 * 1000
        ax.plot(dm, y, "o", color=DBLUE, ms=3.5, zorder=5)
        ax.annotate(lab, (dm, y), xytext=tx, fontsize=6.4, color=DBLUE, ha="left",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=DBLUE))
    ax.set_xlabel("standoff to surface (m)")
    ax.set_ylabel("smallest resolvable crack width (mm)")
    ax.set_xlim(0, 6); ax.set_ylim(0, 14)
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig_resolution.png")


# --------------------------------------------------------------------------- #
# CODE - width estimators on shapes of known width
# --------------------------------------------------------------------------- #
def fig_width():
    import measure
    import test_measure as tm
    cases = [("straight 3", tm.straight(3)), ("straight 9", tm.straight(9)),
             ("branched 3", tm.branched(3)), ("branched 9", tm.branched(9)),
             ("taper 3→15", tm.tapered(3, 15))]
    truth = [3, 9, 3, 9, 15]
    stats = {"ridge p95 (reported)": [], "ridge median": [], "ridge max": [], "area / perimeter": []}
    for _, m in cases:
        c = measure.components(m, 1.0, min_area=1)[0]
        stats["ridge p95 (reported)"].append(c["width_mm"])
        stats["ridge median"].append(c["width_median_mm"])
        stats["ridge max"].append(c["width_max_mm"])
        stats["area / perimeter"].append(c["width_mean_mm"])
    x = np.arange(len(cases)); n = len(stats); bw = 0.17
    fig, ax = plt.subplots(figsize=(W1, 2.55))
    cols = [BLUE, ORANGE, GREY, GREEN]
    for i, (k, v) in enumerate(stats.items()):
        ax.bar(x + (i - 1.5) * bw, v, bw, label=k, color=cols[i])
    ax.plot(x, truth, "k_", ms=14, mew=1.6, label="true width")
    ax.set_xticks(x); ax.set_xticklabels([c[0] for c in cases], fontsize=6.4)
    ax.set_ylabel("estimated width (px)")
    ax.legend(frameon=False, fontsize=6.0, loc="upper left")
    save(fig, "fig_width.png")
    return stats


# --------------------------------------------------------------------------- #
# CODE - valley vs step
# --------------------------------------------------------------------------- #
def fig_valley():
    import verify
    import test_verify as tv
    items = [("3-px dark line", tv.valley_image(3)), ("6-px dark line", tv.valley_image(6)),
             ("shallow crack", tv.valley_image(3, 25)), ("step, same contrast", tv.step_image(60)),
             ("step, high contrast", tv.step_image(110))]
    scores = []
    for _, (img, col) in items:
        rep = verify.report(tv.strip_mask(col), img, min_len_px=10)
        scores.append(rep[0]["valleyness"] if rep else 0.0)
    fig = plt.figure(figsize=(W2, 2.35))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.5, 1], hspace=0.45, wspace=0.18)
    for i, ((nm, (img, col)), s) in enumerate(zip(items, scores)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img[100:300, 100:300], cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(nm, fontsize=7, color=GREEN if i < 3 else RED)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(GREEN if i < 3 else RED)
        ax2 = fig.add_subplot(gs[1, i])
        ax2.plot(img[200, 140:260].astype(float), color=DBLUE, lw=0.9)
        ax2.set_ylim(10, 200); ax2.set_yticks([])
        ax2.set_xlabel("valley depth = %.2f" % s, fontsize=7)
        ax2.spines["left"].set_visible(False)
    save(fig, "fig_valley.png")
    return [(n, s) for (n, _), s in zip(items, scores)]


# --------------------------------------------------------------------------- #
# CODE - straightness
# --------------------------------------------------------------------------- #
def fig_straight():
    import verify
    import test_verify as tv
    img, col = tv.valley_image(3)
    amps = [0, 2, 4, 8, 16]
    sc = []
    for a in amps:
        m = tv.strip_mask(col) if a == 0 else tv.wavy_mask(a)
        sc.append(verify.report(m, img, min_len_px=10)[0]["straightness"])
    fig, ax = plt.subplots(figsize=(W1, 2.15))
    ax.plot(amps, sc, "o-", color=BLUE, ms=3.5)
    ax.axhline(1.12, color=RED, ls="--", lw=0.9)
    ax.text(16, 1.5, "reject below 1.12", color=RED, ha="right", fontsize=6.6)
    ax.set_yscale("log")
    for a, s in zip(amps, sc):
        ax.annotate("%.2f" % s, (a, s), textcoords="offset points",
                    xytext=(12, -9) if a == 0 else (0, 5), ha="center", fontsize=6.3)
    ax.set_xlabel("lateral wander amplitude (px)")
    ax.set_ylabel("straightness (1.0 = ruled line)")
    save(fig, "fig_straight.png")
    return list(zip(amps, sc))


# --------------------------------------------------------------------------- #
# TABLE - false-positive reduction
# --------------------------------------------------------------------------- #
def fig_fp():
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.0))
    ax = axs[0]
    ax.bar(["raw", "after shape\nfilter"], [2.18, 11.33], color=[GREY, RED], width=0.55)
    for i, v in enumerate([2.18, 11.33]):
        ax.text(i, v + 0.3, "%.2f×" % v, ha="center", fontsize=7)
    ax.set_ylabel("border / interior\ndetection-rate ratio"); ax.set_ylim(0, 13.5)
    ax.set_title("(a) tile-border artefacts", fontsize=7.2)
    ax = axs[1]
    ax.bar(["raw\nmodel", "shape-\nfiltered"], [38.0, 1.44], color=[GREY, GREEN], width=0.55)
    for i, v in enumerate([38.0, 1.44]):
        ax.text(i, v + 1, "%.2f %%" % v if v < 10 else "%.0f %%" % v, ha="center", fontsize=7)
    ax.set_ylabel("coverage (%)"); ax.set_ylim(0, 46)
    ax.set_title("(b) person in frame (51 comps.)", fontsize=7.2)
    ax = axs[2]
    ax.bar(["without\nvalley/straight", "with"], [18, 0], color=[RED, GREEN], width=0.55)
    ax.text(0, 18.6, "18", ha="center", fontsize=7); ax.text(1, 0.6, "0", ha="center", fontsize=7)
    ax.set_ylabel("false components"); ax.set_ylim(0, 22)
    ax.set_title("(c) curtains + cables", fontsize=7.2)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_fp.png")


# --------------------------------------------------------------------------- #
# SCHEMA - console layout
# --------------------------------------------------------------------------- #
def fig_console():
    fig, ax = plt.subplots(figsize=(W2, 3.15))
    ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")
    ax.add_patch(Rectangle((1, 1), 98, 54, fc="#FFFFFF", ec=DBLUE, lw=1.0))
    ax.add_patch(Rectangle((1, 49), 98, 6, fc=DBLUE, ec=DBLUE))
    ax.text(3, 52, "CrackNet · console", color="white", fontsize=8, fontweight="bold", va="center")
    box(ax, 22, 50.2, 11, 3.6, "CONNECTED", fc="#DFF1E5", ec=GREEN, fs=6.2)
    box(ax, 35, 50.2, 18, 3.6, "LIVE PREVIEW | SCAN", fc="#DCEBF7", ec=BLUE, fs=6.2)
    ax.text(97, 52, "scan <n> · infer <ms> · gpu <MHz>", color="white", fontsize=6.4, ha="right", va="center")
    # composite
    box(ax, 3, 27, 66, 20, "", fc="#F3F4F6", ec=GREY)
    for i, nm in enumerate(("LEFT", "TOP", "RIGHT")):
        box(ax, 4.2 + i * 21.4, 28.2, 20, 17.6, nm + "\n(red fill = detection,\nyellow = component outline)", fc="#E5E7EB", ec=GREY, fs=6.3)
    ax.text(36, 25.6, "DISPLAY COMPOSITE — detection is per-camera", fontsize=6.5, ha="center", color=RED, style="italic")
    # strip
    for i, nm in enumerate(("LEFT", "TOP", "RIGHT")):
        box(ax, 3 + i * 22.2, 11.2, 21.4, 11.6, "%s\nexp · gain · coverage %%" % nm, fc="#F9FAFB", ec=GREY, fs=6.3)
        ax.add_patch(Circle((5 + i * 22.2, 20.8), 0.8, fc=GREEN, ec=GREEN))
    # log
    box(ax, 3, 2.2, 66, 7.6, "EVENT LOG   [pause] [clear]\n<time> INFO  scan clear   |   <time> CRACK  <camera>  <width> mm  <coverage> %  <n> comp.", fc="#FAFAFA", ec=GREY, fs=6.0)
    # side panels
    panels = [("VERDICT", "CRACK / CLEAR\nn components above threshold", 37.8),
              ("MEASUREMENTS", "coverage · widest crack (mm)\ncomponents · swath · resolution", 27.6),
              ("PIPELINE", "tiles/camera · inference\nscan cycle or fps · threshold", 17.4),
              ("SYSTEM", "power mode · gpu clock\nsoc temp · uptime", 7.2)]
    for t, s_, y in panels:
        ax.add_patch(Rectangle((72, y), 26, 9.0, fc="#F7F9FC", ec=GREY, lw=0.6))
        ax.text(73, y + 7.3, t, fontsize=6.8, fontweight="bold", color=DBLUE)
        ax.text(73, y + 3.4, s_, fontsize=6.0, va="center")
    save(fig, "fig_console.png")


def main():
    print("figures ->", OUT)
    fig_arch(); fig_geometry(); fig_pipeline(); fig_capture_timeline()
    fig_exposure_ladder(); fig_usb(); fig_tiles(); fig_blend()
    fig_training(); fig_sweep(); fig_metu(); fig_latency(); fig_resolution()
    w = fig_width(); v = fig_valley(); s = fig_straight(); fig_fp(); fig_console()
    json.dump({"width": w, "valley": v, "straightness": s},
              open(os.path.join(OUT, "computed_values.json"), "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
