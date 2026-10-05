#!/usr/bin/env python3
"""Measure the real cost of the tiled scan path, without a camera.

docs/multicam-plan.md §4A.3 estimates 0.39 s of inference per scan position at
1080p by multiplying a single-tile figure by 45. That ignores preprocessing, the
host/device copies and the sigmoid, so this runs the actual path end to end on a
synthetic frame and reports where the time goes.

Everything here is real except the camera: real engine, real preprocessing, real
tiling and stitching.

    python3 bench_tiling.py                 # both resolutions, 3 cameras
    python3 bench_tiling.py --cameras 1
"""
import argparse
import time

import cv2
import numpy as np

from tiler import plan_tiles, extract, stitch_masks
from trt_infer import CrackNetTRT, logit



def clocks_locked():
    """jetson_clocks does NOT survive a reboot, and an unlocked GPU idles at
    306 MHz of 1020. Benchmarking in that state understates throughput by 2.6x -
    this script measured 2040 ms per scan position unlocked and 777 ms locked.
    Any timing taken without checking is meaningless, so check.
    """
    base = "/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/"
    try:
        cur = int(open(base + "cur_freq").read())
        mx = int(open(base + "max_freq").read())
        return cur >= 0.9 * mx, cur / 1e6, mx / 1e6
    except OSError:
        return None, 0, 0


def synthetic_frame(w, h, seed=0):
    """Textured frame with crack-like filaments - not flat, so the model has
    something to respond to and the timing is not measured on a trivial input."""
    rng = np.random.default_rng(seed)
    base = rng.integers(90, 150, (h, w), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 3)
    for i in range(6):
        pts = rng.integers(0, [w, h], size=(5, 2)).astype(np.int32)
        cv2.polylines(base, [pts], False, int(rng.integers(30, 70)),
                      int(rng.integers(1, 4)))
    return cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)


def run(net, frame, tiles, thresh):
    """One camera's worth: tile -> preprocess -> infer -> threshold -> stitch."""
    t = {}
    t0 = time.perf_counter()
    crops = extract(frame, tiles)
    t["tile"] = time.perf_counter() - t0

    t_pre = t_inf = 0.0
    masks = []
    for c in crops:
        a = time.perf_counter()
        rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
        x = CrackNetTRT.preprocess(rgb)
        b = time.perf_counter()
        lg = net.infer_logits(x)
        c2 = time.perf_counter()
        t_pre += b - a
        t_inf += c2 - b
        masks.append((lg > thresh).astype(np.uint8) * 255)
    t["preprocess"], t["infer"] = t_pre, t_inf

    t0 = time.perf_counter()
    full = stitch_masks(masks, tiles, frame.shape[1], frame.shape[0])
    t["stitch"] = time.perf_counter() - t0
    return t, full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="cracknet_fp16.engine")
    ap.add_argument("--cameras", type=int, default=3)
    ap.add_argument("--thresh", type=float, default=0.55)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    ok, cur, mx = clocks_locked()
    if ok is False:
        print(f"!! GPU at {cur:.0f} MHz of {mx:.0f} - clocks NOT locked.")
        print("!! Timings understate throughput by ~2.6x. Run: sudo jetson_clocks")
    elif ok:
        print(f"GPU clocks locked at {cur:.0f} MHz")

    net = CrackNetTRT(args.engine)
    thr = logit(args.thresh)
    print(f"engine {args.engine}, model input {net.size}x{net.size}, "
          f"{args.cameras} camera(s)\n")

    for w, h in ((1280, 720), (1920, 1080)):
        tiles = plan_tiles(w, h, tile=net.size)
        frame = synthetic_frame(w, h)
        run(net, frame, tiles, thr)                  # warm up

        best = None
        for _ in range(args.repeat):
            t, full = run(net, frame, tiles, thr)
            total = sum(t.values())
            if best is None or total < sum(best.values()):
                best, mask = t, full

        per_cam = sum(best.values())
        print(f"{w}x{h}  {len(tiles)} tiles/camera")
        for k in ("tile", "preprocess", "infer", "stitch"):
            print(f"    {k:<12} {best[k]*1000:8.1f} ms"
                  f"   {100*best[k]/per_cam:5.1f} %")
        print(f"    {'per camera':<12} {per_cam*1000:8.1f} ms")
        print(f"    {'x' + str(args.cameras) + ' cameras':<12} "
              f"{per_cam*args.cameras*1000:8.1f} ms"
              f"   <- inference cost of one scan position")
        print(f"    mask coverage {100*(mask > 0).mean():.2f} % "
              f"(synthetic input, not a real measurement of accuracy)\n")


if __name__ == "__main__":
    main()
