#!/usr/bin/env python3
"""Live crack segmentation: C920 -> TensorRT CrackNet -> overlay."""
import argparse, os, subprocess, time
import cv2
import numpy as np
from camera_ctl import AutoExposure
from trt_infer import CrackNetTRT

SETUP_SH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_setup.sh")


def lock_camera_controls(dev=0):
    """Re-apply the 4.2 controls AFTER streaming starts.

    Starting the MJPG stream resets exposure_time_absolute back towards its
    default (observed 156 -> 312), which lengthens exposure and reintroduces
    the motion blur 4.2 exists to prevent. Focus survives; exposure does not.
    """
    if not os.path.exists(SETUP_SH):
        return
    subprocess.run(["bash", SETUP_SH, f"/dev/video{dev}"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def open_c920(dev=0, w=1280, h=720, fps=30):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # must come first
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always use the newest frame
    if not cap.isOpened():
        raise RuntimeError(f"cannot open /dev/video{dev}")
    for _ in range(5):          # get the stream actually running first
        cap.read()
    lock_camera_controls(dev)
    return cap


def center_square(frame):
    """Crop the central HxH square so the model sees undistorted geometry."""
    h, w = frame.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return frame[y0:y0 + s, x0:x0 + s], (x0, y0, s)


def clean_mask(mask, min_area, max_area_frac=0.20, max_halfwidth=12.0,
               max_solidity=0.80, shape_filter=True):
    """Drop components that are too small, too big, or the wrong shape.

    Area alone cannot tell "one huge crack" from "one huge blob", which is why
    a person or a curtain can register 19 % coverage - three times denser than
    the ~6.3 % of pixels that are crack in an average CRACK500 mask. Cracks are
    thin branching filaments, so two cheap geometric tests separate them from
    the fat compact regions that out-of-domain objects produce:

      mean half-width = area / perimeter
          A stroke of width w has area/perimeter ~ w/2, so this is a direct
          read of how thick the component is, in pixels.
      solidity = area / convex-hull area
          A meandering filament fills very little of its hull; a blob fills
          most of it. Only applied to large components, since a short stubby
          but genuine crack can be perfectly solid.

    Area comes from the connected-component pixel count, never cv2.contourArea,
    which collapses to ~0 for a 1-px-wide line and would reject real cracks.
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    total = mask.size
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        if shape_filter:
            if area > max_area_frac * total:
                continue
            x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            sub = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
            cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
            if not cnts:
                continue
            perim = sum(cv2.arcLength(c, True) for c in cnts)
            if area / max(perim, 1.0) > max_halfwidth:
                continue                        # too thick to be a crack
            if area > 0.02 * total:
                hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
                hull_area = cv2.contourArea(hull)
                if hull_area > 0 and area / hull_area > max_solidity:
                    continue                    # large and compact -> blob
        keep[lab == i] = 255
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="cracknet_fp16.engine")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--thresh", type=float, default=0.55)
    ap.add_argument("--min-area", type=int, default=300,
                    help="px at model resolution; METU false blobs median ~165px")
    ap.add_argument("--alert-frac", type=float, default=0.01,
                    help="crack pixel fraction to flag a frame as CRACK. "
                         "setup.md's 0.002 fires on the ~0.06%% speck level a "
                         "clean surface produces; real cracks average 6.3%%")
    ap.add_argument("--max-area-frac", type=float, default=0.20,
                    help="reject any single component larger than this fraction")
    ap.add_argument("--max-halfwidth", type=float, default=12.0,
                    help="reject components thicker than this (area/perimeter, px). "
                         "Lower rejects more false positives but starts dropping "
                         "genuinely wide cracks")
    ap.add_argument("--max-solidity", type=float, default=0.80,
                    help="reject large components that fill their convex hull")
    ap.add_argument("--no-shape-filter", action="store_true",
                    help="area-filter only, as in setup.md")
    ap.add_argument("--no-auto-exposure", action="store_true",
                    help="keep the fixed exposure from camera_setup.sh")
    ap.add_argument("--ae-target", type=float, default=120.0,
                    help="target median luma; mid-grey keeps the inspected "
                         "surface off both the clipping and the noise floor")
    ap.add_argument("--gain", type=int, default=0,
                    help="analog gain; high gain turns sensor noise into "
                         "crack-like streaks")
    ap.add_argument("--record", default="", help="optional output .mp4")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    net = CrackNetTRT(args.engine)
    S = net.size
    cap = open_c920(args.cam)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("no frame from camera")
    print("camera:", frame.shape, "model input:", S)

    ae = None
    if not args.no_auto_exposure:
        ae = AutoExposure(cap, target=args.ae_target, gain=args.gain)
        for _ in range(12):             # let the loop settle before inferring
            ok, frame = cap.read()
            if ok:
                ae.update(frame, force=True)
        print("auto-exposure settled:", ae.label())

    writer = None
    if args.record:
        crop, _ = center_square(frame)
        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                 15, (crop.shape[1], crop.shape[0]))

    t_prev, fps, smooth = time.time(), 0.0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        crop, _ = center_square(frame)                    # 720x720 BGR
        if ae:
            ae.update(crop)
        rgb = cv2.cvtColor(cv2.resize(crop, (S, S), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB)

        t0 = time.time()
        prob = net.infer(CrackNetTRT.preprocess(rgb))     # SxS
        t_inf = (time.time() - t0) * 1000

        mask = (prob > args.thresh).astype(np.uint8) * 255
        mask = clean_mask(mask, args.min_area,
                          max_area_frac=args.max_area_frac,
                          max_halfwidth=args.max_halfwidth,
                          max_solidity=args.max_solidity,
                          shape_filter=not args.no_shape_filter)
        frac = float((mask > 0).mean())
        smooth = 0.7 * smooth + 0.3 * frac                # temporal smoothing
        is_crack = smooth > args.alert_frac

        # overlay at camera resolution
        big = cv2.resize(mask, (crop.shape[1], crop.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
        vis = crop.copy()
        vis[big > 0] = (0.4 * vis[big > 0] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
        contours, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 255), 1)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6)); t_prev = now
        label = f"{'CRACK' if is_crack else 'clear'}  cov={100*smooth:.2f}%  " \
                f"inf={t_inf:.1f}ms  fps={fps:.1f}"
        cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if is_crack else (0, 200, 0), 2)
        if ae:
            cv2.putText(vis, ae.label(), (10, 58), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 0), 1)
            label = label + "  " + ae.label()

        if writer:
            writer.write(vis)
        if args.headless:
            print(label, end="\r")
        else:
            cv2.imshow("CrackNet", vis)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
