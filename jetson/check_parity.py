#!/usr/bin/env python3
"""TensorRT vs ONNX Runtime parity check (setup.md 5).

Pass criterion: FP16 mask disagreement below ~0.1 % of pixels at T=0.55.
Also reports how much of each image is above threshold, so a trivially-empty
sample set (which would pass vacuously) is visible rather than hidden.
"""
import argparse, glob, os
import cv2, numpy as np, onnxruntime as ort
from trt_infer import CrackNetTRT

ap = argparse.ArgumentParser()
ap.add_argument("--engine", default="cracknet_fp16.engine")
ap.add_argument("--onnx", default="cracknet_sim.onnx")
ap.add_argument("--dir", default="val_samples")
ap.add_argument("--thresh", type=float, default=0.55)
ap.add_argument("--n", type=int, default=20)
args = ap.parse_args()

T = args.thresh
paths = sorted(glob.glob(os.path.join(args.dir, "*.jpg")) +
               glob.glob(os.path.join(args.dir, "*.png")))[:args.n]
if not paths:
    raise SystemExit(f"no images in {args.dir}/")

trt_net = CrackNetTRT(args.engine)
S = trt_net.size
sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

maxdiff, dis, tot, pos = 0.0, 0, 0, 0
for p in paths:
    img = cv2.imread(p)
    if img is None:
        continue
    rgb = cv2.cvtColor(cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2RGB)
    x = CrackNetTRT.preprocess(rgb)
    a = 1 / (1 + np.exp(-sess.run(None, {"input": x})[0][0, 0]))   # ONNX ref
    b = trt_net.infer(x)                                           # TensorRT
    maxdiff = max(maxdiff, float(np.abs(a - b).max()))
    dis += int(((a > T) != (b > T)).sum())
    pos += int((a > T).sum())
    tot += a.size

print(f"engine            {args.engine}  (input {S}x{S})")
print(f"images            {len(paths)} from {args.dir}/")
print(f"max prob diff     {maxdiff:.2e}")
print(f"mask disagreement {dis}/{tot} = {dis/tot:.6f}  ({100*dis/tot:.4f} %)")
print(f"ONNX pixels > {T}  {pos}/{tot} = {100*pos/tot:.3f} %   "
      f"<- if ~0, the sample set exercises the model weakly")
print("PASS" if dis / tot < 0.001 else "FAIL (>0.1 % disagreement)")
