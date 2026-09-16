#!/usr/bin/env bash
# setup.md 3.1 / 3.2 - build the TensorRT engines. Run on the Jetson.
set -e
cd "$(dirname "$0")"
TRTEXEC=/usr/src/tensorrt/bin/trtexec

echo "=== FP16 engine ==="
"$TRTEXEC" \
  --onnx=cracknet_sim.onnx \
  --saveEngine=cracknet_fp16.engine \
  --fp16 \
  --memPoolSize=workspace:2048 \
  --verbose 2>&1 | tee build_fp16.log | grep -E 'GPU Compute Time|Throughput|Latency|error|Error' || true

echo "=== FP32 engine (reference) ==="
"$TRTEXEC" --onnx=cracknet_sim.onnx --saveEngine=cracknet_fp32.engine \
  2>&1 | tee build_fp32.log | grep -E 'GPU Compute Time|Throughput|error|Error' || true

ls -la cracknet_*.engine
