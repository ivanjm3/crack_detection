# trt_infer.py - minimal TensorRT 10 runner for CrackNet
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401  (creates CUDA context)

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class CrackNetTRT:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bufs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            host = cuda.pagelocked_empty(trt.volume(shape), dtype)
            dev = cuda.mem_alloc(host.nbytes)
            self.ctx.set_tensor_address(name, int(dev))
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            self.bufs[name] = (host, dev, shape, is_input)
        self.in_name = next(n for n, b in self.bufs.items() if b[3])
        self.out_name = next(n for n, b in self.bufs.items() if not b[3])
        self.size = self.bufs[self.in_name][2][-1]   # 512 (or 448)

    @staticmethod
    def preprocess(rgb):
        """rgb: HxWx3 uint8 already resized to SIZE x SIZE -> 1x3xHxW float32."""
        x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None])

    def infer(self, x):
        """x: 1x3xSxS float32 -> probability map SxS float32."""
        h_in, d_in, _, _ = self.bufs[self.in_name]
        h_out, d_out, oshape, _ = self.bufs[self.out_name]
        np.copyto(h_in, x.ravel())
        cuda.memcpy_htod_async(d_in, h_in, self.stream)
        self.ctx.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(h_out, d_out, self.stream)
        self.stream.synchronize()
        logits = h_out.reshape(oshape)[0, 0]
        return 1.0 / (1.0 + np.exp(-logits))
