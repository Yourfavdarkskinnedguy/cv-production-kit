"""
cv_kit/models/tensorrt_model.py
────────────────────────────────
NVIDIA TensorRT inference backend — maximum throughput on Jetson and RTX GPUs.

USE WHEN
────────
- You are on NVIDIA hardware (Jetson Orin, Jetson Xavier, RTX series)
- You need maximum throughput (lowest latency per frame)
- You have run export_tensorrt.py to compile a .engine file

PERFORMANCE (Jetson Orin NX 16GB, YOLOv8n, 640×640)
──────────────────────────────────────────────────────
  FP16 INT8:  ~9ms/frame → ~111 FPS
  FP16 FP16:  ~12ms/frame → ~83 FPS

vs ONNX on same hardware:
  FP16 ONNX:  ~22ms/frame → ~45 FPS

TensorRT is 2–3× faster than ONNX Runtime on Jetson.

SETUP
─────
1. Install TensorRT (comes with JetPack on Jetson, manual install on desktop)
2. Install pycuda: pip install pycuda
3. Export a .engine file:
       python scripts/export_tensorrt.py \
           --onnx models/yolov8n.onnx \
           --output models/yolov8n_fp16.engine \
           --fp16

4. Set in config.yaml:
       inference:
         backend: tensorrt
         model_path: models/yolov8n_fp16.engine
         fp16: true

ENGINE FILES ARE DEVICE-SPECIFIC
─────────────────────────────────
A .engine file compiled on an RTX 3080 will NOT run on a Jetson Orin.
You must compile a separate engine for each target device.
Always version your engine files with the device name:
    yolov8n_rtx3080_fp16.engine
    yolov8n_jetson_orin_nx_fp16.engine
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .base import BaseModel, BaseModelConfig, RawDetection
from .onnx_model import letterbox, to_tensor, scale_boxes

logger = logging.getLogger(__name__)


@dataclass
class TensorRTModelConfig(BaseModelConfig):
    """
    TensorRT-specific config.

    Extra Attributes
    ----------------
    workspace_mb : GPU memory workspace for TensorRT engine in MB.
                   Only used during engine BUILD, not inference.
    """
    workspace_mb: int = 4096


class TensorRTModel(BaseModel):
    """
    TensorRT inference backend.

    Requires: tensorrt, pycuda
    """

    def __init__(self, config: TensorRTModelConfig) -> None:
        super().__init__(config)
        self._engine  = None
        self._context = None
        self._stream  = None
        self._bindings:      List[int]       = []
        self._host_inputs:   List[np.ndarray] = []
        self._host_outputs:  List[np.ndarray] = []
        self._device_inputs:  List = []
        self._device_outputs: List = []

    def load(self) -> None:
        """
        Deserialise the TensorRT engine and allocate pinned GPU buffers.

        Buffer allocation strategy
        ──────────────────────────
        Pinned (page-locked) host memory allows async DMA transfers to the GPU.
        This overlaps data transfer with kernel execution, which is important
        for sustaining high FPS on real camera streams.
        """
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401  — initialises CUDA context
        except ImportError as e:
            raise ImportError(
                f"TensorRT or pycuda not installed ({e}).\n"
                "See docs/tensorrt_setup.md for installation instructions."
            )

        logger.info("Loading TensorRT engine: %s", self._config.model_path)

        with open(self._config.model_path, "rb") as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self._engine = runtime.deserialize_cuda_engine(f.read())

        if self._engine is None:
            raise RuntimeError(
                f"Failed to deserialise TensorRT engine: {self._config.model_path}\n"
                "The file may be corrupt or compiled for a different GPU."
            )

        self._context = self._engine.create_execution_context()
        self._stream  = cuda.Stream()

        # Allocate GPU + pinned host buffers for each binding
        for binding in self._engine:
            shape = self._engine.get_binding_shape(binding)
            size  = trt.volume(shape) * self._engine.max_batch_size
            dtype = trt.nptype(self._engine.get_binding_dtype(binding))

            host_mem   = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self._bindings.append(int(device_mem))

            if self._engine.binding_is_input(binding):
                self._host_inputs.append(host_mem)
                self._device_inputs.append(device_mem)
                logger.debug("Input  binding: %s shape=%s dtype=%s", binding, shape, dtype)
            else:
                self._host_outputs.append(host_mem)
                self._device_outputs.append(device_mem)
                logger.debug("Output binding: %s shape=%s dtype=%s", binding, shape, dtype)

        self._is_loaded = True
        logger.info("TensorRT engine loaded — GPU buffers allocated")

    def predict(self, frame: np.ndarray) -> List[RawDetection]:
        """
        Async GPU inference pipeline:
          1. Preprocess frame on CPU
          2. H2D copy (host → device, async)
          3. TensorRT execute_async_v2
          4. D2H copy (device → host, async)
          5. Stream synchronise
          6. Decode + NMS on CPU
        """
        self._check_loaded()

        try:
            import pycuda.driver as cuda
        except ImportError:
            raise RuntimeError("pycuda not available")

        cfg = self._config
        orig_h, orig_w = frame.shape[:2]

        # ── Preprocess ────────────────────────────
        padded, scale, pad = letterbox(frame, cfg.input_size)
        tensor = to_tensor(padded, fp16=cfg.fp16).ravel()

        # ── H2D transfer ──────────────────────────
        np.copyto(self._host_inputs[0], tensor)
        cuda.memcpy_htod_async(self._device_inputs[0], self._host_inputs[0], self._stream)

        # ── Inference ─────────────────────────────
        self._context.execute_async_v2(
            bindings=self._bindings,
            stream_handle=self._stream.handle,
        )

        # ── D2H transfer ──────────────────────────
        for h_out, d_out in zip(self._host_outputs, self._device_outputs):
            cuda.memcpy_dtoh_async(h_out, d_out, self._stream)

        self._stream.synchronize()

        # ── Decode output ─────────────────────────
        # Infer number of classes from output buffer size
        n_anchors = 8400   # YOLOv8 default for 640×640 input
        raw = self._host_outputs[0].astype(np.float32)

        n_cls = (len(raw) // n_anchors) - 4
        if n_cls <= 0:
            n_cls = max(1, len(cfg.class_names))

        raw = raw.reshape(-1, 4 + n_cls)  # [8400, 4+nc]

        # Reuse ONNX decode + NMS logic
        from .onnx_model import ONNXModel
        _dummy = ONNXModel.__new__(ONNXModel)
        _dummy._config = cfg

        detections = _dummy._decode_and_nms(raw, scale, pad, (orig_h, orig_w))
        self._inference_count += 1
        return detections

    def warmup(self, n_iters: int = 10) -> None:
        self._check_loaded()
        logger.info("Warming up TensorRT engine (%d iterations)...", n_iters)
        dummy = np.zeros((*self._config.input_size, 3), dtype=np.uint8)
        for _ in range(n_iters):
            self.predict(dummy)
        logger.info("TensorRT warmup complete")

    def get_info(self) -> dict:
        return {
            "backend":     "tensorrt",
            "model_path":  self._config.model_path,
            "device":      self._config.device,
            "input_size":  self._config.input_size,
            "fp16":        self._config.fp16,
            "loaded":      self._is_loaded,
            "max_batch":   self._engine.max_batch_size if self._engine else None,
        }