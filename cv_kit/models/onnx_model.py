"""
cv_kit/models/onnx_model.py
────────────────────────────
ONNX Runtime inference backend.

USE WHEN
────────
- You need cross-platform portability (works on CPU, CUDA, OpenVINO, CoreML)
- You want ~1.4× faster inference vs raw Ultralytics predict()
- You are targeting a device without TensorRT (Raspberry Pi, Intel NUC, Mac)
- As a stepping stone before compiling a TensorRT engine

EXPORT
──────
Export from YOLOv8 first:
    python scripts/export_onnx.py --model models/yolov8n.pt --output models/yolov8n.onnx

Then in config.yaml:
    inference:
      backend: onnx
      model_path: models/yolov8n.onnx

PERFORMANCE (RTX 3080, YOLOv8n, 640×640, CUDAExecutionProvider)
───────────────────────────────────────────────────────────────────
  FP32: ~8ms/frame → ~125 FPS
  FP16: ~5ms/frame → ~200 FPS

NOTES ON YOLOv8 ONNX FORMAT
────────────────────────────
Ultralytics exports YOLOv8 with the decode/NMS layers OUTSIDE the ONNX graph.
Raw output shape: [1, 4+num_classes, 8400]
  - 4         = cx, cy, w, h
  - num_classes = class scores (no separate objectness score in v8)
  - 8400      = number of anchor-free prediction slots

We transpose to [8400, 4+nc], decode, and apply OpenCV NMS here.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .base import BaseModel, BaseModelConfig, RawDetection

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

@dataclass
class ONNXModelConfig(BaseModelConfig):
    """
    ONNX-specific config on top of BaseModelConfig.

    Extra Attributes
    ----------------
    num_threads : Number of intra-op CPU threads. Ignored when using GPU.
                  4 is a good default for most server CPUs.
    """
    num_threads: int = 4


# ─────────────────────────────────────────────
# Preprocessing helpers
# ─────────────────────────────────────────────

def letterbox(
    image: np.ndarray,
    target_size: Tuple[int, int],
    stride: int = 32,
    fill: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize preserving aspect ratio, pad to target_size.

    Returns
    -------
    padded  : Resized and padded image
    scale   : Scale factor applied (undo this in postprocessing)
    pad     : (pad_w, pad_h) pixels added on each side
    """
    h0, w0 = image.shape[:2]
    th, tw = target_size
    scale = min(th / h0, tw / w0)
    nh, nw = int(round(h0 * scale)), int(round(w0 * scale))

    # Snap to stride multiple for faster convolutions
    nw = int(np.ceil(nw / stride) * stride)
    nh = int(np.ceil(nh / stride) * stride)

    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_w = (tw - nw) // 2
    pad_h = (th - nh) // 2

    padded = cv2.copyMakeBorder(
        resized,
        pad_h, th - nh - pad_h,
        pad_w, tw - nw - pad_w,
        cv2.BORDER_CONSTANT, value=fill,
    )
    return padded, scale, (pad_w, pad_h)


def to_tensor(image: np.ndarray, fp16: bool = False) -> np.ndarray:
    """
    BGR uint8 HWC  →  float32 (or float16) NCHW tensor.
    Normalises [0, 255] → [0.0, 1.0].
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    t = rgb.astype(np.float32) / 255.0
    t = np.transpose(t, (2, 0, 1))[np.newaxis]  # HWC → NCHW
    if fp16:
        t = t.astype(np.float16)
    return np.ascontiguousarray(t)


def scale_boxes(
    boxes: np.ndarray,
    scale: float,
    pad: Tuple[int, int],
    orig_shape: Tuple[int, int],
) -> np.ndarray:
    """Undo letterbox to get boxes in original image coordinates."""
    boxes = boxes.copy().astype(np.float64)
    boxes[:, [0, 2]] -= pad[0]   # remove x padding
    boxes[:, [1, 3]] -= pad[1]   # remove y padding
    boxes /= scale
    oh, ow = orig_shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, ow)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, oh)
    return boxes


# ─────────────────────────────────────────────
# Backend
# ─────────────────────────────────────────────

class ONNXModel(BaseModel):
    """
    ONNX Runtime inference backend for YOLOv8 models.

    Parameters
    ----------
    config : ONNXModelConfig
    """

    def __init__(self, config: ONNXModelConfig) -> None:
        super().__init__(config)
        self._session = None
        self._input_name: Optional[str] = None

    def load(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime not installed.\n"
                "GPU: pip install onnxruntime-gpu\n"
                "CPU: pip install onnxruntime"
            )

        cfg = self._config

        # Choose execution providers based on device
        if cfg.device.lower() in ("cpu",):
            providers = ["CPUExecutionProvider"]
        else:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = cfg.num_threads

        logger.info("Loading ONNX model: %s | providers=%s", cfg.model_path, providers)
        self._session = ort.InferenceSession(
            cfg.model_path, sess_options=opts, providers=providers,
        )
        self._input_name = self._session.get_inputs()[0].name

        active = self._session.get_providers()
        logger.info("ONNX session active providers: %s", active)
        self._is_loaded = True

    def predict(self, frame: np.ndarray) -> List[RawDetection]:
        """
        Preprocess → ONNX inference → decode → NMS → RawDetection list.
        """
        self._check_loaded()
        cfg = self._config
        orig_h, orig_w = frame.shape[:2]

        # Preprocess
        padded, scale, pad = letterbox(frame, cfg.input_size)
        tensor = to_tensor(padded, fp16=cfg.fp16)

        # Inference
        outputs = self._session.run(None, {self._input_name: tensor})
        raw = outputs[0]   # [1, 4+nc, 8400]

        # Decode YOLOv8 output format
        if raw.ndim == 3 and raw.shape[1] < raw.shape[2]:
            raw = raw[0].T       # → [8400, 4+nc]
        else:
            raw = raw[0]

        detections = self._decode_and_nms(raw, scale, pad, (orig_h, orig_w))
        self._inference_count += 1
        return detections

    def _decode_and_nms(
        self,
        predictions: np.ndarray,
        scale: float,
        pad: Tuple[int, int],
        orig_shape: Tuple[int, int],
    ) -> List[RawDetection]:
        """
        Decode raw [N, 4+nc] tensor → filter → NMS → RawDetection list.

        YOLOv8 format: [cx, cy, w, h, cls0_score, cls1_score, ...]
        No separate objectness — class score IS the confidence.
        """
        cfg = self._config
        if len(predictions) == 0:
            return []

        # Best class and score per prediction
        scores = predictions[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(class_ids)), class_ids]

        # Confidence filter
        mask = confidences >= cfg.confidence_threshold
        if not mask.any():
            return []
        predictions  = predictions[mask]
        confidences  = confidences[mask]
        class_ids    = class_ids[mask]

        # cx,cy,w,h → x1,y1,w,h (OpenCV NMS expects x,y,w,h)
        cx, cy = predictions[:, 0], predictions[:, 1]
        w,  h  = predictions[:, 2], predictions[:, 3]
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        boxes_xywh = np.stack([x1, y1, w, h], axis=1)
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        detections: List[RawDetection] = []

        # Per-class NMS to avoid suppressing different classes
        for cls_id in np.unique(class_ids):
            m = class_ids == cls_id
            cls_xywh  = boxes_xywh[m].tolist()
            cls_confs  = confidences[m].tolist()
            cls_xyxy   = boxes_xyxy[m]

            indices = cv2.dnn.NMSBoxes(
                cls_xywh, cls_confs,
                cfg.confidence_threshold,
                cfg.nms_threshold,
            )
            if len(indices) == 0:
                continue

            indices = np.array(indices).flatten()
            kept_xyxy = scale_boxes(cls_xyxy[indices], scale, pad, orig_shape)
            kept_conf = np.array(cls_confs)[indices]

            for bbox, conf in zip(kept_xyxy, kept_conf):
                detections.append(RawDetection(
                    bbox=bbox,
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=self._resolve_class_name(int(cls_id)),
                ))

        return detections

    def warmup(self, n_iters: int = 10) -> None:
        self._check_loaded()
        cfg = self._config
        logger.info("Warming up ONNX model (%d iterations)...", n_iters)
        dummy_dtype = np.float16 if cfg.fp16 else np.float32
        dummy = np.zeros((1, 3, *cfg.input_size), dtype=dummy_dtype)
        for _ in range(n_iters):
            self._session.run(None, {self._input_name: dummy})
        logger.info("ONNX warmup complete")

    def get_info(self) -> dict:
        providers = self._session.get_providers() if self._session else []
        return {
            "backend":     "onnx",
            "model_path":  self._config.model_path,
            "device":      self._config.device,
            "input_size":  self._config.input_size,
            "fp16":        self._config.fp16,
            "providers":   providers,
            "loaded":      self._is_loaded,
        }