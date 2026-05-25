"""
cv_kit/models/yolo.py
──────────────────────
YOLOv8 backend using the Ultralytics library.

USE WHEN
────────
- Rapid prototyping or first-time setup
- You don't yet have ONNX/TensorRT installed
- You need the simplest path from .pt weights to running detections

PERFORMANCE (RTX 3080, YOLOv8n, 640×640)
──────────────────────────────────────────
  FP32 native predict(): ~12ms/frame → ~83 FPS
  Bottleneck: Python overhead in Ultralytics' results parsing

For production throughput, export to ONNX or TensorRT and use those backends.
This backend is the reference implementation you validate the others against.
"""

from __future__ import annotations

import logging
import time
from typing import List

import numpy as np

from .base import BaseModel, BaseModelConfig, RawDetection

logger = logging.getLogger(__name__)


class YOLOModelConfig(BaseModelConfig):
    """No extra fields needed — YOLOv8 handles everything internally."""
    pass


class YOLOModel(BaseModel):
    """
    Ultralytics YOLOv8 inference backend.

    Parameters
    ----------
    config : YOLOModelConfig
    """

    def __init__(self, config: YOLOModelConfig) -> None:
        super().__init__(config)
        self._model = None

    def load(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "Ultralytics not installed.\n"
                "Run: pip install ultralytics"
            )

        logger.info("Loading YOLOv8 model from %s on %s", self._config.model_path, self._config.device)
        self._model = YOLO(self._config.model_path)

        # Force model to the target device
        self._model.to(self._config.device)

        self._is_loaded = True
        logger.info("YOLOv8 model loaded — classes=%d", len(self._model.names))

        # Populate class names from model if config didn't specify them
        if not self._config.class_names:
            self._config.class_names = [
                self._model.names[i] for i in range(len(self._model.names))
            ]

    def predict(self, frame: np.ndarray) -> List[RawDetection]:
        """
        Run YOLOv8 inference on a single BGR frame.

        Parameters
        ----------
        frame : uint8 BGR numpy array from OpenCV

        Returns
        -------
        List[RawDetection] — NMS already applied by Ultralytics internally
        """
        self._check_loaded()

        results = self._model.predict(
            source=frame,
            conf=self._config.confidence_threshold,
            iou=self._config.nms_threshold,
            device=self._config.device,
            half=self._config.fp16,
            verbose=False,
            imgsz=self._config.input_size[0],
        )

        detections: List[RawDetection] = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()     # (N, 4) float32
            confs = r.boxes.conf.cpu().numpy()     # (N,)
            clses = r.boxes.cls.cpu().numpy().astype(int)  # (N,)

            for bbox, conf, cls_id in zip(boxes, confs, clses):
                detections.append(RawDetection(
                    bbox=bbox,
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=self._resolve_class_name(int(cls_id)),
                ))

        self._inference_count += 1
        return detections

    def warmup(self, n_iters: int = 10) -> None:
        self._check_loaded()
        logger.info("Warming up YOLOv8 (%d iterations)...", n_iters)
        dummy = np.zeros((*self._config.input_size, 3), dtype=np.uint8)
        for _ in range(n_iters):
            self._model.predict(dummy, verbose=False)
        logger.info("YOLOv8 warmup complete")

    def get_info(self) -> dict:
        return {
            "backend":     "yolo",
            "model_path":  self._config.model_path,
            "device":      self._config.device,
            "input_size":  self._config.input_size,
            "fp16":        self._config.fp16,
            "num_classes": len(self._config.class_names),
            "loaded":      self._is_loaded,
        }