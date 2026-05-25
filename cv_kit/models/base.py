"""
cv_kit/models/base.py
──────────────────────
Abstract base class that every model backend must implement.

All three backends — YOLOv8, ONNX, TensorRT — expose the same interface
so that pipeline/inference.py never needs to know which backend it is
talking to. Swapping backends is a one-line config change.

Subclasses must implement:
    load()       — load weights, allocate GPU buffers
    predict()    — run a single frame through the model
    warmup()     — prime GPU kernels before the pipeline starts
    get_info()   — return metadata dict for logging/monitoring
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple


import numpy as np


# ─────────────────────────────────────────────
# Detection result
# ─────────────────────────────────────────────

@dataclass
class RawDetection:
    """
    One detected object returned by a model backend.

    Attributes
    ----------
    bbox        : [x1, y1, x2, y2] in original image pixels.
    confidence  : Model confidence score 0-1.
    class_id    : Integer class index.
    class_name  : Human-readable label (empty string if class_names not set).
    """
    bbox: np.ndarray
    confidence: float
    class_id: int
    class_name: str = ""

    def __post_init__(self) -> None:
        self.bbox = np.asarray(self.bbox, dtype=np.float64)

    def to_dict(self) -> dict:
        return {
            "bbox":       self.bbox.tolist(),
            "confidence": round(self.confidence, 4),
            "class_id":   self.class_id,
            "class_name": self.class_name,
        }

    def __repr__(self) -> str:
        x1, y1, x2, y2 = self.bbox.astype(int)
        return (
            f"RawDetection(class={self.class_name!r}, conf={self.confidence:.2f}, "
            f"bbox=[{x1},{y1},{x2},{y2}])"
        )


# ─────────────────────────────────────────────
# Base model config
# ─────────────────────────────────────────────

@dataclass
class BaseModelConfig:
    """
    Shared configuration for all model backends.
    Backend-specific configs (ONNXModelConfig, etc.) inherit from this.
    """
    model_path: str
    input_size: Tuple[int, int] = (640, 640)          # (height, width)
    confidence_threshold: float = 0.40
    nms_threshold: float = 0.45
    class_names: List[str] = field(default_factory=list)
    device: str = "cuda"                               # "cuda" | "cpu"
    fp16: bool = False

    def __post_init__(self) -> None:
        from pathlib import Path
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not 0 < self.confidence_threshold < 1:
            raise ValueError(f"confidence_threshold must be in (0,1), got {self.confidence_threshold}")
        if not 0 < self.nms_threshold < 1:
            raise ValueError(f"nms_threshold must be in (0,1), got {self.nms_threshold}")


# ─────────────────────────────────────────────
# Abstract interface
# ─────────────────────────────────────────────

class BaseModel(ABC):
    """
    Abstract base for all inference backends.

    Pipeline code should only ever hold a reference to BaseModel —
    never to a concrete subclass. This enforces the abstraction and
    makes unit testing trivial (mock this class, not the GPU).
    """

    def __init__(self, config: BaseModelConfig) -> None:
        self._config = config
        self._is_loaded: bool = False
        self._inference_count: int = 0

    # ── Must implement ────────────────────────

    @abstractmethod
    def load(self) -> None:
        """
        Load model weights and allocate any required GPU memory.
        Called once before the pipeline starts.
        Must set self._is_loaded = True on success.
        """
        ...

    @abstractmethod
    def predict(self, frame: np.ndarray) -> List[RawDetection]:
        """
        Run inference on a single BGR frame.

        Parameters
        ----------
        frame : uint8 BGR numpy array (H, W, 3) — raw from OpenCV.

        Returns
        -------
        List of RawDetection. Empty list = no detections above threshold.
        NMS must already have been applied before returning.
        """
        ...

    @abstractmethod
    def warmup(self, n_iters: int = 10) -> None:
        """
        Run n_iters dummy inferences to prime GPU kernels.
        Must be called after load() and before the first real predict().
        """
        ...

    @abstractmethod
    def get_info(self) -> dict:
        """
        Return a dict of backend metadata for logging and monitoring.
        Must include at minimum: backend, model_path, device, input_size.
        """
        ...

    # ── Provided ──────────────────────────────

    def _check_loaded(self) -> None:
        if not self._is_loaded:
            raise RuntimeError(
                f"{self.__class__.__name__} is not loaded. Call load() first."
            )

    def _resolve_class_name(self, class_id: int) -> str:
        names = self._config.class_names
        if names and 0 <= class_id < len(names):
            return names[class_id]
        return str(class_id)

    @property
    def input_size(self) -> Tuple[int, int]:
        return self._config.input_size

    @property
    def confidence_threshold(self) -> float:
        return self._config.confidence_threshold

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def inference_count(self) -> int:
        return self._inference_count

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"loaded={self._is_loaded}, "
            f"device={self._config.device!r}, "
            f"inferences={self._inference_count})"
        )