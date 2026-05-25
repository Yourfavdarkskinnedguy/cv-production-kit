"""
cv_kit/pipeline/smoother.py
────────────────────────────
Count and value smoothers for real-time CV pipelines.

Three smoothers:
  CountSmoother — rolling median for integer counts (people, cars)
  ValueSmoother — EMA for continuous values (confidence, speed)
  BBoxSmoother  — per-track EMA for bounding box coordinates

See docstrings on each class for full detail.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Deque, Dict, List, Optional


class CountSmoother:
    """
    Rolling median over a sliding window.

    Best for: occupancy counts, queue lengths, people counters.
    Median is robust to outliers — one bad frame with count=0 won't
    drag the output down if all surrounding frames show count=7.

    Parameters
    ----------
    window_size : Frames in the rolling window.
                  Rule of thumb: camera_fps * desired_lag_seconds
                  e.g. 30fps × 1.5s = 45 frames
    min_value   : Clamp output minimum (default 0).
    """

    def __init__(self, window_size: int = 45, min_value: int = 0) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        self._window_size = window_size
        self._min_value   = min_value
        self._values: Deque[int] = deque(maxlen=window_size)
        self._last: int = 0
        self._lock = threading.Lock()

    def update(self, count: int) -> int:
        with self._lock:
            self._values.append(max(count, self._min_value))
            vals = sorted(self._values)
            n = len(vals)
            med = vals[n // 2] if n % 2 == 1 else (vals[n//2-1] + vals[n//2]) // 2
            self._last = max(med, self._min_value)
            return self._last

    def get(self) -> int:
        with self._lock:
            return self._last

    def reset(self) -> None:
        with self._lock:
            self._values.clear()
            self._last = 0

    @property
    def is_warm(self) -> bool:
        """True when the window is fully filled."""
        with self._lock:
            return len(self._values) >= self._window_size

    def __repr__(self) -> str:
        return f"CountSmoother(window={self._window_size}, last={self._last})"


class ValueSmoother:
    """
    Exponential Moving Average for continuous scalar values.

        y_t = alpha * x_t + (1 - alpha) * y_{t-1}

    alpha=0.1 → heavy smoothing, slow reaction
    alpha=0.5 → balanced
    alpha=0.9 → light smoothing, fast reaction

    Best for: confidence scores, bounding box centres, speed estimates.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        initial: Optional[float] = None,
        clamp_min: Optional[float] = None,
        clamp_max: Optional[float] = None,
    ) -> None:
        if not 0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self._alpha     = alpha
        self._value     = initial
        self._clamp_min = clamp_min
        self._clamp_max = clamp_max
        self._n         = 0
        self._lock      = threading.Lock()

    def update(self, value: float) -> float:
        with self._lock:
            if self._value is None:
                self._value = value
            else:
                self._value = self._alpha * value + (1.0 - self._alpha) * self._value
            if self._clamp_min is not None:
                self._value = max(self._value, self._clamp_min)
            if self._clamp_max is not None:
                self._value = min(self._value, self._clamp_max)
            self._n += 1
            return self._value

    def get(self) -> Optional[float]:
        with self._lock:
            return self._value

    def reset(self, initial: Optional[float] = None) -> None:
        with self._lock:
            self._value = initial
            self._n = 0

    def __repr__(self) -> str:
        v = f"{self._value:.4f}" if self._value is not None else "None"
        return f"ValueSmoother(alpha={self._alpha}, value={v})"


class BBoxSmoother:
    """
    Per-track bounding box coordinate EMA.

    Eliminates the 5-10px jitter that makes bounding boxes look unstable
    on screen even when the detected object is not moving.

    Parameters
    ----------
    alpha      : EMA smoothing factor. 0.4 works well for most detectors.
    max_tracks : Maximum simultaneous tracks to hold state for (LRU eviction).
    """

    def __init__(self, alpha: float = 0.4, max_tracks: int = 100) -> None:
        self._alpha         = alpha
        self._max_tracks    = max_tracks
        self._state: Dict[int, List[float]] = {}
        self._order: List[int] = []
        self._lock = threading.Lock()

    def update(self, track_id: int, bbox: List[float]) -> List[float]:
        """
        Return smoothed [x1, y1, x2, y2] for a track.
        Seeds with raw bbox on first call for this track_id.
        """
        with self._lock:
            if track_id not in self._state:
                self._state[track_id] = list(map(float, bbox))
                self._order.append(track_id)
                self._evict()
            else:
                cur = self._state[track_id]
                self._state[track_id] = [
                    self._alpha * float(n) + (1.0 - self._alpha) * o
                    for n, o in zip(bbox, cur)
                ]
                self._order.remove(track_id)
                self._order.append(track_id)
            return list(self._state[track_id])

    def remove(self, track_id: int) -> None:
        with self._lock:
            self._state.pop(track_id, None)
            if track_id in self._order:
                self._order.remove(track_id)

    def clear(self) -> None:
        with self._lock:
            self._state.clear()
            self._order.clear()

    def _evict(self) -> None:
        while len(self._state) > self._max_tracks:
            old = self._order.pop(0)
            self._state.pop(old, None)

    def __repr__(self) -> str:
        return f"BBoxSmoother(alpha={self._alpha}, tracks={len(self._state)})"


def build_smoothers(camera_fps: float = 30.0, count_lag_s: float = 1.5) -> dict:
    """
    Build the standard set of smoothers for a pipeline.

    Returns dict with keys: "count", "confidence", "bbox"

    Usage
    -----
        smoothers = build_smoothers(camera_fps=30.0)
        count = smoothers["count"].update(len(tracks))
        conf  = smoothers["confidence"].update(avg_conf)
        bbox  = smoothers["bbox"].update(track.track_id, track.bbox.tolist())
    """
    w = max(1, int(camera_fps * count_lag_s))
    return {
        "count":      CountSmoother(window_size=w),
        "confidence": ValueSmoother(alpha=0.3, clamp_min=0.0, clamp_max=1.0),
        "bbox":       BBoxSmoother(alpha=0.4),
    }


__all__ = ["CountSmoother", "ValueSmoother", "BBoxSmoother", "build_smoothers"]