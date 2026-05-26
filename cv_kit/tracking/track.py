"""
cv_kit/tracking/track.py
─────────────────────────
Track dataclass and TrackState enum.

STATE MACHINE
─────────────
  TENTATIVE → CONFIRMED → LOST → (deleted)

  TENTATIVE : 1 to (min_hits-1) detections. Not reported to the user yet.
              Suppresses false positives from single-frame ghost detections.
  CONFIRMED : min_hits or more detections. Included in PipelineResult.tracks.
  LOST      : No matched detection for 1..max_age frames.
              Kalman is still predicting. Deleted after max_age frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Tuple

import numpy as np


class TrackState(Enum):
    TENTATIVE = auto()
    CONFIRMED = auto()
    LOST      = auto()


@dataclass
class Track:
    """
    One tracked object.

    Attributes
    ----------
    track_id             : Unique int, monotonically increasing, never reused.
    bbox                 : [x1, y1, x2, y2] in pixels.
    class_id             : Integer class index from the model.
    class_name           : Human-readable label.
    confidence           : Most recent detection confidence (0–1).
    state                : Current TrackState.
    age                  : Frames since this track was created.
    hit_count            : Frames where a real detection was matched.
    frames_since_update  : Consecutive frames without a matched detection.
    frame_id_created     : Camera frame index when the track was first created.
    """

    track_id:            int
    bbox:                np.ndarray
    class_id:            int
    class_name:          str
    confidence:          float
    state:               TrackState = TrackState.TENTATIVE
    age:                 int        = 0
    hit_count:           int        = 0
    frames_since_update: int        = 0
    frame_id_created:    int        = 0

    def __post_init__(self) -> None:
        self.bbox = np.asarray(self.bbox, dtype=np.float64)

    # ── Derived properties ────────────────────────

    @property
    def centre(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return max(0.0, self.width * self.height)

    @property
    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    @property
    def is_lost(self) -> bool:
        return self.state == TrackState.LOST

    @property
    def is_tentative(self) -> bool:
        return self.state == TrackState.TENTATIVE

    def bbox_as_int(self) -> Tuple[int, int, int, int]:
        """Integer [x1, y1, x2, y2] for OpenCV drawing."""
        return tuple(self.bbox.astype(int).tolist())

    def to_dict(self) -> dict:
        """Serialise to a plain dict (JSON-safe)."""
        return {
            "track_id":            self.track_id,
            "bbox":                self.bbox.tolist(),
            "class_id":            self.class_id,
            "class_name":          self.class_name,
            "confidence":          round(self.confidence, 4),
            "state":               self.state.name,
            "age":                 self.age,
            "hit_count":           self.hit_count,
            "frames_since_update": self.frames_since_update,
            "centre":              list(self.centre),
            "area":                round(self.area, 1),
        }

    def __repr__(self) -> str:
        x1, y1, x2, y2 = self.bbox.astype(int)
        return (
            f"Track(id={self.track_id}, class={self.class_name!r}, "
            f"conf={self.confidence:.2f}, state={self.state.name}, "
            f"bbox=[{x1},{y1},{x2},{y2}], age={self.age})"
        )