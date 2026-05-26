"""
cv_kit/tracking/kalman.py
──────────────────────────
Constant-velocity Kalman filter for 2-D bounding box tracking.

STATE VECTOR  [cx, cy, s, r, ẋ, ẏ, ṡ]
──────────────
  cx, cy : bounding box centre (pixels)
  s      : box area (pixels²)
  r      : aspect ratio w/h  — treated as nearly constant
  ẋ, ẏ  : velocity of centre (pixels/frame)
  ṡ      : rate of change of area

MEASUREMENT VECTOR  [cx, cy, s, r]
───────────────────
Detections provide the first 4 components.
Velocities are inferred by the filter over time.

WHY THIS MATTERS
────────────────
Camera: 30 FPS.  Inference: 18 FPS.  Gap: 12 frames/second with no detection.
Without prediction, tracked objects vanish and re-appear with a new ID.
The Kalman filter predicts where each object should be on the missing frames
using its last known velocity, so the tracker can match them correctly when
a detection fires again.
"""

from __future__ import annotations
from typing import Optional
import numpy as np


# ─────────────────────────────────────────────
# Coordinate helpers
# ─────────────────────────────────────────────

def xyxy_to_z(bbox: np.ndarray) -> np.ndarray:
    """[x1,y1,x2,y2] → measurement vector [cx,cy,s,r]."""
    x1, y1, x2, y2 = bbox[:4].astype(float)
    w  = x2 - x1
    h  = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    s  = w * h                    # area
    r  = w / (h + 1e-6)           # aspect ratio
    return np.array([cx, cy, s, r], dtype=np.float64)


def z_to_xyxy(state: np.ndarray) -> np.ndarray:
    """Kalman state [cx,cy,s,r,...] → [x1,y1,x2,y2]."""
    cx, cy, s, r = state[:4]
    s = max(s, 1.0)
    w = np.sqrt(s * max(r, 1e-6))
    h = s / (w + 1e-6)
    return np.array([
        cx - w / 2.0,
        cy - h / 2.0,
        cx + w / 2.0,
        cy + h / 2.0,
    ], dtype=np.float64)


# ─────────────────────────────────────────────
# Filter
# ─────────────────────────────────────────────

class BBoxKalmanFilter:
    """
    One instance per tracked object.
    Created by MultiObjectTracker on every new detection.

    Parameters
    ----------
    process_noise_scale : Tune up for fast-moving objects.
    measure_noise_scale : Tune up for noisy / occluded detections.
    """

    DIM_X = 7   # state  : cx cy s r  ẋ ẏ ṡ
    DIM_Z = 4   # measure: cx cy s r

    def __init__(
        self,
        process_noise_scale: float = 1.0,
        measure_noise_scale: float = 1.0,
    ) -> None:

        # ── State transition F ────────────────────
        # x_{t+1} = F * x_t   (constant velocity model)
        F = np.eye(self.DIM_X, dtype=np.float64)
        # cx += ẋ,  cy += ẏ,  s += ṡ
        F[0, 4] = 1.0   # cx  ← ẋ
        F[1, 5] = 1.0   # cy  ← ẏ
        F[2, 6] = 1.0   # s   ← ṡ
        self.F = F

        # ── Measurement matrix H ──────────────────
        # We observe [cx, cy, s, r] directly
        self.H = np.eye(self.DIM_Z, self.DIM_X, dtype=np.float64)

        # ── Process noise Q ───────────────────────
        self.Q = np.diag([
            1.0,    # cx
            1.0,    # cy
            10.0,   # s   (area changes more than centre)
            0.01,   # r   (aspect ratio is stable)
            0.01,   # ẋ
            0.01,   # ẏ
            0.0001, # ṡ
        ]).astype(np.float64) * process_noise_scale

        # ── Measurement noise R ───────────────────
        self.R = np.diag(
            [1.0, 1.0, 10.0, 0.01]
        ).astype(np.float64) * measure_noise_scale

        # ── Initial covariance P ──────────────────
        # Large = high uncertainty at birth of a new track
        self.P = np.diag(
            [10.0, 10.0, 100.0, 1.0, 1000.0, 1000.0, 100.0]
        ).astype(np.float64)

        self.x: Optional[np.ndarray] = None  # state vector

        # Bookkeeping
        self.age:                 int = 0
        self.hit_count:           int = 0
        self.frames_since_update: int = 0

    # ── Public API ────────────────────────────────

    def initialise(self, bbox: np.ndarray) -> None:
        """Seed the filter from the first detection for this track."""
        z = xyxy_to_z(bbox)
        # Initial state: position from detection, zero velocity
        self.x = np.concatenate([z, np.zeros(self.DIM_X - self.DIM_Z)])
        self.age = 0
        self.hit_count = 1
        self.frames_since_update = 0

    def predict(self) -> np.ndarray:
        """
        Advance state one timestep.
        Call ONCE per frame, before update().
        Returns predicted [x1,y1,x2,y2].
        """
        if self.x is None:
            raise RuntimeError("Filter not initialised — call initialise() first")

        # Prevent area from going negative during extrapolation
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0.0

        # State prediction: x = F*x
        self.x = self.F @ self.x
        # Covariance prediction: P = F*P*F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age += 1
        self.frames_since_update += 1

        return z_to_xyxy(self.x)

    def update(self, bbox: np.ndarray) -> np.ndarray:
        """
        Correct state with a new matched detection.
        Returns corrected [x1,y1,x2,y2].
        """
        if self.x is None:
            self.initialise(bbox)
            return z_to_xyxy(self.x)

        z = xyxy_to_z(bbox).reshape(-1, 1)

        # Innovation
        y = z - (self.H @ self.x).reshape(-1, 1)

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + (K @ y).flatten()

        # Covariance update — Joseph form (numerically stable)
        I_KH = np.eye(self.DIM_X) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        self.frames_since_update = 0
        self.hit_count += 1

        return z_to_xyxy(self.x)

    def get_bbox(self) -> Optional[np.ndarray]:
        """Current estimated [x1,y1,x2,y2] or None if uninitialised."""
        return None if self.x is None else z_to_xyxy(self.x)

    def is_confirmed(self, min_hits: int = 3) -> bool:
        """True once we have seen min_hits matched detections."""
        return self.hit_count >= min_hits

    def is_lost(self, max_age: int = 30) -> bool:
        """True if unmatched for more than max_age frames → delete track."""
        return self.frames_since_update > max_age

    def __repr__(self) -> str:
        bbox = "uninit" if self.x is None else z_to_xyxy(self.x).round(1).tolist()
        return (
            f"BBoxKalmanFilter(bbox={bbox}, "
            f"hits={self.hit_count}, "
            f"since_update={self.frames_since_update})"
        )