"""
cv_kit/tracking/tracker.py
───────────────────────────
Multi-object tracker: Hungarian assignment + Kalman prediction.

PIPELINE (per inference frame)
───────────────────────────────
1. kf.predict() for every existing track  → predicted bboxes
2. Compute IoU matrix  [n_tracks × n_detections]
3. Hungarian algorithm on (1 - IoU) cost matrix
4. Accept matches where IoU ≥ iou_threshold
5. Unmatched detections → new tentative tracks
6. Unmatched tracks → frames_since_update++
7. Prune lost tracks (frames_since_update > max_age)
8. Tentative → confirmed after min_hits detections

WHY IoU NOT EUCLIDEAN DISTANCE?
────────────────────────────────
Euclidean distance breaks at different scales (small objects far away vs
large objects close up). IoU is scale-invariant — 85% overlap means a match
regardless of absolute pixel size.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman import BBoxKalmanFilter
from .track  import Track, TrackState
from ..models.base import RawDetection

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# IoU helpers
# ─────────────────────────────────────────────

def _iou_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Pairwise IoU: a (N,4) × b (M,4) → (N,M).
    Uses numpy broadcasting — no Python loops.
    """
    tl = np.maximum(a[:, None, :2], b[None, :, :2])   # (N,M,2)
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])   # (N,M,2)
    wh = np.maximum(br - tl, 0.0)
    inter = wh[:, :, 0] * wh[:, :, 1]                 # (N,M)

    area_a = ((a[:, 2]-a[:, 0]) * (a[:, 3]-a[:, 1]))[:, None]
    area_b = ((b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1]))[None, :]
    union  = area_a + area_b - inter

    return np.where(union > 0, inter / union, 0.0)


def _hungarian(
    cost: np.ndarray,
    threshold: float,
) -> Tuple[List[Tuple[int,int]], List[int], List[int]]:
    """
    Hungarian (Munkres) assignment with IoU threshold gate.

    Returns
    -------
    matches           : [(track_idx, det_idx), ...]
    unmatched_tracks  : [track_idx, ...]
    unmatched_dets    : [det_idx, ...]
    """
    if cost.size == 0:
        nt, nd = cost.shape
        return [], list(range(nt)), list(range(nd))

    row_idx, col_idx = linear_sum_assignment(cost)

    matched_t, matched_d = set(), set()
    matches = []

    for r, c in zip(row_idx, col_idx):
        if cost[r, c] <= (1.0 - threshold):
            matches.append((r, c))
            matched_t.add(r)
            matched_d.add(c)

    unmatched_t = [i for i in range(cost.shape[0]) if i not in matched_t]
    unmatched_d = [j for j in range(cost.shape[1]) if j not in matched_d]

    return matches, unmatched_t, unmatched_d


# ─────────────────────────────────────────────
# Tracker
# ─────────────────────────────────────────────

class MultiObjectTracker:
    """
    Kalman-based multi-object tracker with Hungarian assignment.

    Parameters
    ----------
    iou_threshold   : Minimum IoU to accept a detection–track match.
    max_age         : Frames without a match before a track is deleted.
    min_hits        : Detections needed to confirm a tentative track.
    class_sensitive : Only match detections to tracks of the same class.
    process_noise   : Passed to BBoxKalmanFilter.
    measure_noise   : Passed to BBoxKalmanFilter.
    """

    def __init__(
        self,
        iou_threshold:   float = 0.30,
        max_age:         int   = 30,
        min_hits:        int   = 3,
        class_sensitive: bool  = True,
        process_noise:   float = 1.0,
        measure_noise:   float = 1.0,
    ) -> None:
        self.iou_threshold   = iou_threshold
        self.max_age         = max_age
        self.min_hits        = min_hits
        self.class_sensitive = class_sensitive
        self._pnoise         = process_noise
        self._mnoise         = measure_noise

        self._tracks:    Dict[int, Track] = {}
        self._kf_map:    Dict[int, BBoxKalmanFilter] = {}  # track_id → filter
        self._next_id:   int = 1
        self._frame_count: int = 0

        self._lock = threading.RLock()   # safe for get_tracks() from display thread

        logger.info(
            "MultiObjectTracker ready — iou=%.2f max_age=%d min_hits=%d",
            iou_threshold, max_age, min_hits,
        )

    # ── Public API ────────────────────────────────

    def update(
        self,
        detections: List[RawDetection],
        frame_id: Optional[int] = None,
    ) -> List[Track]:
        """
        Process one inference frame.

        Parameters
        ----------
        detections : List[RawDetection] from the model.
        frame_id   : Optional camera frame index for logging.

        Returns
        -------
        Confirmed tracks visible in this frame.
        """
        with self._lock:
            self._frame_count += 1
            fid = frame_id if frame_id is not None else self._frame_count

            track_list = list(self._tracks.values())

            # ── Step 1: Predict all tracks ─────────────
            for t in track_list:
                kf = self._kf_map[t.track_id]
                t.bbox = kf.predict()
                t.age  = kf.age
                t.frames_since_update = kf.frames_since_update

            # ── Step 2: IoU cost matrix ─────────────────
            if track_list and detections:
                track_bboxes = np.array([t.bbox for t in track_list])
                det_bboxes   = np.array([d.bbox for d in detections])
                iou_mat      = _iou_batch(track_bboxes, det_bboxes)

                if self.class_sensitive:
                    for ti, t in enumerate(track_list):
                        for di, d in enumerate(detections):
                            if t.class_id != d.class_id:
                                iou_mat[ti, di] = 0.0

                cost = 1.0 - iou_mat
                matches, unmatched_t, unmatched_d = _hungarian(cost, self.iou_threshold)
            else:
                matches      = []
                unmatched_t  = list(range(len(track_list)))
                unmatched_d  = list(range(len(detections)))

            # ── Step 3: Update matched tracks ──────────
            for ti, di in matches:
                t   = track_list[ti]
                det = detections[di]
                kf  = self._kf_map[t.track_id]

                t.bbox       = kf.update(det.bbox)
                t.confidence = det.confidence
                t.frames_since_update = 0
                t.hit_count  = kf.hit_count

                if t.state == TrackState.TENTATIVE and kf.is_confirmed(self.min_hits):
                    t.state = TrackState.CONFIRMED
                    logger.debug("Track %d confirmed", t.track_id)

            # ── Step 4: New tracks for unmatched dets ──
            for di in unmatched_d:
                self._create_track(detections[di], fid)

            # ── Step 5: Handle unmatched tracks ────────
            to_delete = []
            for ti in unmatched_t:
                t  = track_list[ti]
                kf = self._kf_map[t.track_id]
                if kf.is_lost(self.max_age):
                    to_delete.append(t.track_id)
                elif t.state == TrackState.CONFIRMED:
                    t.state = TrackState.LOST

            for tid in to_delete:
                del self._tracks[tid]
                del self._kf_map[tid]

            return [t for t in self._tracks.values()
                    if t.state == TrackState.CONFIRMED]

    def predict_only(self) -> List[Track]:
        """
        Advance all tracks one frame without detections.
        Call this for frames that inference skipped (camera faster than model).
        Returns confirmed tracks with Kalman-predicted positions.
        """
        with self._lock:
            for t in self._tracks.values():
                kf    = self._kf_map[t.track_id]
                t.bbox = kf.predict()
                t.age  = kf.age
                t.frames_since_update = kf.frames_since_update

            # Prune lost tracks
            to_delete = [
                tid for tid, t in self._tracks.items()
                if self._kf_map[tid].is_lost(self.max_age)
            ]
            for tid in to_delete:
                del self._tracks[tid]
                del self._kf_map[tid]

            return [t for t in self._tracks.values()
                    if t.state == TrackState.CONFIRMED]

    def get_tracks(self, include_tentative: bool = False) -> List[Track]:
        """Thread-safe snapshot of current tracks."""
        with self._lock:
            if include_tentative:
                return list(self._tracks.values())
            return [t for t in self._tracks.values()
                    if t.state == TrackState.CONFIRMED]

    def reset(self) -> None:
        """Clear all tracks and reset the ID counter."""
        with self._lock:
            self._tracks.clear()
            self._kf_map.clear()
            self._next_id   = 1
            self._frame_count = 0
        logger.info("Tracker reset")

    def get_stats(self) -> dict:
        with self._lock:
            confirmed  = sum(1 for t in self._tracks.values() if t.state == TrackState.CONFIRMED)
            tentative  = sum(1 for t in self._tracks.values() if t.state == TrackState.TENTATIVE)
            lost       = sum(1 for t in self._tracks.values() if t.state == TrackState.LOST)
        return {
            "confirmed_tracks": confirmed,
            "tentative_tracks": tentative,
            "lost_tracks":      lost,
            "total_tracks_ever": self._next_id - 1,
            "frame_count":       self._frame_count,
        }

    # ── Internal ──────────────────────────────────

    def _create_track(self, det: RawDetection, frame_id: int) -> Track:
        kf = BBoxKalmanFilter(
            process_noise_scale=self._pnoise,
            measure_noise_scale=self._mnoise,
        )
        kf.initialise(det.bbox)

        track = Track(
            track_id=self._next_id,
            bbox=det.bbox.copy(),
            class_id=det.class_id,
            class_name=det.class_name,
            confidence=det.confidence,
            state=TrackState.TENTATIVE,
            frame_id_created=frame_id,
        )

        self._tracks[self._next_id]  = track
        self._kf_map[self._next_id]  = kf
        self._next_id += 1
        return track

    # ── Dunder ────────────────────────────────────

    def __len__(self) -> int:
        return len(self._tracks)

    def __repr__(self) -> str:
        confirmed = sum(1 for t in self._tracks.values() if t.is_confirmed)
        return (
            f"MultiObjectTracker(tracks={len(self._tracks)}, "
            f"confirmed={confirmed}, frame={self._frame_count})"
        )