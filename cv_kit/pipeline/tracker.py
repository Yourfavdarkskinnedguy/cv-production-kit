"""
cv_kit/pipeline/tracker.py
───────────────────────────
Thin wrapper that instantiates MultiObjectTracker from config dict.

WHY THIS FILE EXISTS
────────────────────
The pipeline package needs to build a tracker from a raw config dict
without knowing about the tracking package internals. This wrapper
handles the translation and gives pipeline.py a clean one-liner:

    tracker = build_tracker(cfg["tracking"])

All the actual tracking logic lives in cv_kit/tracking/tracker.py.
"""

from __future__ import annotations

from ..tracking.tracker import MultiObjectTracker


def build_tracker(cfg: dict) -> MultiObjectTracker:
    """
    Build a MultiObjectTracker from a config dict.

    Expected keys (all optional — defaults shown):
        iou_threshold   : float = 0.30
        max_age         : int   = 30
        min_hits        : int   = 3
        class_sensitive : bool  = True
        process_noise   : float = 1.0
        measure_noise   : float = 1.0

    Example
    -------
        tracker = build_tracker(config["tracking"])
    """
    return MultiObjectTracker(
        iou_threshold=cfg.get("iou_threshold", 0.30),
        max_age=cfg.get("max_age", 30),
        min_hits=cfg.get("min_hits", 3),
        class_sensitive=cfg.get("class_sensitive", True),
        process_noise=cfg.get("process_noise", 1.0),
        measure_noise=cfg.get("measure_noise", 1.0),
    )


__all__ = ["build_tracker"]