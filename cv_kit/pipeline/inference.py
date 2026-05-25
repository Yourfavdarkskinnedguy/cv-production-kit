"""
cv_kit/pipeline/inference.py
─────────────────────────────
Inference worker thread — the consumer side of the pipeline.

RESPONSIBILITY
──────────────
Pull Frame objects from the FrameBuffer, run the model, pass detections
to the tracker, collect results, push PipelineResult to the output queue.

Also handles the dropped-frame gap: whenever the inference thread was busy
and missed camera frames, it calls tracker.predict_only() for each gap
so the Kalman filter fills the missing positions correctly.

LATENCY TRACKING
────────────────
Every result carries:
  capture_ts    — wall-clock time the camera read the frame
  inference_ts  — wall-clock time inference started
  result_ts     — wall-clock time result was ready
  latency_ms    — capture_ts → result_ts (true end-to-end latency)

This is what you feed into the monitoring module for drift detection.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .frame_buffer import Frame, FrameBuffer
from ..models.base import BaseModel, RawDetection
from ..tracking.tracker import MultiObjectTracker
from ..tracking.track import Track

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────

@dataclass
class PipelineResult:
    """
    Emitted once per inference frame.

    Attributes
    ----------
    raw_frame    : Unmodified BGR frame from the camera.
    frame        : Annotated frame (if visualize=True, else same as raw_frame).
    tracks       : Confirmed Track objects for this frame.
    detections   : Raw model detections before tracking.
    frame_id     : Camera frame sequence number.
    source_id    : Camera identifier.
    capture_ts   : time.monotonic() at camera capture.
    inference_ts : time.monotonic() at inference start.
    result_ts    : time.monotonic() when result was ready.
    latency_ms   : capture → result latency in milliseconds.
    inference_ms : Model-only inference time in milliseconds.
    """
    raw_frame:    np.ndarray
    frame:        np.ndarray
    tracks:       List[Track]
    detections:   List[RawDetection]
    frame_id:     int
    source_id:    str
    capture_ts:   float
    inference_ts: float = field(default_factory=time.monotonic)
    result_ts:    float = field(default_factory=time.monotonic)
    latency_ms:   float = 0.0
    inference_ms: float = 0.0

    @property
    def track_count(self) -> int:
        return len(self.tracks)

    @property
    def detection_count(self) -> int:
        return len(self.detections)


# ─────────────────────────────────────────────
# Inference thread
# ─────────────────────────────────────────────

class InferenceThread:
    """
    Background thread: FrameBuffer → Model → Tracker → result_queue.

    Parameters
    ----------
    model          : Any loaded BaseModel subclass.
    tracker        : MultiObjectTracker instance.
    buffer         : FrameBuffer to pull frames from.
    result_queue   : queue.Queue to push PipelineResult into.
    on_result      : Optional callback fired per result (e.g. for webhooks).
    draw_fn        : Optional function(frame, tracks, detections) → annotated_frame.
    """

    def __init__(
        self,
        model:        BaseModel,
        tracker:      MultiObjectTracker,
        buffer:       FrameBuffer,
        result_queue: queue.Queue,
        on_result:    Optional[Callable[[PipelineResult], None]] = None,
        draw_fn:      Optional[Callable] = None,
    ) -> None:
        self._model         = model
        self._tracker       = tracker
        self._buffer        = buffer
        self._result_queue  = result_queue
        self._on_result     = on_result
        self._draw_fn       = draw_fn

        self._stop_event    = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Bookkeeping
        self._lock                = threading.Lock()
        self._frames_processed    = 0
        self._last_frame_id       = -1
        self._latency_window:  list[float] = []
        self._inference_window: list[float] = []

    # ── Lifecycle ─────────────────────────────────

    def start(self, thread_name: str = "inference") -> "InferenceThread":
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=thread_name,
        )
        self._thread.start()
        logger.info("InferenceThread started")
        return self

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: float = 5.0) -> bool:
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        return not (self._thread and self._thread.is_alive())

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ── Stats ──────────────────────────────────────

    def get_stats(self) -> dict:
        with self._lock:
            lats = list(self._latency_window)
            infs = list(self._inference_window)
            processed = self._frames_processed

        avg_lat = sum(lats) / len(lats) if lats else 0.0
        avg_inf = sum(infs) / len(infs) if infs else 0.0
        fps     = 1000.0 / avg_inf if avg_inf > 0 else 0.0

        return {
            "frames_processed": processed,
            "avg_latency_ms":   round(avg_lat, 2),
            "avg_inference_ms": round(avg_inf, 2),
            "p99_inference_ms": round(
                sorted(infs)[int(len(infs) * 0.99)] if len(infs) > 10 else avg_inf, 2
            ),
            "fps":              round(fps, 1),
            "backend":          self._model.get_info().get("backend", "?"),
        }

    # ── Internal ───────────────────────────────────

    def _run(self) -> None:
        logger.debug("InferenceThread running")

        while not self._stop_event.is_set():
            frame = self._buffer.get(timeout=0.05)
            if frame is None:
                continue   # buffer timeout — keep looping

            # Fill gaps with Kalman prediction for frames inference skipped
            if self._last_frame_id >= 0:
                gap = frame.frame_id - self._last_frame_id - 1
                for _ in range(min(gap, 5)):   # cap at 5 to avoid long fills
                    self._tracker.predict_only()

            self._last_frame_id = frame.frame_id

            # ── Inference ─────────────────────────
            t_inf = time.monotonic()
            try:
                detections = self._model.predict(frame.image)
            except Exception as e:
                logger.error("Model.predict() raised: %s", e, exc_info=True)
                continue
            inference_ms = (time.monotonic() - t_inf) * 1000.0

            # ── Tracking ──────────────────────────
            tracks = self._tracker.update(
                detections=detections,
                frame_id=frame.frame_id,
            )

            # ── Annotate ──────────────────────────
            annotated = frame.image
            if self._draw_fn is not None:
                try:
                    annotated = self._draw_fn(frame.image.copy(), tracks, detections)
                except Exception as e:
                    logger.warning("draw_fn raised: %s", e)

            # ── Emit result ───────────────────────
            result_ts  = time.monotonic()
            latency_ms = (result_ts - frame.timestamp) * 1000.0

            result = PipelineResult(
                raw_frame=frame.image,
                frame=annotated,
                tracks=tracks,
                detections=detections,
                frame_id=frame.frame_id,
                source_id=frame.source_id,
                capture_ts=frame.timestamp,
                inference_ts=t_inf,
                result_ts=result_ts,
                latency_ms=latency_ms,
                inference_ms=inference_ms,
            )

            try:
                self._result_queue.put_nowait(result)
            except queue.Full:
                logger.debug("result_queue full — dropping frame %d", frame.frame_id)

            if self._on_result:
                try:
                    self._on_result(result)
                except Exception as e:
                    logger.error("on_result callback raised: %s", e, exc_info=True)

            # Update stats
            with self._lock:
                self._frames_processed += 1
                self._latency_window.append(latency_ms)
                self._inference_window.append(inference_ms)
                if len(self._latency_window) > 100:
                    self._latency_window.pop(0)
                    self._inference_window.pop(0)

        logger.debug("InferenceThread exited")