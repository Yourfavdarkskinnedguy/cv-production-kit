"""
cv_kit/pipeline/pipeline.py
────────────────────────────
Top-level pipeline orchestrator.

Wires together: CaptureThread → FrameBuffer → InferenceThread → result_queue

USAGE
─────
    import yaml
    from cv_kit.pipeline.pipeline import Pipeline

    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    pipeline = Pipeline.from_config(cfg)
    pipeline.start()

    try:
        while pipeline.is_running():
            result = pipeline.get_result(timeout=0.05)
            if result:
                cv2.imshow("cv-kit", result.frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        pipeline.stop()
        pipeline.join()

CONTEXT MANAGER
───────────────
    with Pipeline.from_config(cfg) as pipeline:
        while pipeline.is_running():
            result = pipeline.get_result()
            ...
"""

from __future__ import annotations

import logging
import queue
import time
from typing import Any, Callable, Dict, Optional

import cv2

from .capture      import CaptureThread
from .frame_buffer import FrameBuffer
from .inference    import InferenceThread, PipelineResult
from .tracker      import build_tracker
from .smoother     import build_smoothers
from ..models      import build_model
from ..tracking.tracker import MultiObjectTracker

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Full async two-thread CV pipeline.

    Parameters
    ----------
    capture_thread   : Configured CaptureThread (camera producer)
    inference_thread : Configured InferenceThread (model consumer)
    buffer           : Shared FrameBuffer between the two threads
    result_queue     : Queue where inference results are delivered
    smoothers        : Dict of smoother objects (count, confidence, bbox)
    """

    def __init__(
        self,
        capture_thread:   CaptureThread,
        inference_thread: InferenceThread,
        buffer:           FrameBuffer,
        result_queue:     queue.Queue,
        smoothers:        Optional[dict] = None,
    ) -> None:
        self._capture   = capture_thread
        self._inference = inference_thread
        self._buffer    = buffer
        self._rq        = result_queue
        self._smoothers = smoothers or {}
        self._start_time: Optional[float] = None

    # ── Factory ───────────────────────────────────

    @classmethod
    def from_config(
        cls,
        cfg: dict,
        draw_fn: Optional[Callable] = None,
        on_result: Optional[Callable[[PipelineResult], None]] = None,
    ) -> "Pipeline":
        """
        Build a complete pipeline from a config dict.

        Parameters
        ----------
        cfg       : Parsed YAML config dict (see configs/default.yaml)
        draw_fn   : Optional function(frame, tracks, detections) → annotated_frame
        on_result : Optional callback fired per result (logging, webhooks)
        """
        cam_cfg  = cfg.get("camera",    {})
        buf_cfg  = cfg.get("buffer",    {})
        inf_cfg  = cfg.get("inference", {})
        trk_cfg  = cfg.get("tracking",  {})
        out_cfg  = cfg.get("output",    {})

        # ── Buffer ────────────────────────────────
        buffer = FrameBuffer(
            maxsize=buf_cfg.get("maxsize", 4),
            drop_policy=buf_cfg.get("drop_policy", "oldest"),
            name="MainBuffer",
        )

        # ── Model ─────────────────────────────────
        model = build_model(
            backend=inf_cfg.get("backend", "yolo"),
            model_path=inf_cfg["model_path"],
            input_size=tuple(inf_cfg.get("input_size", [640, 640])),
            confidence_threshold=inf_cfg.get("confidence_threshold", 0.40),
            nms_threshold=inf_cfg.get("nms_threshold", 0.45),
            class_names=inf_cfg.get("class_names", []),
            device=inf_cfg.get("device", "cuda"),
            fp16=inf_cfg.get("fp16", False),
        )

        # Warmup after load
        model.warmup(n_iters=inf_cfg.get("warmup_iters", 10))

        # ── Tracker ───────────────────────────────
        tracker = build_tracker(trk_cfg)

        # ── Result queue ──────────────────────────
        result_queue: queue.Queue[PipelineResult] = queue.Queue(
            maxsize=out_cfg.get("result_queue_size", 8)
        )

        # ── Threads ───────────────────────────────
        source = cam_cfg.get("source", 0)
        try:
            source = int(source)
        except (ValueError, TypeError):
            pass   # keep as string (RTSP URL or file path)

        capture_thread = CaptureThread(
            source=source,
            buffer=buffer,
            source_id=cam_cfg.get("source_id", "cam_00"),
            max_retries=cam_cfg.get("max_retries", 10),
        )

        inference_thread = InferenceThread(
            model=model,
            tracker=tracker,
            buffer=buffer,
            result_queue=result_queue,
            on_result=on_result,
            draw_fn=draw_fn,
        )

        # ── Smoothers ─────────────────────────────
        smoothers = build_smoothers(
            camera_fps=cam_cfg.get("fps", 30.0),
            count_lag_s=1.5,
        )

        return cls(
            capture_thread=capture_thread,
            inference_thread=inference_thread,
            buffer=buffer,
            result_queue=result_queue,
            smoothers=smoothers,
        )

    # ── Lifecycle ─────────────────────────────────

    def start(self) -> "Pipeline":
        """Start capture and inference threads. Returns self."""
        self._start_time = time.monotonic()
        self._buffer.start(monitor_interval=10.0)
        self._capture.start()
        self._inference.start()
        logger.info("Pipeline started")
        return self

    def stop(self) -> None:
        """Signal both threads to stop. Non-blocking."""
        logger.info("Pipeline stopping...")
        self._capture.stop()
        self._inference.stop()
        self._buffer.stop()

    def join(self, timeout: float = 5.0) -> bool:
        """Wait for both threads to exit cleanly."""
        c = self._capture.join(timeout=timeout)
        i = self._inference.join(timeout=timeout)
        logger.info("Pipeline stopped (capture_clean=%s, inference_clean=%s)", c, i)
        return c and i

    # ── Consumer API ──────────────────────────────

    def get_result(self, timeout: float = 0.033) -> Optional[PipelineResult]:
        """
        Get the next result from the output queue.

        Call this in your display/logging loop. Returns None if no result
        is available within `timeout` seconds.
        """
        try:
            return self._rq.get(timeout=timeout)
        except queue.Empty:
            return None

    def smooth_count(self, raw_count: int) -> int:
        """Apply rolling median smoothing to a raw track count."""
        s = self._smoothers.get("count")
        return s.update(raw_count) if s else raw_count

    def is_running(self) -> bool:
        return self._capture.is_alive() or self._inference.is_alive()

    # ── Stats ──────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """
        Aggregate stats from all components.
        Feed this into the monitoring module or a Prometheus exporter.
        """
        elapsed = time.monotonic() - (self._start_time or time.monotonic())
        buf     = self._buffer.get_stats()
        cap     = self._capture.get_stats()
        inf     = self._inference.get_stats()

        return {
            "pipeline": {
                "uptime_s":         round(elapsed, 1),
                "frames_captured":  cap.frames_read,
                "frames_processed": inf["frames_processed"],
                "effective_fps":    round(inf["frames_processed"] / max(elapsed, 1e-6), 1),
                "reconnects":       cap.reconnects,
            },
            "buffer": {
                "drop_rate":     round(buf.drop_rate, 4),
                "avg_wait_ms":   round(buf.avg_wait_ms, 2),
                "current_depth": buf.current_depth,
                "total_dropped": buf.total_dropped,
            },
            "inference": inf,
        }

    # ── Context manager ────────────────────────────

    def __enter__(self) -> "Pipeline":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
        self.join()

    def __repr__(self) -> str:
        return f"Pipeline(running={self.is_running()})"