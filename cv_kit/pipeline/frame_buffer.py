"""
cv_kit/pipeline/frame_buffer.py
───────────────────────────────
Thread-safe bounded frame buffer — decouples camera capture from inference.

DESIGN
──────
Camera thread (producer) → FrameBuffer (queue, maxsize=4) → Inference thread (consumer)

The camera NEVER blocks waiting for inference. If the buffer is full, the
oldest frame is evicted to make room. This means inference always sees the
freshest available frame, not a stale one from 500ms ago.

Memory usage is bounded:
    4 frames × 1920×1080 × 3 channels × 1 byte = ~24 MB max
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    """
    A single video frame with capture metadata.

    Attributes
    ----------
    image     : Raw BGR uint8 numpy array (H, W, 3).
    frame_id  : Monotonically increasing integer. The tracker uses gaps
                in this sequence to detect dropped frames.
    timestamp : Wall-clock time at CAPTURE (time.monotonic()).
                Used to compute true end-to-end pipeline latency.
    source_id : Camera/stream identifier for multi-camera setups.
    """
    image:     np.ndarray
    frame_id:  int
    timestamp: float = field(default_factory=time.monotonic)
    source_id: str   = "default"


@dataclass
class BufferStats:
    total_produced:  int   = 0
    total_consumed:  int   = 0
    total_dropped:   int   = 0
    current_depth:   int   = 0
    drop_rate:       float = 0.0   # 0.0 – 1.0
    avg_wait_ms:     float = 0.0
    throughput_fps:  float = 0.0


class FrameBuffer:
    """
    Bounded, thread-safe frame queue.

    Parameters
    ----------
    maxsize     : Maximum frames held in the buffer (default 4).
    drop_policy : "oldest" (default) — evict oldest to make room for new frame.
                  "newest"           — discard incoming frame if buffer is full.
    name        : Identifier for log messages.
    """

    def __init__(
        self,
        maxsize: int = 4,
        drop_policy: str = "oldest",
        name: str = "FrameBuffer",
    ) -> None:
        if maxsize < 1:
            raise ValueError(f"maxsize must be >= 1, got {maxsize}")
        if drop_policy not in ("oldest", "newest"):
            raise ValueError(f"drop_policy must be 'oldest' or 'newest', got {drop_policy!r}")

        self._q           = queue.Queue(maxsize=maxsize)
        self._maxsize     = maxsize
        self._drop_policy = drop_policy
        self._name        = name

        self._lock            = threading.Lock()
        self._total_produced  = 0
        self._total_consumed  = 0
        self._total_dropped   = 0
        self._wait_ms_window: list[float] = []   # rolling last-100

        self._running        = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._last_consume   = time.monotonic()

    # ── Lifecycle ─────────────────────────────────

    def start(self, monitor_interval: float = 10.0) -> "FrameBuffer":
        """Start background health-monitor thread. Returns self."""
        self._running.set()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(monitor_interval,),
            daemon=True,
            name=f"{self._name}-monitor",
        )
        self._monitor_thread.start()
        return self

    def stop(self) -> None:
        """Stop the monitor thread."""
        self._running.clear()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)

    # ── Producer API ─────────────────────────

    def put(self, frame: Frame) -> bool:
        """
        Non-blocking insert.
        Returns True if queued, False if dropped.
        """
        with self._lock:
            self._total_produced += 1

        try:
            self._q.put_nowait(frame)
            return True
        except queue.Full:
            with self._lock:
                self._total_dropped += 1

            if self._drop_policy == "oldest":
                try:
                    evicted = self._q.get_nowait()
                    logger.debug(
                        "%s evicted frame_id=%d for frame_id=%d",
                        self._name, evicted.frame_id, frame.frame_id,
                    )
                except queue.Empty:
                    pass
                try:
                    self._q.put_nowait(frame)
                    return True
                except queue.Full:
                    pass

            return False

    # ── Consumer API ──────────────────────────────

    def get(self, timeout: float = 0.1) -> Optional[Frame]:
        """
        Blocking get. Returns None on timeout.

        Inference loop pattern:
            while running:
                frame = buf.get(timeout=0.05)
                if frame is None:
                    continue
                detections = model.predict(frame.image)
        """
        try:
            frame = self._q.get(timeout=timeout)
        except queue.Empty:
            return None

        wait_ms = (time.monotonic() - frame.timestamp) * 1000.0
        with self._lock:
            self._total_consumed += 1
            self._last_consume = time.monotonic()
            self._wait_ms_window.append(wait_ms)
            if len(self._wait_ms_window) > 100:
                self._wait_ms_window.pop(0)

        return frame

    # ── Introspection ──────────────────────────────

    def get_stats(self) -> BufferStats:
        with self._lock:
            produced  = self._total_produced
            consumed  = self._total_consumed
            dropped   = self._total_dropped
            waits     = list(self._wait_ms_window)
            last_cons = self._last_consume

        drop_rate  = dropped / produced if produced > 0 else 0.0
        avg_wait   = sum(waits) / len(waits) if waits else 0.0
        elapsed    = time.monotonic() - last_cons
        throughput = consumed / max(elapsed, 1e-6)

        return BufferStats(
            total_produced=produced,
            total_consumed=consumed,
            total_dropped=dropped,
            current_depth=self._q.qsize(),
            drop_rate=drop_rate,
            avg_wait_ms=avg_wait,
            throughput_fps=min(throughput, 9999.0),
        )

    def clear(self) -> int:
        """Drain all frames. Returns number removed."""
        count = 0
        while True:
            try:
                self._q.get_nowait()
                count += 1
            except queue.Empty:
                break
        return count

    def qsize(self)   -> int:  return self._q.qsize()
    def is_empty(self) -> bool: return self._q.empty()
    def is_full(self)  -> bool: return self._q.full()
    def __len__(self)  -> int:  return self._q.qsize()

    # ── Internal ─────────────────────────────────

    def _monitor_loop(self, interval: float) -> None:
        while self._running.is_set():
            time.sleep(interval)
            s = self.get_stats()
            if s.drop_rate > 0.10:
                logger.warning(
                    "%s HIGH DROP RATE %.1f%% — inference too slow? "
                    "(produced=%d consumed=%d dropped=%d depth=%d/%d)",
                    self._name, s.drop_rate * 100,
                    s.total_produced, s.total_consumed, s.total_dropped,
                    s.current_depth, self._maxsize,
                )
            else:
                logger.debug(
                    "%s healthy — drop=%.1f%% depth=%d/%d wait=%.1fms",
                    self._name, s.drop_rate * 100,
                    s.current_depth, self._maxsize, s.avg_wait_ms,
                )

    def __repr__(self) -> str:
        return (
            f"FrameBuffer(name={self._name!r}, size={self._q.qsize()}/{self._maxsize}, "
            f"policy={self._drop_policy!r})"
        )