"""
cvkit/frame_buffer.py
─────────────────────
Thread-safe, bounded frame buffer that decouples camera capture (producer)
from model inference (consumer).

WHY THIS EXISTS
───────────────
Camera threads run at 25–30 FPS. Inference threads run at 8–25 FPS depending
on hardware. If you call model.predict() directly inside the capture loop the
camera blocks every time inference runs — you drop frames silently, your
perceived latency spikes, and on edge devices the OS camera buffer overflows.

The fix: put a bounded Queue between them. The camera pushes frames at full
speed. Inference pulls whenever it is ready. The buffer absorbs the timing
mismatch. If inference is too slow the buffer fills and the camera drops the
oldest frame — which is correct behaviour (you want the *freshest* frame, not
a 500ms stale one).

DESIGN DECISIONS
────────────────
- maxsize=4 by default: small enough that we never build up more than ~130ms
  of backlog at 30 FPS, large enough to absorb brief inference spikes.
- drop_policy="newest" vs "oldest": oldest (default FIFO) means inference
  always sees frames in order. "newest" can be useful for live dashboards
  where you want the latest frame even if you skipped some.
- Thread-safe: uses queue.Queue (backed by a threading.Condition) so it is
  safe across Python threads without extra locking.
- Metrics: tracks drop rate, queue depth, and throughput so the monitoring
  layer can detect when the pipeline is falling behind.

USAGE
─────
    buf = FrameBuffer(maxsize=4)
    buf.start()                        # start the drop-monitor thread

    # Producer thread (camera)
    buf.put(frame)                     # non-blocking, drops oldest if full

    # Consumer thread (inference)
    frame = buf.get(timeout=0.1)       # blocks until a frame arrives
    if frame is None:
        continue                       # timeout — no frame yet

    buf.stop()
    stats = buf.get_stats()
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


# ──────────────────────────────────────────────
# Data container
# ──────────────────────────────────────────────

@dataclass
class Frame:
    """
    Wraps a raw image with the metadata needed downstream.

    Attributes
    ----------
    image       : Raw BGR numpy array from OpenCV / camera SDK.
    frame_id    : Monotonically increasing counter. Used by the tracker to
                  detect gaps (i.e. dropped frames that need Kalman filling).
    timestamp   : Wall-clock time at *capture*, not at inference. Lets you
                  measure true end-to-end latency.
    source_id   : Camera or RTSP stream identifier. Useful in multi-camera
                  deployments so you know which stream a detection came from.
    """
    image: np.ndarray
    frame_id: int
    timestamp: float = field(default_factory=time.monotonic)
    source_id: str = "default"


@dataclass
class BufferStats:
    """Snapshot of buffer health at a point in time."""
    total_produced: int = 0
    total_consumed: int = 0
    total_dropped: int = 0
    current_depth: int = 0
    drop_rate: float = 0.0          # drops / produced, 0–1
    avg_wait_ms: float = 0.0        # average time a frame waits in buffer
    throughput_fps: float = 0.0     # consumed frames per second


# ──────────────────────────────────────────────
# Core buffer
# ──────────────────────────────────────────────

class FrameBuffer:
    """
    Bounded, thread-safe FIFO frame buffer.

    Parameters
    ----------
    maxsize     : Maximum number of frames in the buffer. Default 4.
    drop_policy : "oldest" (default) drops the head to make room for new
                  frames. "newest" discards the incoming frame instead —
                  use this if you need strict temporal ordering.
    name        : Identifier for log messages (useful in multi-camera setups).
    """

    def __init__(
        self,
        maxsize: int = 4,
        drop_policy: str = "oldest",
        name: str = "FrameBuffer",
    ) -> None:
        if maxsize < 1:
            raise ValueError(f"maxsize must be ≥ 1, got {maxsize}")
        if drop_policy not in ("oldest", "newest"):
            raise ValueError(f"drop_policy must be 'oldest' or 'newest', got {drop_policy!r}")

        self._q: queue.Queue[Frame] = queue.Queue(maxsize=maxsize)
        self._maxsize = maxsize
        self._drop_policy = drop_policy
        self._name = name

        # Stats — protected by a single lock so reads are consistent
        self._stats_lock = threading.Lock()
        self._total_produced: int = 0
        self._total_consumed: int = 0
        self._total_dropped: int = 0
        self._wait_times: list[float] = []      # ring-buffer of last 100 waits
        self._last_consume_ts: float = time.monotonic()

        # Monitor thread (logs warnings when drop rate exceeds threshold)
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = threading.Event()

        logger.debug("%s initialised — maxsize=%d policy=%s", name, maxsize, drop_policy)

    # ── Lifecycle ────────────────────────────────

    def start(self, monitor_interval: float = 5.0) -> "FrameBuffer":
        """
        Start the background monitor thread.

        The monitor logs a WARNING every `monitor_interval` seconds if the
        drop rate exceeds 10% — this is the earliest signal that your
        inference thread is too slow for the camera framerate.

        Returns self so you can do: buf = FrameBuffer().start()
        """
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
        """Signal the monitor thread to stop. Idempotent."""
        self._running.clear()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)

    # ── Producer API ─────────────────────────────

    def put(self, frame: Frame) -> bool:
        """
        Non-blocking insert. Returns True if the frame was queued, False if
        it was dropped.

        This is ALWAYS non-blocking by design. The camera thread must never
        stall waiting for the inference thread — that would defeat the purpose
        of having a buffer entirely.

        With drop_policy="oldest": if the queue is full, the oldest frame is
        evicted to make room for the new one. Inference sees the freshest data.

        With drop_policy="newest": if the queue is full, the new frame is
        silently discarded. Useful when temporal ordering is critical.
        """
        with self._stats_lock:
            self._total_produced += 1

        try:
            self._q.put_nowait(frame)
            return True
        except queue.Full:
            with self._stats_lock:
                self._total_dropped += 1

            if self._drop_policy == "oldest":
                # Evict the oldest frame and insert the new one
                try:
                    evicted = self._q.get_nowait()
                    logger.debug(
                        "%s: evicted frame_id=%d to make room for frame_id=%d",
                        self._name, evicted.frame_id, frame.frame_id,
                    )
                except queue.Empty:
                    pass  # Race condition — queue drained between Full and get
                try:
                    self._q.put_nowait(frame)
                    return True
                except queue.Full:
                    pass  # Lost the race — increment dropped and move on

            logger.debug("%s: dropped frame_id=%d (buffer full)", self._name, frame.frame_id)
            return False

    # ── Consumer API ─────────────────────────────

    def get(self, timeout: float = 0.1) -> Optional[Frame]:
        """
        Blocking get with timeout. Returns None if no frame arrives within
        `timeout` seconds.

        The inference loop should look like:
            while running:
                frame = buf.get(timeout=0.05)
                if frame is None:
                    continue   # no frame yet — keep looping
                result = model(frame.image)
        """
        try:
            frame = self._q.get(timeout=timeout)
        except queue.Empty:
            return None

        wait_ms = (time.monotonic() - frame.timestamp) * 1000.0

        with self._stats_lock:
            self._total_consumed += 1
            self._last_consume_ts = time.monotonic()
            # Maintain a rolling window of the last 100 wait times
            self._wait_times.append(wait_ms)
            if len(self._wait_times) > 100:
                self._wait_times.pop(0)

        return frame

    # ── Introspection ────────────────────────────

    def qsize(self) -> int:
        """Current number of frames waiting. Approximate under concurrency."""
        return self._q.qsize()

    def is_empty(self) -> bool:
        return self._q.empty()

    def is_full(self) -> bool:
        return self._q.full()

    def get_stats(self) -> BufferStats:
        """Return a consistent snapshot of buffer metrics."""
        with self._stats_lock:
            produced = self._total_produced
            consumed = self._total_consumed
            dropped = self._total_dropped
            wait_times = list(self._wait_times)
            last_ts = self._last_consume_ts

        drop_rate = dropped / produced if produced > 0 else 0.0
        avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0.0
        elapsed = time.monotonic() - last_ts
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
        """Drain all frames from the buffer. Returns number of frames removed."""
        count = 0
        while True:
            try:
                self._q.get_nowait()
                count += 1
            except queue.Empty:
                break
        logger.debug("%s: cleared %d frames", self._name, count)
        return count

    # ── Internal ─────────────────────────────────

    def _monitor_loop(self, interval: float) -> None:
        """Background thread: periodically logs health warnings."""
        while self._running.is_set():
            time.sleep(interval)
            stats = self.get_stats()

            if stats.drop_rate > 0.10:
                logger.warning(
                    "%s: HIGH DROP RATE %.1f%% — inference may be too slow "
                    "(produced=%d consumed=%d dropped=%d depth=%d/%d)",
                    self._name,
                    stats.drop_rate * 100,
                    stats.total_produced,
                    stats.total_consumed,
                    stats.total_dropped,
                    stats.current_depth,
                    self._maxsize,
                )
            elif stats.drop_rate > 0.02:
                logger.info(
                    "%s: drop_rate=%.1f%% avg_wait=%.1fms fps=%.1f",
                    self._name,
                    stats.drop_rate * 100,
                    stats.avg_wait_ms,
                    stats.throughput_fps,
                )
            else:
                logger.debug(
                    "%s: healthy — drop_rate=%.1f%% depth=%d/%d fps=%.1f",
                    self._name,
                    stats.drop_rate * 100,
                    stats.current_depth,
                    self._maxsize,
                    stats.throughput_fps,
                )

    # ── Dunder ───────────────────────────────────

    def __len__(self) -> int:
        return self._q.qsize()

    def __repr__(self) -> str:
        return (
            f"FrameBuffer(name={self._name!r}, maxsize={self._maxsize}, "
            f"qsize={self._q.qsize()}, policy={self._drop_policy!r})"
        )