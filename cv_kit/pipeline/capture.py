"""
cv_kit/pipeline/capture.py
───────────────────────────
Camera capture thread — the producer side of the pipeline.

RESPONSIBILITY
──────────────
One job: read frames from a source as fast as possible and push them into
the FrameBuffer. Nothing else. No inference, no tracking, no drawing.

This strict separation is what allows the camera to run at 30 FPS
independently of how slow inference is.

SOURCES SUPPORTED
─────────────────
  - Integer  : webcam index (0 = first USB/built-in camera)
  - RTSP URL : "rtsp://192.168.1.100:554/stream1"
  - File path: "data/video.mp4" or "data/video.avi"

RECONNECTION
────────────
RTSP streams drop. Cameras get unplugged. CaptureThread automatically
retries on failure with exponential back-off (max 30s between retries).
It logs a WARNING on first failure and ERROR if it cannot reconnect after
max_retries attempts, then sets the stop event so the pipeline shuts down.

USAGE
─────
    buf = FrameBuffer(maxsize=4)
    capture = CaptureThread(source=0, buffer=buf)
    capture.start()
    # ... pipeline runs ...
    capture.stop()
    capture.join()
    stats = capture.get_stats()
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import cv2

from .frame_buffer import Frame, FrameBuffer

logger = logging.getLogger(__name__)


@dataclass
class CaptureStats:
    frames_read:    int   = 0
    frames_dropped: int   = 0   # failed cap.read() calls
    reconnects:     int   = 0
    uptime_s:       float = 0.0
    fps_actual:     float = 0.0


class CaptureThread:
    """
    Background thread that reads frames from a camera or video source
    and pushes them into a FrameBuffer.

    Parameters
    ----------
    source      : Camera index, RTSP URL, or file path.
    buffer      : FrameBuffer instance to push frames into.
    source_id   : String label for this camera (used in logs and Frame metadata).
    max_retries : Max reconnection attempts before giving up (-1 = infinite).
    retry_delay : Initial seconds between retries (doubles each attempt, cap 30s).
    """

    def __init__(
        self,
        source: Union[int, str],
        buffer: FrameBuffer,
        source_id: str = "cam_00",
        max_retries: int = 10,
        retry_delay: float = 1.0,
    ) -> None:
        self._source      = source
        self._buffer      = buffer
        self._source_id   = source_id
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        self._stop_event  = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None

        # Stats
        self._lock           = threading.Lock()
        self._frames_read    = 0
        self._frames_dropped = 0
        self._reconnects     = 0
        self._start_time: Optional[float] = None

    # ── Lifecycle ─────────────────────────────────

    def start(self) -> "CaptureThread":
        """Start the capture thread. Returns self for chaining."""
        self._stop_event.clear()
        self._start_time = time.monotonic()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"capture[{self._source_id}]",
        )
        self._thread.start()
        logger.info("CaptureThread started — source=%s id=%s", self._source, self._source_id)
        return self

    def stop(self) -> None:
        """Signal the thread to stop. Non-blocking."""
        self._stop_event.set()
        logger.debug("CaptureThread stop requested")

    def join(self, timeout: float = 5.0) -> bool:
        """Wait for thread to exit. Returns True if clean."""
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        if self._cap and self._cap.isOpened():
            self._cap.release()
        return not (self._thread and self._thread.is_alive())

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ── Stats ──────────────────────────────────────

    def get_stats(self) -> CaptureStats:
        with self._lock:
            elapsed = time.monotonic() - (self._start_time or time.monotonic())
            fps = self._frames_read / max(elapsed, 1e-6)
            return CaptureStats(
                frames_read=self._frames_read,
                frames_dropped=self._frames_dropped,
                reconnects=self._reconnects,
                uptime_s=elapsed,
                fps_actual=fps,
            )

    # ── Internal ───────────────────────────────────

    def _open_source(self) -> bool:
        """Open (or re-open) the video source. Returns True on success."""
        if self._cap and self._cap.isOpened():
            self._cap.release()

        self._cap = cv2.VideoCapture(self._source)

        if not self._cap.isOpened():
            logger.error("Failed to open source: %s", self._source)
            return False

        # Tell the OS driver to only buffer 1 frame so we always get fresh data.
        # Without this, on RTSP streams you can get frames that are seconds old.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        logger.info(
            "Source opened: %s — size=%dx%d fps=%.1f",
            self._source,
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            self._cap.get(cv2.CAP_PROP_FPS),
        )
        return True

    def _run(self) -> None:
        """Main capture loop — runs in the background thread."""
        retries     = 0
        delay       = self._retry_delay
        frame_id    = 0

        if not self._open_source():
            self._stop_event.set()
            return

        while not self._stop_event.is_set():
            ret, image = self._cap.read()

            if not ret:
                # End of file (video) — stop cleanly
                if isinstance(self._source, str) and not self._source.startswith("rtsp"):
                    logger.info("Video file ended at frame %d", frame_id)
                    self._stop_event.set()
                    break

                # Camera or stream failure — try to reconnect
                with self._lock:
                    self._frames_dropped += 1

                retries += 1
                if self._max_retries >= 0 and retries > self._max_retries:
                    logger.error(
                        "Source %s: max reconnect attempts (%d) reached — stopping",
                        self._source, self._max_retries,
                    )
                    self._stop_event.set()
                    break

                logger.warning(
                    "Source %s read failed (attempt %d/%s) — retrying in %.1fs",
                    self._source, retries,
                    self._max_retries if self._max_retries >= 0 else "∞",
                    delay,
                )
                time.sleep(delay)
                delay = min(delay * 2.0, 30.0)   # exponential back-off, cap 30s

                if self._open_source():
                    retries = 0
                    delay   = self._retry_delay
                    with self._lock:
                        self._reconnects += 1
                continue

            # Successful read
            retries = 0
            delay   = self._retry_delay

            frame = Frame(
                image=image,
                frame_id=frame_id,
                timestamp=time.monotonic(),
                source_id=self._source_id,
            )
            self._buffer.put(frame)

            with self._lock:
                self._frames_read += 1

            frame_id += 1

        # Cleanup
        if self._cap and self._cap.isOpened():
            self._cap.release()
        logger.debug("CaptureThread exited at frame_id=%d", frame_id)