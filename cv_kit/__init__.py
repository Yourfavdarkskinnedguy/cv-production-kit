"""
cv-production-kit
─────────────────
Production-ready computer vision pipeline.
Async inference · Multi-object tracking · Kalman filtering · Model monitoring.

Quick start
───────────
    import yaml
    from cv_kit import Pipeline

    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    with Pipeline.from_config(cfg) as pipeline:
        while pipeline.is_running():
            result = pipeline.get_result()
            if result:
                import cv2
                cv2.imshow("cv-kit", result.frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
"""

__version__ = "0.1.0"
__author__  = "Daniel"

# Top-level public API
from .pipeline.pipeline   import Pipeline
from .pipeline.inference  import PipelineResult
from .pipeline.frame_buffer import FrameBuffer, Frame
from .pipeline.smoother   import CountSmoother, ValueSmoother, BBoxSmoother

from .models              import build_model, BaseModel, RawDetection
from .tracking            import MultiObjectTracker, Track, TrackState, BBoxKalmanFilter
from .utils               import load_config, get

__all__ = [
    # Pipeline
    "Pipeline", "PipelineResult",
    "FrameBuffer", "Frame",
    "CountSmoother", "ValueSmoother", "BBoxSmoother",
    # Models
    "build_model", "BaseModel", "RawDetection",
    # Tracking
    "MultiObjectTracker", "Track", "TrackState", "BBoxKalmanFilter",
    # Utils
    "load_config", "get",
]