"""cv_kit/pipeline — capture, buffer, inference, tracking, smoother, orchestrator."""
from .frame_buffer import FrameBuffer, Frame, BufferStats
from .capture      import CaptureThread
from .inference    import InferenceThread, PipelineResult
from .smoother     import CountSmoother, ValueSmoother, BBoxSmoother, build_smoothers
from .tracker      import build_tracker
from .pipeline     import Pipeline

__all__ = [
    "FrameBuffer", "Frame", "BufferStats",
    "CaptureThread",
    "InferenceThread", "PipelineResult",
    "CountSmoother", "ValueSmoother", "BBoxSmoother", "build_smoothers",
    "build_tracker",
    "Pipeline",
]