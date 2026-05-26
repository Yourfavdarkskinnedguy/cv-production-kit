"""cv_kit/tracking — Kalman filter, track dataclass, multi-object tracker."""
from .kalman  import BBoxKalmanFilter
from .track   import Track, TrackState
from .tracker import MultiObjectTracker

__all__ = [
    "BBoxKalmanFilter",
    "Track", "TrackState",
    "MultiObjectTracker",
]