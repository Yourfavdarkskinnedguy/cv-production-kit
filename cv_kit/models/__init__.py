"""cv_kit/models — model backend registry."""
from .base          import BaseModel, BaseModelConfig, RawDetection
from .yolo          import YOLOModel, YOLOModelConfig
from .onnx_model    import ONNXModel, ONNXModelConfig
from .tensorrt_model import TensorRTModel, TensorRTModelConfig

# Registry: backend string → (model class, config class)
BACKEND_REGISTRY = {
    "yolo":      (YOLOModel,      YOLOModelConfig),
    "onnx":      (ONNXModel,      ONNXModelConfig),
    "tensorrt":  (TensorRTModel,  TensorRTModelConfig),
}


def build_model(backend: str, **kwargs) -> BaseModel:
    """
    Factory function. Build and load a model from a backend string.

    Parameters
    ----------
    backend : "yolo" | "onnx" | "tensorrt"
    **kwargs : passed directly to the config dataclass

    Example
    -------
    model = build_model(
        backend="onnx",
        model_path="models/yolov8n.onnx",
        confidence_threshold=0.4,
        device="cuda",
    )
    detections = model.predict(frame)
    """
    if backend not in BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend: {backend!r}. "
            f"Choose from: {list(BACKEND_REGISTRY.keys())}"
        )
    model_cls, config_cls = BACKEND_REGISTRY[backend]

    # Only pass kwargs that the config dataclass accepts
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(config_cls)}
    filtered = {k: v for k, v in kwargs.items() if k in valid_fields}

    config = config_cls(**filtered)
    model  = model_cls(config)
    model.load()
    return model


__all__ = [
    "BaseModel", "BaseModelConfig", "RawDetection",
    "YOLOModel", "YOLOModelConfig",
    "ONNXModel", "ONNXModelConfig",
    "TensorRTModel", "TensorRTModelConfig",
    "BACKEND_REGISTRY", "build_model",
]