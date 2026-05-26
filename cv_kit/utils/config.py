"""
cv_kit/utils/config.py
───────────────────────
Config loader: reads YAML, validates required keys, merges CLI overrides.

USAGE
─────
    # Load from file
    cfg = load_config("configs/default.yaml")

    # Load and override specific keys
    cfg = load_config("configs/default.yaml", overrides={
        "inference.model_path": "models/yolov8s.onnx",
        "inference.backend":    "onnx",
        "camera.source":        "rtsp://192.168.1.10:554/stream",
    })

    # Access nested keys safely
    conf_thresh = get(cfg, "inference.confidence_threshold", default=0.4)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Required keys — raise early if missing
# ─────────────────────────────────────────────

_REQUIRED_KEYS = [
    "camera.source",
    "inference.model_path",
    "inference.backend",
]


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def load_config(
    config_path: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Load a YAML config file and apply optional key overrides.

    Parameters
    ----------
    config_path : Path to the YAML file.
    overrides   : Dict of dotted-key → value pairs that override config values.
                  e.g. {"inference.backend": "onnx", "camera.source": 0}

    Returns
    -------
    Merged config dict with all required sections present.

    Raises
    ------
    FileNotFoundError : config_path does not exist.
    ValueError        : A required key is missing after loading.
    yaml.YAMLError    : The file is not valid YAML.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Tip: run from the project root or pass an absolute path."
        )

    with open(path, encoding="utf-8") as f:
        cfg: dict = yaml.safe_load(f) or {}

    # Ensure every top-level section exists
    for section in ("camera", "buffer", "inference", "tracking", "output", "monitoring", "logging"):
        cfg.setdefault(section, {})

    # Apply overrides (dotted notation: "inference.backend" → cfg["inference"]["backend"])
    if overrides:
        for dotted_key, value in overrides.items():
            _set_nested(cfg, dotted_key, value)
            logger.debug("Config override: %s = %r", dotted_key, value)

    # Validate required keys
    for key in _REQUIRED_KEYS:
        if get(cfg, key) is None:
            raise ValueError(
                f"Required config key missing: {key!r}\n"
                f"Add it to {config_path} or pass it as an override."
            )

    # Coerce camera.source to int if it looks like a webcam index
    source = cfg["camera"].get("source")
    if isinstance(source, str):
        try:
            cfg["camera"]["source"] = int(source)
        except ValueError:
            pass   # keep as string (RTSP URL, file path)

    logger.info("Config loaded: %s", config_path)
    return cfg


def get(cfg: dict, dotted_key: str, default: Any = None) -> Any:
    """
    Safely read a nested config value using dotted notation.

    Example
    -------
        thresh = get(cfg, "inference.confidence_threshold", default=0.4)
    """
    keys = dotted_key.split(".")
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def merge_configs(base: dict, override: dict) -> dict:
    """
    Deep-merge two config dicts. override takes precedence.
    Does not mutate either input.
    """
    import copy
    result = copy.deepcopy(base)
    _deep_merge(result, override)
    return result


def save_config(cfg: dict, output_path: str) -> None:
    """Write a config dict back to a YAML file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    logger.info("Config saved: %s", output_path)


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _set_nested(cfg: dict, dotted_key: str, value: Any) -> None:
    """Set a value at a dotted path, creating intermediate dicts as needed."""
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def _deep_merge(base: dict, override: dict) -> None:
    """In-place recursive merge of override into base."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v