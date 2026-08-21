from __future__ import annotations

from collections.abc import Mapping

import numpy as np


HISTORY_STEPS = 16
FUTURE_STEPS = 20
STATE_DIM = 8
TARGET_DIM = 5
SAMPLE_HZ = 4.0

_REQUIRED_KEYS = {
    "image",
    "state_history",
    "future_target",
    "history_mask",
    "future_mask",
    "sample_id",
    "route_id",
    "source",
    "split",
    "scene_type",
}
_ALLOWED_SPLITS = {"train", "validation", "test"}
_ALLOWED_SOURCES = {"comma2k19", "waymo_e2e", "carla"}


def make_sample_id(source: str, route_id: str, timestamp_micros: int) -> str:
    if not source or not route_id:
        raise ValueError("source and route_id must be non-empty")
    if timestamp_micros < 0:
        raise ValueError("timestamp_micros must be non-negative")
    return f"{source}:{route_id}:{timestamp_micros}"


class SequenceContract:
    """Strict validator for model-ready sequence dictionaries."""

    @staticmethod
    def validate(sample: Mapping[str, object]) -> None:
        missing = sorted(_REQUIRED_KEYS.difference(sample))
        if missing:
            raise ValueError(f"missing sample keys: {', '.join(missing)}")

        arrays = {
            "image": np.asarray(sample["image"]),
            "state_history": np.asarray(sample["state_history"]),
            "future_target": np.asarray(sample["future_target"]),
            "history_mask": np.asarray(sample["history_mask"]),
            "future_mask": np.asarray(sample["future_mask"]),
        }
        expected_shapes = {
            "image": (3, 224, 224),
            "state_history": (HISTORY_STEPS, STATE_DIM),
            "future_target": (FUTURE_STEPS, TARGET_DIM),
            "history_mask": (HISTORY_STEPS,),
            "future_mask": (FUTURE_STEPS,),
        }
        for name, expected_shape in expected_shapes.items():
            if arrays[name].shape != expected_shape:
                raise ValueError(
                    f"{name} shape must be {expected_shape}, got {arrays[name].shape}"
                )

        if arrays["history_mask"].dtype != np.bool_:
            raise ValueError("history_mask must have boolean dtype")
        if arrays["future_mask"].dtype != np.bool_:
            raise ValueError("future_mask must have boolean dtype")
        for name in ("image", "state_history", "future_target"):
            if not np.isfinite(arrays[name]).all():
                raise ValueError(f"{name} must contain only finite values")

        for key in ("sample_id", "route_id", "scene_type"):
            if not isinstance(sample[key], str) or not sample[key]:
                raise ValueError(f"{key} must be a non-empty string")
        if sample["source"] not in _ALLOWED_SOURCES:
            raise ValueError(f"unsupported source: {sample['source']}")
        if sample["split"] not in _ALLOWED_SPLITS:
            raise ValueError(f"unsupported split: {sample['split']}")
