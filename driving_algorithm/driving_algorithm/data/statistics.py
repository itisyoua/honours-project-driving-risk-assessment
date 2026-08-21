from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contracts import STATE_DIM


@dataclass(frozen=True)
class StateStatistics:
    mean: np.ndarray
    std: np.ndarray
    count: int

    def as_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "count": self.count,
        }


def _resolve(data_root: Path, relative_path: str) -> Path:
    path = (data_root / relative_path).resolve()
    if os.path.commonpath((data_root, path)) != str(data_root):
        raise ValueError(f"manifest path escapes data root: {relative_path}")
    return path


def compute_state_statistics(
    manifest_path: Path,
    data_root: Path,
    minimum_std: float = 1e-6,
) -> StateStatistics:
    if minimum_std <= 0:
        raise ValueError("minimum_std must be positive")
    data_root = Path(data_root).resolve()
    with Path(manifest_path).open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("manifest contains no samples")

    total = np.zeros(STATE_DIM, dtype=np.float64)
    total_squared = np.zeros(STATE_DIM, dtype=np.float64)
    count = 0
    for row in rows:
        with np.load(_resolve(data_root, row["motion_path"]), allow_pickle=False) as data:
            state = np.asarray(data["state_history"], dtype=np.float64)
            mask = np.asarray(data["history_mask"], dtype=np.bool_)
        if state.shape != (16, STATE_DIM) or mask.shape != (16,):
            raise ValueError(f"invalid state history in {row['motion_path']}")
        valid = state[mask]
        if valid.size == 0 or not np.isfinite(valid).all():
            raise ValueError(f"no finite valid state steps in {row['motion_path']}")
        total += valid.sum(axis=0)
        total_squared += np.square(valid).sum(axis=0)
        count += valid.shape[0]

    mean = total / count
    variance = np.maximum(total_squared / count - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(variance), minimum_std)
    return StateStatistics(
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        count=count,
    )
