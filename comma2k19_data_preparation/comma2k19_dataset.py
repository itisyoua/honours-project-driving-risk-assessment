from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .comma2k19_utils import (
        decode_video_frames,
        extract_motion_window,
        load_pose_arrays,
        read_csv_rows,
    )
except ImportError:
    from comma2k19_utils import (
        decode_video_frames,
        extract_motion_window,
        load_pose_arrays,
        read_csv_rows,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPARATION_DIR = Path(__file__).resolve().parent


def _default_index_path():
    combined = PREPARATION_DIR / "combined_results" / "comma2k19_combined_sequence_index.csv"
    if combined.exists():
        return combined
    return PREPARATION_DIR / "chunk_1_results" / "comma2k19_chunk_1_sequence_index.csv"


@lru_cache(maxsize=32)
def _cached_pose_arrays(segment_path: str):
    return load_pose_arrays(Path(segment_path))


class Comma2k19Dataset(Dataset):
    """PyTorch dataset returning video history, state history and future motion targets."""

    def __init__(
        self,
        index_csv: str | Path | None = None,
        data_root: str | Path | None = None,
        split: str | None = None,
        image_size: tuple[int, int] = (224, 224),
        normalisation_json: str | Path | None = None,
        normalise_images: bool = True,
        normalise_state: bool = True,
        max_samples: int | None = None,
    ):
        self.index_csv = Path(index_csv or _default_index_path())
        self.data_root = Path(data_root or PROJECT_ROOT / "comma2k19")
        self.image_size = image_size
        self.normalise_images = normalise_images
        self.normalise_state = normalise_state
        self.rows = read_csv_rows(self.index_csv)
        if split:
            self.rows = [row for row in self.rows if row["split"] == split]
        if max_samples is not None:
            self.rows = self.rows[:max_samples]

        inferred_stats = Path(
            str(self.index_csv).replace("_sequence_index.csv", "_normalisation.json")
        )
        stats_path = Path(normalisation_json or inferred_stats)
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        self.state_mean = torch.tensor(stats["state_history"]["mean"], dtype=torch.float32)
        self.state_std = torch.tensor(stats["state_history"]["std"], dtype=torch.float32)

        self.image_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        self.image_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        video_path = self.data_root / row["video_path"]
        segment_path = self.data_root / row["segment_path"]
        history_start = int(row["history_start_frame"])
        history_end = int(row["history_end_frame"])
        target_start = int(row["target_start_frame"])
        target_end = int(row["target_end_frame"])

        frames = decode_video_frames(video_path, history_start, history_end, self.image_size)
        frames = torch.from_numpy(np.array(frames, copy=True)).permute(0, 3, 1, 2).float() / 255.0
        if self.normalise_images:
            frames = (frames - self.image_mean) / self.image_std

        arrays = _cached_pose_arrays(str(segment_path))
        state_history, future_target = extract_motion_window(
            arrays, history_start, history_end, target_start, target_end
        )
        state_history = torch.from_numpy(state_history)
        if self.normalise_state:
            state_history = (state_history - self.state_mean) / self.state_std

        return {
            "frames": frames,
            "state_history": state_history,
            "future_target": torch.from_numpy(future_target),
            "sequence_id": row["sequence_id"],
            "route_id": row["route_id"],
            "split": row["split"],
        }


if __name__ == "__main__":
    dataset = Comma2k19Dataset(split="validation", max_samples=1)
    sample = dataset[0]
    print("sequence_id:", sample["sequence_id"])
    print("frames:", tuple(sample["frames"].shape))
    print("state_history:", tuple(sample["state_history"].shape))
    print("future_target:", tuple(sample["future_target"].shape))
