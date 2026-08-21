from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .contracts import SequenceContract


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


class WaymoE2EDataset(Dataset):
    """PyTorch loader for converted Waymo E2E current-image samples."""

    def __init__(
        self,
        manifest_path: Path,
        data_root: Path,
        image_size: tuple[int, int] = (224, 224),
        normalise_images: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root).resolve()
        self.image_size = image_size
        self.normalise_images = normalise_images
        with self.manifest_path.open(newline="", encoding="utf-8") as source:
            self.rows = list(csv.DictReader(source))

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve(self, relative_path: str) -> Path:
        resolved = (self.data_root / relative_path).resolve()
        if os.path.commonpath((self.data_root, resolved)) != str(self.data_root):
            raise ValueError(f"manifest path escapes data root: {relative_path}")
        return resolved

    def _load_image(self, relative_path: str) -> np.ndarray:
        with Image.open(self._resolve(relative_path)) as source:
            image = source.convert("RGB").resize(self.image_size, Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        if self.normalise_images:
            array = (array - _IMAGENET_MEAN) / _IMAGENET_STD
        return np.ascontiguousarray(array, dtype=np.float32)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        with np.load(self._resolve(row["motion_path"]), allow_pickle=False) as motion:
            sample = {
                "image": torch.from_numpy(self._load_image(row["image_path"])),
                "state_history": torch.from_numpy(
                    motion["state_history"].astype(np.float32, copy=True)
                ),
                "future_target": torch.from_numpy(
                    motion["future_target"].astype(np.float32, copy=True)
                ),
                "history_mask": torch.from_numpy(
                    motion["history_mask"].astype(np.bool_, copy=True)
                ),
                "future_mask": torch.from_numpy(
                    motion["future_mask"].astype(np.bool_, copy=True)
                ),
                "sample_id": row["sample_id"],
                "route_id": row["route_id"],
                "source": "waymo_e2e",
                "split": row["split"],
                "scene_type": row["scene_type"],
            }
        SequenceContract.validate(sample)
        return sample
