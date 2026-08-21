from __future__ import annotations

import argparse
import csv
import os
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "driving-algorithm-matplotlib")
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _resolve(data_root: Path, relative_path: str) -> Path:
    resolved = (data_root / relative_path).resolve()
    if os.path.commonpath((data_root, resolved)) != str(data_root):
        raise ValueError(f"path escapes data root: {relative_path}")
    return resolved


def preview_sample(
    manifest_path: Path,
    data_root: Path,
    output_path: Path,
    index: int = 0,
) -> Path:
    data_root = Path(data_root).resolve()
    with Path(manifest_path).open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("manifest contains no samples")
    row = rows[index]

    with Image.open(_resolve(data_root, row["image_path"])) as source_image:
        image = source_image.convert("RGB").copy()
    with np.load(_resolve(data_root, row["motion_path"]), allow_pickle=False) as motion:
        history = motion["state_history"].copy()
        future = motion["future_target"].copy()

    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    axes[0].imshow(image)
    axes[0].set_title("Current FRONT camera")
    axes[0].axis("off")

    axes[1].plot(history[:, 0], history[:, 1], color="#2563a6", linewidth=2.5)
    axes[1].plot(future[:, 0], future[:, 1], color="#d33f32", linewidth=2.5)
    axes[1].scatter([0.0], [0.0], color="black", s=55, zorder=4)
    axes[1].scatter(
        [future[-1, 0]], [future[-1, 1]], color="#d33f32", s=55, zorder=4
    )
    axes[1].set_title("Blue: 4 s history   Red: 5 s target")
    axes[1].set_xlabel("Forward (m)")
    axes[1].set_ylabel("Left (m)")
    axes[1].grid(alpha=0.25)
    axes[1].set_aspect("equal", adjustable="datalim")
    figure.suptitle(
        f"{row['sample_id']} | {row['split']} | {row['scene_type']}\n"
        f"Target end: forward {future[-1, 0]:.1f} m, lateral {future[-1, 1]:.1f} m",
        fontsize=11,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=120)
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview a converted Waymo sample.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    print(preview_sample(args.manifest, args.data_root, args.output, args.index))


if __name__ == "__main__":
    main()
