from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

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


def draw_trajectory(history, future, size=(640, 300)):
    canvas = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(canvas)
    margin = 36
    points = np.vstack((history[:, :2], future[:, :2]))
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    midpoint = (minimum + maximum) / 2.0
    span = maximum - minimum
    available_width = size[0] - 2 * margin
    available_height = size[1] - 2 * margin
    scale = min(
        available_width / max(float(span[0]), 1.0),
        available_height / max(float(span[1]), 1.0),
    )

    def project(point):
        x = size[0] / 2 + (point[0] - midpoint[0]) * scale
        y = size[1] / 2 - (point[1] - midpoint[1]) * scale
        return int(x), int(y)

    history_points = [project(point) for point in history[:, :2]]
    future_points = [project(point) for point in future[:, :2]]
    draw.line(history_points, fill=(36, 99, 170), width=4)
    draw.line([history_points[-1], *future_points], fill=(205, 62, 51), width=4)
    draw.ellipse((*np.subtract(history_points[-1], 6), *np.add(history_points[-1], 6)), fill="black")
    draw.ellipse((*np.subtract(future_points[-1], 6), *np.add(future_points[-1], 6)), fill=(205, 62, 51))
    draw.rectangle((margin, margin, size[0] - margin, size[1] - margin), outline=(190, 190, 190))
    draw.text((margin, 10), "Blue: 30-frame history    Red: 20-frame prediction target", fill="black")
    draw.text(
        (margin, size[1] - 24),
        f"Target end: forward {future[-1, 0]:.1f} m, lateral {future[-1, 1]:.1f} m",
        fill="black",
    )
    return canvas


def create_preview(row, data_root: Path, output_dir: Path):
    history_start = int(row["history_start_frame"])
    history_end = int(row["history_end_frame"])
    target_start = int(row["target_start_frame"])
    target_end = int(row["target_end_frame"])

    frames = decode_video_frames(
        data_root / row["video_path"], history_start, history_end, image_size=(320, 240)
    )
    arrays = load_pose_arrays(data_root / row["segment_path"])
    state_history, future_target = extract_motion_window(
        arrays, history_start, history_end, target_start, target_end
    )

    selected_indices = np.linspace(0, len(frames) - 1, 6, dtype=int)
    contact = Image.new("RGB", (960, 480), "white")
    draw = ImageDraw.Draw(contact)
    for panel, frame_index in enumerate(selected_indices):
        image = Image.fromarray(frames[frame_index])
        x = (panel % 3) * 320
        y = (panel // 3) * 240
        contact.paste(image, (x, y))
        draw.rectangle((x + 5, y + 5, x + 94, y + 25), fill="black")
        draw.text((x + 10, y + 8), f"frame {history_start + frame_index}", fill="white")

    trajectory = draw_trajectory(state_history, future_target, size=(960, 300))
    combined = Image.new("RGB", (960, 780), "white")
    combined.paste(contact, (0, 0))
    combined.paste(trajectory, (0, 480))

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "sequence_preview.png"
    gif_path = output_dir / "sequence_preview.gif"
    metadata_path = output_dir / "metadata.json"
    combined.save(png_path)
    gif_frames = [Image.fromarray(frame) for frame in frames]
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=50,
        loop=0,
        optimize=True,
    )
    metadata = {
        "sequence_id": row["sequence_id"],
        "chunk": row["chunk"],
        "split": row["split"],
        "route_id": row["route_id"],
        "segment_id": row["segment_id"],
        "history_frames": [history_start, history_end],
        "target_frames": [target_start, target_end],
        "mean_history_speed_mps": float(row["mean_history_speed_mps"]),
        "target_end_longitudinal_m": float(future_target[-1, 0]),
        "target_end_lateral_m": float(future_target[-1, 1]),
        "preview_png": str(png_path),
        "preview_gif": str(gif_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return png_path, gif_path, metadata


def main():
    parser = argparse.ArgumentParser(description="Create a visual check for one comma2k19 sequence.")
    parser.add_argument(
        "--index-csv",
        default=str(PREPARATION_DIR / "chunk_1_results" / "comma2k19_chunk_1_sequence_index.csv"),
    )
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "comma2k19"))
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--sample-number", type=int, default=0)
    parser.add_argument(
        "--output-dir", default=str(PREPARATION_DIR / "chunk_1_results" / "previews")
    )
    args = parser.parse_args()

    rows = [row for row in read_csv_rows(Path(args.index_csv)) if row["split"] == args.split]
    if not rows:
        raise ValueError(f"No rows found for split: {args.split}")
    row = rows[args.sample_number]
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    png_path, gif_path, _ = create_preview(row, data_root, output_dir)
    print("sequence_id:", row["sequence_id"])
    print("preview_png:", png_path)
    print("preview_gif:", gif_path)


if __name__ == "__main__":
    main()
