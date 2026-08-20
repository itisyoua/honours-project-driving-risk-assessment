from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from .comma2k19_utils import (
        POSE_FILES,
        STATE_FEATURE_NAMES,
        TARGET_FEATURE_NAMES,
        extract_motion_window,
        load_array,
        load_pose_arrays,
        scalar_motion_features,
    )
except ImportError:
    from comma2k19_utils import (
        POSE_FILES,
        STATE_FEATURE_NAMES,
        TARGET_FEATURE_NAMES,
        extract_motion_window,
        load_array,
        load_pose_arrays,
        scalar_motion_features,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPARATION_DIR = Path(__file__).resolve().parent


def safe_float(value):
    if value is None:
        return ""
    return round(float(value), 6)


def array_shape(path: Path):
    array = load_array(path)
    if array is None:
        return ""
    return "x".join(str(value) for value in array.shape)


def segment_sort_key(video_path: Path):
    route = video_path.parent.parent.name
    segment = video_path.parent.name
    try:
        segment_number = int(segment)
    except ValueError:
        segment_number = segment
    return route, segment_number


def assign_route_splits(route_ids, validation_ratio: float, test_ratio: float, seed: int):
    routes = sorted(set(route_ids))
    if len(routes) < 3:
        raise ValueError("At least three routes are required for train/validation/test splitting")

    shuffled = routes.copy()
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, round(len(routes) * validation_ratio))
    test_count = max(1, round(len(routes) * test_ratio))
    if validation_count + test_count >= len(routes):
        raise ValueError("Validation and test ratios leave no training routes")

    test_routes = set(shuffled[:test_count])
    validation_routes = set(shuffled[test_count : test_count + validation_count])
    return {
        route: (
            "test"
            if route in test_routes
            else "validation"
            if route in validation_routes
            else "train"
        )
        for route in routes
    }


def inspect_segment(video_path: Path, chunk_path: Path):
    segment_path = video_path.parent
    route_path = segment_path.parent
    relative_segment_path = segment_path.relative_to(chunk_path.parent)
    row = {
        "chunk": chunk_path.name,
        "route_id": route_path.name,
        "segment_id": segment_path.name,
        "segment_path": str(relative_segment_path),
        "video_path": str(video_path.relative_to(chunk_path.parent)),
        "video_size_bytes": video_path.stat().st_size,
        "has_raw_log_bz2": (segment_path / "raw_log.bz2").exists(),
        "has_preview_png": (segment_path / "preview.png").exists(),
    }

    missing = []
    arrays = {}
    for name, relative_path in POSE_FILES.items():
        path = segment_path / relative_path
        exists = path.exists()
        row[f"has_{name}"] = exists
        row[f"{name}_shape"] = array_shape(path) if exists else ""
        if exists:
            arrays[name] = load_array(path)
        else:
            missing.append(name)

    frame_times = arrays.get("frame_times")
    frame_count = int(len(frame_times)) if frame_times is not None else 0
    row["frame_count"] = frame_count
    inconsistent = [name for name, array in arrays.items() if len(array) != frame_count]
    non_finite = [name for name, array in arrays.items() if not np.all(np.isfinite(array))]

    if frame_count:
        row["start_time_s"] = safe_float(frame_times[0])
        row["end_time_s"] = safe_float(frame_times[-1])
        duration = float(frame_times[-1] - frame_times[0]) if frame_count > 1 else 0.0
        row["duration_s"] = safe_float(duration)
        row["estimated_fps"] = safe_float((frame_count - 1) / duration) if duration > 0 else ""
        row["timestamps_monotonic"] = bool(np.all(np.diff(frame_times) > 0))
    else:
        row.update(
            start_time_s="",
            end_time_s="",
            duration_s="",
            estimated_fps="",
            timestamps_monotonic=False,
        )

    velocities = arrays.get("frame_velocities")
    if velocities is not None and velocities.ndim == 2 and len(velocities):
        speeds = np.linalg.norm(np.asarray(velocities), axis=1)
        row["mean_speed_mps"] = safe_float(np.mean(speeds))
        row["max_speed_mps"] = safe_float(np.max(speeds))
    else:
        row["mean_speed_mps"] = ""
        row["max_speed_mps"] = ""

    issues = []
    if missing:
        issues.append("missing_" + "|".join(missing))
    if inconsistent:
        issues.append("length_mismatch_" + "|".join(inconsistent))
    if non_finite:
        issues.append("non_finite_" + "|".join(non_finite))
    if frame_count and not row["timestamps_monotonic"]:
        issues.append("non_monotonic_timestamps")
    if video_path.stat().st_size <= 0:
        issues.append("empty_video")

    row["usable"] = not issues and frame_count > 0
    row["issue"] = "ok" if row["usable"] else ";".join(issues) or "no_frames"
    return row


def build_sequence_rows(
    segment_rows,
    chunk_parent: Path,
    route_splits,
    history_len: int,
    prediction_len: int,
    stride: int,
):
    sequence_rows = []
    for row in segment_rows:
        if not row["usable"]:
            continue

        frame_count = int(row["frame_count"])
        total_length = history_len + prediction_len
        if frame_count < total_length:
            continue

        segment_path = chunk_parent / row["segment_path"]
        arrays = load_pose_arrays(segment_path)
        speed, _ = scalar_motion_features(arrays["frame_times"], arrays["frame_velocities"])

        for history_start in range(0, frame_count - total_length + 1, stride):
            history_end = history_start + history_len - 1
            target_start = history_end + 1
            target_end = target_start + prediction_len - 1
            state_history, future_target = extract_motion_window(
                arrays, history_start, history_end, target_start, target_end
            )
            sequence_id = (
                f"{row['chunk']}__{row['route_id']}__segment_{row['segment_id']}__"
                f"history_{history_start}_{history_end}__target_{target_start}_{target_end}"
            )
            sequence_rows.append(
                {
                    "sequence_id": sequence_id,
                    "split": route_splits[row["route_id"]],
                    "chunk": row["chunk"],
                    "route_id": row["route_id"],
                    "segment_id": row["segment_id"],
                    "segment_path": row["segment_path"],
                    "video_path": row["video_path"],
                    "history_start_frame": history_start,
                    "history_end_frame": history_end,
                    "target_start_frame": target_start,
                    "target_end_frame": target_end,
                    "history_len": history_len,
                    "prediction_len": prediction_len,
                    "reference_time_s": safe_float(arrays["frame_times"][history_end]),
                    "mean_history_speed_mps": safe_float(np.mean(speed[history_start : history_end + 1])),
                    "target_end_longitudinal_m": safe_float(future_target[-1, 0]),
                    "target_end_lateral_m": safe_float(future_target[-1, 1]),
                    "target_end_speed_mps": safe_float(future_target[-1, 2]),
                    "target_status": "future_motion_available",
                }
            )
    return sequence_rows


def write_csv(path: Path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def calculate_normalisation_stats(sequence_rows, chunk_parent: Path):
    state_sum = np.zeros(len(STATE_FEATURE_NAMES), dtype=np.float64)
    state_square_sum = np.zeros_like(state_sum)
    target_sum = np.zeros(len(TARGET_FEATURE_NAMES), dtype=np.float64)
    target_square_sum = np.zeros_like(target_sum)
    state_count = 0
    target_count = 0
    cached_path = None
    arrays = None

    for row in sequence_rows:
        if row["split"] != "train":
            continue
        segment_path = chunk_parent / row["segment_path"]
        if segment_path != cached_path:
            arrays = load_pose_arrays(segment_path)
            cached_path = segment_path
        state, target = extract_motion_window(
            arrays,
            int(row["history_start_frame"]),
            int(row["history_end_frame"]),
            int(row["target_start_frame"]),
            int(row["target_end_frame"]),
        )
        state_sum += state.sum(axis=0)
        state_square_sum += np.square(state, dtype=np.float64).sum(axis=0)
        target_sum += target.sum(axis=0)
        target_square_sum += np.square(target, dtype=np.float64).sum(axis=0)
        state_count += len(state)
        target_count += len(target)

    def finish(total, square_total, count, names):
        mean = total / count
        variance = np.maximum(square_total / count - np.square(mean), 1e-12)
        return {
            "feature_names": list(names),
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "sample_frames": count,
        }

    return {
        "state_history": finish(state_sum, state_square_sum, state_count, STATE_FEATURE_NAMES),
        "future_target": finish(target_sum, target_square_sum, target_count, TARGET_FEATURE_NAMES),
        "calculated_from_split": "train",
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare comma2k19 for CNN-LSTM prediction.")
    parser.add_argument("--chunk", default=str(PROJECT_ROOT / "comma2k19" / "Chunk_1"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--history-len", type=int, default=30)
    parser.add_argument("--prediction-len", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()

    chunk_path = Path(args.chunk).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else PREPARATION_DIR / f"{chunk_path.name.lower()}_results"
    )
    if not chunk_path.exists():
        raise FileNotFoundError(f"Chunk path does not exist: {chunk_path}")
    if min(args.history_len, args.prediction_len, args.stride) <= 0:
        raise ValueError("History length, prediction length and stride must be positive")
    if args.validation_ratio <= 0 or args.test_ratio <= 0:
        raise ValueError("Validation and test ratios must be positive")
    if args.validation_ratio + args.test_ratio >= 1:
        raise ValueError("Validation and test ratios must leave a training split")

    video_paths = sorted(chunk_path.rglob("video.hevc"), key=segment_sort_key)
    segment_rows = [inspect_segment(video_path, chunk_path) for video_path in video_paths]
    route_splits = assign_route_splits(
        [row["route_id"] for row in segment_rows],
        args.validation_ratio,
        args.test_ratio,
        args.split_seed,
    )
    for row in segment_rows:
        row["split"] = route_splits[row["route_id"]]

    sequence_rows = build_sequence_rows(
        segment_rows,
        chunk_path.parent,
        route_splits,
        args.history_len,
        args.prediction_len,
        args.stride,
    )

    slug = chunk_path.name.lower()
    manifest_path = output_dir / f"comma2k19_{slug}_manifest.csv"
    sequence_path = output_dir / f"comma2k19_{slug}_sequence_index.csv"
    route_split_path = output_dir / f"comma2k19_{slug}_route_splits.csv"
    stats_path = output_dir / f"comma2k19_{slug}_normalisation.json"
    summary_path = output_dir / f"comma2k19_{slug}_summary.json"

    write_csv(manifest_path, segment_rows)
    write_csv(sequence_path, sequence_rows)
    write_csv(
        route_split_path,
        [{"route_id": route, "split": split} for route, split in sorted(route_splits.items())],
    )
    for split in ("train", "validation", "test"):
        write_csv(
            output_dir / "splits" / f"comma2k19_{slug}_{split}.csv",
            [row for row in sequence_rows if row["split"] == split],
            fieldnames=list(sequence_rows[0].keys()) if sequence_rows else None,
        )

    stats = calculate_normalisation_stats(sequence_rows, chunk_path.parent)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    sequence_counts = Counter(row["split"] for row in sequence_rows)
    route_counts = Counter(route_splits.values())
    usable_segments = sum(bool(row["usable"]) for row in segment_rows)
    summary = {
        "chunk": chunk_path.name,
        "chunk_path": str(chunk_path),
        "routes": len(route_splits),
        "segments": len(segment_rows),
        "usable_segments": usable_segments,
        "history_len_frames": args.history_len,
        "prediction_len_frames": args.prediction_len,
        "stride_frames": args.stride,
        "estimated_history_seconds": safe_float(args.history_len / 20.0),
        "estimated_prediction_seconds": safe_float(args.prediction_len / 20.0),
        "route_counts": dict(route_counts),
        "sequence_counts": dict(sequence_counts),
        "sequence_samples": len(sequence_rows),
        "manifest_path": str(manifest_path),
        "sequence_index_path": str(sequence_path),
        "route_split_path": str(route_split_path),
        "normalisation_path": str(stats_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
