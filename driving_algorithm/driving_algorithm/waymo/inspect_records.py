from __future__ import annotations

import argparse
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.protos import end_to_end_driving_data_pb2

from .records import front_image_bytes, parse_e2e_record
from .tfrecord import iter_tfrecord


def _string_counts(counter: Counter) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter, key=str)}


def _has_history_window(timestamps: list[int]) -> bool:
    ordered = np.asarray(sorted(timestamps), dtype=np.int64)
    if ordered.size < 16:
        return False
    for start in range(ordered.size - 15):
        gaps = np.diff(ordered[start : start + 16])
        if np.all(np.abs(gaps - 250_000) <= 25_000):
            return True
    return False


def inspect_shards(paths, report_path: Path) -> dict:
    shard_paths = [Path(path) for path in paths]
    if not shard_paths:
        raise ValueError("at least one TFRecord shard is required")

    run_timestamps: dict[str, list[int]] = defaultdict(list)
    camera_names: Counter = Counter()
    image_dimensions: Counter = Counter()
    past_lengths: Counter = Counter()
    future_lengths: Counter = Counter()
    intent_counts: Counter = Counter()
    front_camera_records = 0
    records = 0
    malformed_records = 0

    for shard_path in shard_paths:
        for payload in iter_tfrecord(shard_path):
            try:
                record = parse_e2e_record(payload)
                run_id = record.frame.context.name
                timestamp = int(record.frame.timestamp_micros)
                if not run_id or timestamp < 0:
                    raise ValueError("record is missing run ID or timestamp")

                records += 1
                run_timestamps[run_id].append(timestamp)
                past_lengths[len(record.past_states.pos_x)] += 1
                future_lengths[len(record.future_states.pos_x)] += 1
                intent_name = end_to_end_driving_data_pb2.EgoIntent.Intent.Name(
                    record.intent
                )
                intent_counts[intent_name] += 1

                front_payload = front_image_bytes(record)
                for camera in record.frame.images:
                    camera_names[dataset_pb2.CameraName.Name.Name(camera.name)] += 1
                if front_payload is not None:
                    front_camera_records += 1
                    with Image.open(io.BytesIO(front_payload)) as image:
                        image_dimensions[f"{image.width}x{image.height}"] += 1
            except Exception:
                malformed_records += 1

    all_gaps = []
    for timestamps in run_timestamps.values():
        if len(timestamps) > 1:
            all_gaps.extend(np.diff(np.asarray(sorted(timestamps), dtype=np.int64)))
    if all_gaps:
        gap_array = np.asarray(all_gaps, dtype=np.int64)
        gap_summary = {
            "count": int(gap_array.size),
            "min": int(gap_array.min()),
            "median": int(np.median(gap_array)),
            "max": int(gap_array.max()),
        }
    else:
        gap_summary = {"count": 0, "min": None, "median": None, "max": None}

    complete_motion = (
        records > 0
        and past_lengths == Counter({16: records})
        and future_lengths == Counter({20: records})
    )
    compatible = (
        malformed_records == 0
        and complete_motion
        and any(_has_history_window(values) for values in run_timestamps.values())
    )

    report = {
        "files": [
            {"path": str(path), "bytes": path.stat().st_size}
            for path in shard_paths
        ],
        "records": records,
        "runs": len(run_timestamps),
        "front_camera_records": front_camera_records,
        "camera_name_counts": _string_counts(camera_names),
        "image_dimensions": _string_counts(image_dimensions),
        "past_length_counts": _string_counts(past_lengths),
        "future_length_counts": _string_counts(future_lengths),
        "intent_counts": _string_counts(intent_counts),
        "timestamp_gap_micros": gap_summary,
        "malformed_records": malformed_records,
        "compatible_with_16_frame_history": compatible,
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Waymo E2E TFRecord shards.")
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_shards(args.shards, args.report), indent=2))


if __name__ == "__main__":
    main()
