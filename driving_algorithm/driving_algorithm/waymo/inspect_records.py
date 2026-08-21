from __future__ import annotations

import argparse
import io
import json
from collections import Counter
from pathlib import Path

from PIL import Image
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.protos import end_to_end_driving_data_pb2

from .records import front_image_bytes, parse_e2e_record, split_frame_name
from .tfrecord import iter_tfrecord


def _string_counts(counter: Counter) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter, key=str)}


def inspect_shards(paths, report_path: Path) -> dict:
    shard_paths = [Path(path) for path in paths]
    if not shard_paths:
        raise ValueError("at least one TFRecord shard is required")

    frame_ids: Counter = Counter()
    route_ids: Counter = Counter()
    camera_names: Counter = Counter()
    image_dimensions: Counter = Counter()
    past_lengths: Counter = Counter()
    future_lengths: Counter = Counter()
    intent_counts: Counter = Counter()
    front_camera_records = 0
    records = 0
    malformed_records = 0
    zero_timestamp_records = 0
    complete_past_records = 0
    complete_future_records = 0

    for shard_path in shard_paths:
        for payload in iter_tfrecord(shard_path):
            try:
                record = parse_e2e_record(payload)
                frame_id = record.frame.context.name
                timestamp = int(record.frame.timestamp_micros)
                if not frame_id or timestamp < 0:
                    raise ValueError("record is missing frame ID or has a negative timestamp")

                records += 1
                frame_ids[frame_id] += 1
                route_id, _ = split_frame_name(frame_id)
                route_ids[route_id] += 1
                if timestamp == 0:
                    zero_timestamp_records += 1
                past_lengths[len(record.past_states.pos_x)] += 1
                future_lengths[len(record.future_states.pos_x)] += 1
                if all(
                    len(getattr(record.past_states, field)) == 16
                    for field in (
                        "pos_x",
                        "pos_y",
                        "vel_x",
                        "vel_y",
                        "accel_x",
                        "accel_y",
                    )
                ):
                    complete_past_records += 1
                if all(
                    len(getattr(record.future_states, field)) == 20
                    for field in ("pos_x", "pos_y")
                ):
                    complete_future_records += 1
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

    duplicate_frame_ids = sum(count - 1 for count in frame_ids.values() if count > 1)
    compatible = (
        records > 0
        and malformed_records == 0
        and duplicate_frame_ids == 0
        and front_camera_records == records
        and complete_past_records == records
        and complete_future_records == records
    )

    report = {
        "files": [
            {"path": str(path), "bytes": path.stat().st_size}
            for path in shard_paths
        ],
        "records": records,
        "routes": len(route_ids),
        "unique_frame_ids": len(frame_ids),
        "duplicate_frame_ids": duplicate_frame_ids,
        "zero_timestamp_records": zero_timestamp_records,
        "front_camera_records": front_camera_records,
        "camera_name_counts": _string_counts(camera_names),
        "image_dimensions": _string_counts(image_dimensions),
        "past_length_counts": _string_counts(past_lengths),
        "future_length_counts": _string_counts(future_lengths),
        "intent_counts": _string_counts(intent_counts),
        "complete_past_state_records": complete_past_records,
        "complete_future_target_records": complete_future_records,
        "malformed_records": malformed_records,
        "compatible_with_cnn_lstm_sample": compatible,
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
