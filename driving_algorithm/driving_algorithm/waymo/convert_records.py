from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from waymo_open_dataset.protos import end_to_end_driving_data_pb2

from driving_algorithm.data.motion import derive_future_target, derive_state_history

from .records import front_image_bytes, parse_e2e_record, split_frame_name
from .tfrecord import iter_tfrecord


MANIFEST_FIELDS = (
    "sample_id",
    "frame_id",
    "route_id",
    "split",
    "scene_type",
    "image_path",
    "motion_path",
    "source_shard",
)


@dataclass(frozen=True)
class ConversionSummary:
    records: int
    samples: int
    routes: int
    rejected: int
    manifest_path: Path


def _safe_stem(frame_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", frame_id).strip("._")
    digest = hashlib.sha256(frame_id.encode("utf-8")).hexdigest()[:10]
    return f"{readable[:100]}-{digest}" if readable else digest


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_write_motion(path: Path, **arrays) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(output, **arrays)
    os.replace(temporary, path)


def _intent_name(intent: int) -> str:
    proto_name = end_to_end_driving_data_pb2.EgoIntent.Intent.Name(intent)
    return {
        "GO_STRAIGHT": "go_straight",
        "GO_LEFT": "go_left",
        "GO_RIGHT": "go_right",
    }.get(proto_name, "unknown")


def _motion_arrays(record) -> dict[str, np.ndarray]:
    state = derive_state_history(
        record.past_states.pos_x,
        record.past_states.pos_y,
        record.past_states.vel_x,
        record.past_states.vel_y,
        record.past_states.accel_x,
        record.past_states.accel_y,
        dt=0.25,
    )
    initial_velocity = np.array(
        [record.past_states.vel_x[-1], record.past_states.vel_y[-1]],
        dtype=np.float32,
    )
    target = derive_future_target(
        record.future_states.pos_x,
        record.future_states.pos_y,
        initial_velocity,
        dt=0.25,
    )
    return {
        "state_history": state,
        "future_target": target,
        "history_mask": np.ones(16, dtype=np.bool_),
        "future_mask": np.ones(20, dtype=np.bool_),
    }


def convert_shards(
    paths: Sequence[Path],
    output_root: Path,
    manifest_path: Path,
    split: str,
) -> ConversionSummary:
    if split not in {"train", "validation"}:
        raise ValueError("Waymo supervised conversion split must be train or validation")
    shard_paths = [Path(path) for path in paths]
    if not shard_paths:
        raise ValueError("at least one Waymo shard is required")

    output_root = Path(output_root)
    image_root = output_root / "images"
    motion_root = output_root / "motion"
    image_root.mkdir(parents=True, exist_ok=True)
    motion_root.mkdir(parents=True, exist_ok=True)

    rows = []
    routes = set()
    seen_frame_ids = set()
    records = 0
    rejected = 0
    for shard_path in shard_paths:
        for payload in iter_tfrecord(shard_path):
            records += 1
            try:
                record = parse_e2e_record(payload)
                frame_id = record.frame.context.name
                if not frame_id:
                    raise ValueError("missing frame ID")
                if frame_id in seen_frame_ids:
                    raise ValueError(f"duplicate frame ID: {frame_id}")
                front_jpeg = front_image_bytes(record)
                if front_jpeg is None:
                    raise ValueError(f"missing FRONT image: {frame_id}")
                motion = _motion_arrays(record)
            except (ValueError, IndexError):
                rejected += 1
                continue

            seen_frame_ids.add(frame_id)
            route_id, _ = split_frame_name(frame_id)
            routes.add(route_id)
            stem = _safe_stem(frame_id)
            image_relative = Path("images") / f"{stem}.jpg"
            motion_relative = Path("motion") / f"{stem}.npz"
            _atomic_write_bytes(output_root / image_relative, front_jpeg)
            _atomic_write_motion(output_root / motion_relative, **motion)
            rows.append(
                {
                    "sample_id": f"waymo_e2e:{frame_id}",
                    "frame_id": frame_id,
                    "route_id": route_id,
                    "split": split,
                    "scene_type": _intent_name(record.intent),
                    "image_path": image_relative.as_posix(),
                    "motion_path": motion_relative.as_posix(),
                    "source_shard": shard_path.name,
                }
            )

    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
    with temporary_manifest.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_manifest, manifest_path)
    return ConversionSummary(
        records=records,
        samples=len(rows),
        routes=len(routes),
        rejected=rejected,
        manifest_path=manifest_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Waymo E2E TFRecords.")
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("train", "validation"))
    args = parser.parse_args()
    summary = convert_shards(
        args.shards,
        output_root=args.output_root,
        manifest_path=args.manifest,
        split=args.split,
    )
    print(summary)


if __name__ == "__main__":
    main()
