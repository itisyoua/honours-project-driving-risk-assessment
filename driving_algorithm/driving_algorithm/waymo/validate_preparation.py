from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from driving_algorithm.data.contracts import SequenceContract

from .convert_records import MANIFEST_FIELDS


def _resolve(data_root: Path, relative_path: str) -> Path:
    resolved = (data_root / relative_path).resolve()
    if os.path.commonpath((data_root, resolved)) != str(data_root):
        raise ValueError(f"path escapes data root: {relative_path}")
    return resolved


def validate_preparation(manifest_path: Path, data_root: Path) -> dict:
    manifest_path = Path(manifest_path)
    data_root = Path(data_root).resolve()
    with manifest_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
        columns = set(reader.fieldnames or ())

    errors = []
    missing_columns = sorted(set(MANIFEST_FIELDS) - columns)
    if missing_columns:
        errors.append({"code": "missing_columns", "columns": missing_columns})
        return {
            "samples": len(rows),
            "routes": 0,
            "split_counts": {},
            "intent_counts": {},
            "total_bytes": 0,
            "speed_mps": None,
            "errors": errors,
        }

    sample_counts = Counter(row["sample_id"] for row in rows)
    for sample_id, count in sorted(sample_counts.items()):
        if count > 1:
            errors.append(
                {"code": "duplicate_sample_id", "sample_id": sample_id, "count": count}
            )

    route_splits = defaultdict(set)
    for row in rows:
        route_splits[row["route_id"]].add(row["split"])
    for route_id, splits in sorted(route_splits.items()):
        if len(splits) > 1:
            errors.append(
                {
                    "code": "route_split_leakage",
                    "route_id": route_id,
                    "splits": sorted(splits),
                }
            )

    speeds = []
    measured_files = set()
    total_bytes = 0
    for row in rows:
        paths = {}
        missing = False
        for key in ("image_path", "motion_path"):
            relative_path = row[key]
            try:
                path = _resolve(data_root, relative_path)
            except ValueError as error:
                errors.append(
                    {
                        "code": "unsafe_path",
                        "sample_id": row["sample_id"],
                        "path": relative_path,
                        "detail": str(error),
                    }
                )
                missing = True
                continue
            paths[key] = path
            if not path.is_file():
                errors.append(
                    {
                        "code": "missing_file",
                        "sample_id": row["sample_id"],
                        "path": relative_path,
                    }
                )
                missing = True
            elif path not in measured_files:
                total_bytes += path.stat().st_size
                measured_files.add(path)
        if missing:
            continue

        try:
            with Image.open(paths["image_path"]) as image:
                image.verify()
            with np.load(paths["motion_path"], allow_pickle=False) as motion:
                state = motion["state_history"].astype(np.float32, copy=False)
                target = motion["future_target"].astype(np.float32, copy=False)
                history_mask = motion["history_mask"].astype(np.bool_, copy=False)
                future_mask = motion["future_mask"].astype(np.bool_, copy=False)
            SequenceContract.validate(
                {
                    "image": np.zeros((3, 224, 224), dtype=np.float32),
                    "state_history": state,
                    "future_target": target,
                    "history_mask": history_mask,
                    "future_mask": future_mask,
                    "sample_id": row["sample_id"],
                    "route_id": row["route_id"],
                    "source": "waymo_e2e",
                    "split": row["split"],
                    "scene_type": row["scene_type"],
                }
            )
            speeds.extend(state[:, 4].tolist())
        except Exception as error:
            errors.append(
                {
                    "code": "invalid_sample",
                    "sample_id": row["sample_id"],
                    "detail": str(error),
                }
            )

    speed_summary = None
    if speeds:
        speed_array = np.asarray(speeds, dtype=np.float64)
        speed_summary = {
            "min": float(speed_array.min()),
            "median": float(np.median(speed_array)),
            "max": float(speed_array.max()),
        }
    return {
        "samples": len(rows),
        "routes": len(route_splits),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "intent_counts": dict(
            sorted(Counter(row["scene_type"] for row in rows).items())
        ),
        "total_bytes": total_bytes,
        "speed_mps": speed_summary,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate converted Waymo E2E data.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = validate_preparation(args.manifest, args.data_root)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
