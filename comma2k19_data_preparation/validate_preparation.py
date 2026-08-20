from __future__ import annotations

import json
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from .comma2k19_dataset import Comma2k19Dataset
    from .comma2k19_utils import read_csv_rows
except ImportError:
    from comma2k19_dataset import Comma2k19Dataset
    from comma2k19_utils import read_csv_rows


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPARATION_DIR = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    default_index = PREPARATION_DIR / "combined_results" / "comma2k19_combined_sequence_index.csv"
    if not default_index.exists():
        default_index = PREPARATION_DIR / "chunk_1_results" / "comma2k19_chunk_1_sequence_index.csv"
    parser = argparse.ArgumentParser(description="Validate prepared comma2k19 indexes and tensors.")
    parser.add_argument("--index-csv", default=str(default_index))
    args = parser.parse_args()

    index_path = Path(args.index_csv)
    split_path = Path(str(index_path).replace("_sequence_index.csv", "_route_splits.csv"))
    stats_path = Path(str(index_path).replace("_sequence_index.csv", "_normalisation.json"))
    rows = read_csv_rows(index_path)
    route_rows = read_csv_rows(split_path)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    route_to_split = {row["route_id"]: row["split"] for row in route_rows}
    require(len(route_to_split) == len(route_rows), "Duplicate route IDs in route split file")
    require(set(route_to_split.values()) == {"train", "validation", "test"}, "Missing split")
    require(all(route_to_split[row["route_id"]] == row["split"] for row in rows), "Route leakage")

    data_root = PROJECT_ROOT / "comma2k19"
    for row in rows:
        history_start = int(row["history_start_frame"])
        history_end = int(row["history_end_frame"])
        target_start = int(row["target_start_frame"])
        target_end = int(row["target_end_frame"])
        require(history_end - history_start + 1 == int(row["history_len"]), "Unexpected history length")
        require(target_end - target_start + 1 == int(row["prediction_len"]), "Unexpected prediction length")
        require(history_end + 1 == target_start, "History and target are not consecutive")
        require((data_root / row["video_path"]).exists(), "Missing indexed video")

    for section in ("state_history", "future_target"):
        mean = np.asarray(stats[section]["mean"], dtype=np.float64)
        std = np.asarray(stats[section]["std"], dtype=np.float64)
        require(np.all(np.isfinite(mean)), f"Non-finite {section} mean")
        require(np.all(np.isfinite(std)) and np.all(std > 0), f"Invalid {section} std")

    decoded_samples = {}
    for split in ("train", "validation", "test"):
        dataset = Comma2k19Dataset(
            index_csv=index_path,
            normalisation_json=stats_path,
            split=split,
            max_samples=1,
        )
        sample = dataset[0]
        require(tuple(sample["frames"].shape) == (30, 3, 224, 224), "Invalid frame tensor")
        require(tuple(sample["state_history"].shape) == (30, 8), "Invalid state tensor")
        require(tuple(sample["future_target"].shape) == (20, 5), "Invalid target tensor")
        require(bool(sample["frames"].isfinite().all()), "Non-finite image tensor")
        require(bool(sample["state_history"].isfinite().all()), "Non-finite state tensor")
        require(bool(sample["future_target"].isfinite().all()), "Non-finite target tensor")
        decoded_samples[split] = sample["sequence_id"]

    report = {
        "status": "passed",
        "sequence_count": len(rows),
        "sequence_counts": dict(Counter(row["split"] for row in rows)),
        "route_counts": dict(Counter(route_to_split.values())),
        "route_leakage": False,
        "tensor_shapes": {
            "frames": [30, 3, 224, 224],
            "state_history": [30, 8],
            "future_target": [20, 5],
        },
        "decoded_sample_per_split": decoded_samples,
    }
    output_path = Path(str(index_path).replace("_sequence_index.csv", "_validation_report.json"))
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
