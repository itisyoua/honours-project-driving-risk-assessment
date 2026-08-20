from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from .comma2k19_utils import read_csv_rows
    from .comma2k19_manifest import write_csv
except ImportError:
    from comma2k19_utils import read_csv_rows
    from comma2k19_manifest import write_csv


PREPARATION_DIR = Path(__file__).resolve().parent


def result_directory_sort_key(path: Path):
    match = re.search(r"chunk_(\d+)_results$", path.name)
    return int(match.group(1)) if match else path.name


def merge_stats(sections):
    counts = np.asarray([section["sample_frames"] for section in sections], dtype=np.float64)
    means = np.asarray([section["mean"] for section in sections], dtype=np.float64)
    stds = np.asarray([section["std"] for section in sections], dtype=np.float64)
    total = int(counts.sum())
    mean = (means * counts[:, None]).sum(axis=0) / total
    second_moment = ((np.square(stds) + np.square(means)) * counts[:, None]).sum(axis=0) / total
    std = np.sqrt(np.maximum(second_moment - np.square(mean), 1e-12))
    return {
        "feature_names": sections[0]["feature_names"],
        "mean": mean.tolist(),
        "std": std.tolist(),
        "sample_frames": total,
    }


def main():
    detected_result_dirs = sorted(
        PREPARATION_DIR.glob("chunk_*_results"), key=result_directory_sort_key
    )
    parser = argparse.ArgumentParser(description="Combine prepared comma2k19 chunk indexes.")
    parser.add_argument(
        "--result-dirs",
        nargs="+",
        default=[str(path) for path in detected_result_dirs],
    )
    parser.add_argument("--output-dir", default=str(PREPARATION_DIR / "combined_results"))
    args = parser.parse_args()

    manifests = []
    sequences = []
    route_rows = []
    stats_documents = []
    summaries = []
    seen_routes = {}

    for result_dir_value in args.result_dirs:
        result_dir = Path(result_dir_value)
        summary_candidates = sorted(result_dir.glob("comma2k19_chunk_*_summary.json"))
        if len(summary_candidates) != 1:
            raise ValueError(f"Expected one chunk summary in {result_dir}")
        summary = json.loads(summary_candidates[0].read_text(encoding="utf-8"))
        slug = summary["chunk"].lower()
        summaries.append(summary)
        manifests.extend(read_csv_rows(result_dir / f"comma2k19_{slug}_manifest.csv"))
        sequences.extend(read_csv_rows(result_dir / f"comma2k19_{slug}_sequence_index.csv"))
        chunk_routes = read_csv_rows(result_dir / f"comma2k19_{slug}_route_splits.csv")
        for row in chunk_routes:
            existing = seen_routes.get(row["route_id"])
            if existing and existing != row["split"]:
                raise ValueError(f"Route assigned to two splits: {row['route_id']}")
            if not existing:
                seen_routes[row["route_id"]] = row["split"]
                route_rows.append(row)
        stats_documents.append(
            json.loads((result_dir / f"comma2k19_{slug}_normalisation.json").read_text(encoding="utf-8"))
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "comma2k19_combined_manifest.csv"
    sequence_path = output_dir / "comma2k19_combined_sequence_index.csv"
    route_path = output_dir / "comma2k19_combined_route_splits.csv"
    stats_path = output_dir / "comma2k19_combined_normalisation.json"
    summary_path = output_dir / "comma2k19_combined_summary.json"

    write_csv(manifest_path, manifests)
    write_csv(sequence_path, sequences)
    write_csv(route_path, sorted(route_rows, key=lambda row: row["route_id"]))
    for split in ("train", "validation", "test"):
        write_csv(
            output_dir / "splits" / f"comma2k19_combined_{split}.csv",
            [row for row in sequences if row["split"] == split],
            fieldnames=list(sequences[0].keys()),
        )

    combined_stats = {
        "state_history": merge_stats([item["state_history"] for item in stats_documents]),
        "future_target": merge_stats([item["future_target"] for item in stats_documents]),
        "calculated_from_split": "train",
        "source_chunks": [summary["chunk"] for summary in summaries],
    }
    stats_path.write_text(json.dumps(combined_stats, indent=2) + "\n", encoding="utf-8")

    summary = {
        "chunks": [item["chunk"] for item in summaries],
        "chunk_paths": [item["chunk_path"] for item in summaries],
        "routes": len(route_rows),
        "segments": len(manifests),
        "usable_segments": sum(row["usable"].lower() == "true" for row in manifests),
        "history_len_frames": int(sequences[0]["history_len"]),
        "prediction_len_frames": int(sequences[0]["prediction_len"]),
        "route_counts": dict(Counter(row["split"] for row in route_rows)),
        "sequence_counts": dict(Counter(row["split"] for row in sequences)),
        "sequence_samples": len(sequences),
        "manifest_path": str(manifest_path),
        "sequence_index_path": str(sequence_path),
        "route_split_path": str(route_path),
        "normalisation_path": str(stats_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
