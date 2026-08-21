from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from driving_algorithm.data.waymo_dataset import WaymoE2EDataset
from driving_algorithm.models.cnn_lstm import CNNLSTMConfig, CNNLSTMTrajectoryPredictor
from driving_algorithm.runtime import select_device
from driving_algorithm.training.engine import (
    checkpoint_metadata,
    evaluate_model,
    load_checkpoint,
    manifest_fingerprint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a CNN-LSTM checkpoint.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args()

    device = select_device(args.device)
    metadata = checkpoint_metadata(args.checkpoint)
    model = CNNLSTMTrajectoryPredictor(
        CNNLSTMConfig(**metadata["model_config"])
    ).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    dataset = WaymoE2EDataset(args.manifest, args.data_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    report = evaluate_model(model, loader, device, max_batches=args.max_batches)
    report["checkpoint_epoch"] = metadata["epoch"]
    report["device"] = str(device)
    report["manifest_matches_checkpoint"] = (
        manifest_fingerprint(args.manifest) == metadata["manifest_fingerprint"]
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
