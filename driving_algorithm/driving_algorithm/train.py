from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from driving_algorithm.data.statistics import compute_state_statistics
from driving_algorithm.data.waymo_dataset import WaymoE2EDataset
from driving_algorithm.models.cnn_lstm import CNNLSTMConfig, CNNLSTMTrajectoryPredictor
from driving_algorithm.runtime import select_device
from driving_algorithm.training.engine import (
    manifest_fingerprint,
    save_checkpoint,
    train_one_epoch,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CNN-LSTM baseline.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--train-backbone", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")

    device = select_device(args.device)
    statistics = compute_state_statistics(args.manifest, args.data_root)
    dataset = WaymoE2EDataset(args.manifest, args.data_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    config = CNNLSTMConfig(
        pretrained_backbone=args.pretrained_backbone,
        freeze_backbone=not args.train_backbone,
    )
    model = CNNLSTMTrajectoryPredictor(
        config, state_mean=statistics.mean, state_std=statistics.std
    ).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    fingerprint = manifest_fingerprint(args.manifest)
    for epoch in range(1, args.epochs + 1):
        losses = train_one_epoch(
            model,
            loader,
            optimizer,
            device,
            max_batches=args.max_batches,
        )
        save_checkpoint(args.checkpoint, model, optimizer, epoch, fingerprint)
        print(json.dumps({"epoch": epoch, "device": str(device), "losses": losses}))


if __name__ == "__main__":
    main()
