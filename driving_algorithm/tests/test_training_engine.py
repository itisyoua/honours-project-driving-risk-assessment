import math

import torch
from torch.utils.data import DataLoader, Dataset

from driving_algorithm.models.cnn_lstm import (
    CNNLSTMConfig,
    CNNLSTMTrajectoryPredictor,
)
from driving_algorithm.training.engine import (
    evaluate_model,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)


class TinyDrivingDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        return {
            "image": torch.randn(3, 64, 64, generator=generator),
            "state_history": torch.randn(16, 8, generator=generator),
            "future_target": torch.randn(20, 5, generator=generator),
            "history_mask": torch.ones(16, dtype=torch.bool),
            "future_mask": torch.ones(20, dtype=torch.bool),
            "source": "waymo_e2e",
            "scene_type": "go_straight",
        }


def make_model():
    return CNNLSTMTrajectoryPredictor(
        CNNLSTMConfig(
            image_feature_dim=16,
            state_feature_dim=8,
            lstm_hidden_dim=12,
            fusion_hidden_dim=16,
            dropout=0.0,
            pretrained_backbone=False,
            freeze_backbone=True,
        )
    )


def test_training_step_evaluation_and_checkpoint_round_trip(tmp_path):
    loader = DataLoader(TinyDrivingDataset(), batch_size=2)
    model = make_model()
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-3,
    )
    original = model.trajectory_head[-1].weight.detach().clone()

    losses = train_one_epoch(model, loader, optimizer, torch.device("cpu"))
    metrics = evaluate_model(model, loader, torch.device("cpu"))

    assert math.isfinite(losses["total"])
    assert not torch.equal(original, model.trajectory_head[-1].weight)
    assert metrics["overall"]["samples"] == 2
    assert metrics["by_source"]["waymo_e2e"]["samples"] == 2
    assert metrics["by_scene_type"]["go_straight"]["samples"] == 2

    checkpoint = tmp_path / "baseline.pt"
    saved = model.trajectory_head[-1].weight.detach().clone()
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        epoch=1,
        manifest_fingerprint="abc123",
    )
    with torch.no_grad():
        model.trajectory_head[-1].weight.zero_()
    payload = load_checkpoint(checkpoint, model, optimizer, map_location="cpu")

    torch.testing.assert_close(model.trajectory_head[-1].weight, saved)
    assert payload["epoch"] == 1
    assert payload["manifest_fingerprint"] == "abc123"
