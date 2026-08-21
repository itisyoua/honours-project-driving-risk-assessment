import torch
from torchvision.models import resnet18 as torchvision_resnet18

from driving_algorithm.models.cnn_lstm import (
    CNNLSTMConfig,
    CNNLSTMTrajectoryPredictor,
)


def make_model():
    config = CNNLSTMConfig(
        image_feature_dim=32,
        state_feature_dim=16,
        lstm_hidden_dim=24,
        fusion_hidden_dim=32,
        dropout=0.0,
        pretrained_backbone=False,
        freeze_backbone=True,
    )
    return CNNLSTMTrajectoryPredictor(
        config,
        state_mean=torch.arange(8, dtype=torch.float32),
        state_std=torch.full((8,), 2.0),
    )


def test_model_outputs_finite_trajectory_and_backpropagates():
    model = make_model()
    image = torch.randn(2, 3, 64, 64)
    state = torch.randn(2, 16, 8)
    mask = torch.ones(2, 16, dtype=torch.bool)

    prediction = model(image, state, mask)
    prediction.square().mean().backward()

    assert prediction.shape == (2, 20, 5)
    assert torch.isfinite(prediction).all()
    assert model.trajectory_head[-1].weight.grad is not None
    torch.testing.assert_close(model.state_mean, torch.arange(8, dtype=torch.float32))
    torch.testing.assert_close(model.state_std, torch.full((8,), 2.0))


def test_model_rejects_sample_with_no_valid_history():
    model = make_model()
    image = torch.zeros(1, 3, 64, 64)
    state = torch.zeros(1, 16, 8)
    mask = torch.zeros(1, 16, dtype=torch.bool)

    try:
        model(image, state, mask)
    except ValueError as error:
        assert "history" in str(error)
    else:
        raise AssertionError("expected empty history to be rejected")


def test_masked_history_positions_do_not_change_prediction():
    model = make_model().eval()
    image = torch.zeros(1, 3, 64, 64)
    valid_steps = torch.randn(3, 8)
    left_padded = torch.randn(1, 16, 8)
    right_padded = torch.randn(1, 16, 8)
    left_mask = torch.zeros(1, 16, dtype=torch.bool)
    right_mask = torch.zeros(1, 16, dtype=torch.bool)
    left_padded[0, -3:] = valid_steps
    right_padded[0, :3] = valid_steps
    left_mask[0, -3:] = True
    right_mask[0, :3] = True

    with torch.no_grad():
        left_prediction = model(image, left_padded, left_mask)
        right_prediction = model(image, right_padded, right_mask)

    torch.testing.assert_close(left_prediction, right_prediction)


def test_checkpoint_construction_can_skip_pretrained_download(monkeypatch):
    observed_weights = []

    def recording_resnet18(*, weights):
        observed_weights.append(weights)
        return torchvision_resnet18(weights=None)

    monkeypatch.setattr(
        "driving_algorithm.models.cnn_lstm.resnet18", recording_resnet18
    )

    model = CNNLSTMTrajectoryPredictor(
        CNNLSTMConfig(pretrained_backbone=True), load_backbone_weights=False
    )

    assert model.config.pretrained_backbone is True
    assert observed_weights == [None]
