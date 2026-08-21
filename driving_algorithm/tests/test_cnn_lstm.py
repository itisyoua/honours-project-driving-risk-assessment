import torch

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
