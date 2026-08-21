import math

import torch

from driving_algorithm.training.losses import LossWeights, trajectory_loss


def test_perfect_prediction_has_zero_loss():
    target = torch.randn(2, 20, 5)
    mask = torch.ones(2, 20, dtype=torch.bool)

    losses = trajectory_loss(target.clone(), target, mask, LossWeights())

    for value in losses.values():
        torch.testing.assert_close(value, torch.tensor(0.0))


def test_masked_target_corruption_does_not_change_loss():
    prediction = torch.zeros(1, 20, 5)
    target = torch.zeros_like(prediction)
    mask = torch.ones(1, 20, dtype=torch.bool)
    mask[:, -1] = False
    target[:, -1] = 1000.0

    losses = trajectory_loss(prediction, target, mask, LossWeights())

    torch.testing.assert_close(losses["total"], torch.tensor(0.0))


def test_heading_loss_wraps_across_pi_boundary():
    prediction = torch.zeros(1, 20, 5)
    target = torch.zeros_like(prediction)
    prediction[:, :, 4] = math.pi - 0.05
    target[:, :, 4] = -math.pi + 0.05
    mask = torch.ones(1, 20, dtype=torch.bool)

    losses = trajectory_loss(prediction, target, mask, LossWeights())

    assert 0.0 < losses["heading"].item() < 0.01

