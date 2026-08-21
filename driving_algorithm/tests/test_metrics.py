import math

import torch

from driving_algorithm.evaluation.metrics import trajectory_metrics


def test_metrics_report_known_one_metre_longitudinal_error():
    target = torch.zeros(2, 20, 5)
    prediction = target.clone()
    prediction[:, :, 0] = 1.0
    mask = torch.ones(2, 20, dtype=torch.bool)

    metrics = trajectory_metrics(prediction, target, mask)

    assert metrics["ade"] == 1.0
    assert metrics["fde"] == 1.0
    assert metrics["x_rmse"] == 1.0
    assert metrics["y_rmse"] == 0.0


def test_metrics_ignore_masked_steps_and_wrap_heading():
    target = torch.zeros(1, 20, 5)
    prediction = target.clone()
    mask = torch.zeros(1, 20, dtype=torch.bool)
    mask[:, 0] = True
    prediction[:, 1:, :4] = 1000.0
    prediction[:, 0, 4] = math.pi - 0.05
    target[:, 0, 4] = -math.pi + 0.05

    metrics = trajectory_metrics(prediction, target, mask)

    assert metrics["ade"] == 0.0
    assert metrics["fde"] == 0.0
    assert abs(metrics["heading_mae"] - 0.1) < 1e-5
