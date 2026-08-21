from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class LossWeights:
    position: float = 1.0
    speed: float = 0.2
    acceleration: float = 0.1
    heading: float = 0.1


def _validate_inputs(
    prediction: torch.Tensor,
    target: torch.Tensor,
    future_mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must have matching [B, K, 5] shapes")
    if prediction.shape[-1] != 5:
        raise ValueError("trajectory feature dimension must be 5")
    if future_mask.shape != prediction.shape[:2]:
        raise ValueError("future_mask must have shape [B, K]")
    mask = future_mask.to(dtype=torch.bool)
    if not torch.any(mask):
        raise ValueError("future_mask must contain a valid target")
    return mask


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded_mask = mask
    while expanded_mask.ndim < values.ndim:
        expanded_mask = expanded_mask.unsqueeze(-1)
    expanded_mask = expanded_mask.expand_as(values)
    return values[expanded_mask].mean()


def trajectory_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    future_mask: torch.Tensor,
    weights: LossWeights | None = None,
) -> dict[str, torch.Tensor]:
    weights = weights or LossWeights()
    mask = _validate_inputs(prediction, target, future_mask)

    position = _masked_mean(
        F.smooth_l1_loss(prediction[..., :2], target[..., :2], reduction="none"),
        mask,
    )
    speed = _masked_mean(
        F.smooth_l1_loss(prediction[..., 2], target[..., 2], reduction="none"),
        mask,
    )
    acceleration = _masked_mean(
        F.smooth_l1_loss(prediction[..., 3], target[..., 3], reduction="none"),
        mask,
    )
    heading_difference = torch.atan2(
        torch.sin(prediction[..., 4] - target[..., 4]),
        torch.cos(prediction[..., 4] - target[..., 4]),
    )
    heading = _masked_mean(
        F.smooth_l1_loss(
            heading_difference, torch.zeros_like(heading_difference), reduction="none"
        ),
        mask,
    )
    total = (
        weights.position * position
        + weights.speed * speed
        + weights.acceleration * acceleration
        + weights.heading * heading
    )
    return {
        "total": total,
        "position": position,
        "speed": speed,
        "acceleration": acceleration,
        "heading": heading,
    }
