from __future__ import annotations

import torch


def trajectory_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    future_mask: torch.Tensor,
) -> dict[str, float | int]:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must have matching [B, K, 5] shapes")
    if prediction.shape[-1] != 5 or future_mask.shape != prediction.shape[:2]:
        raise ValueError("trajectory features or future mask have invalid shapes")
    mask = future_mask.to(device=prediction.device, dtype=torch.bool)
    valid_per_sample = mask.sum(dim=1)
    if torch.any(valid_per_sample == 0):
        raise ValueError("every sample must have a valid future target")

    difference = prediction - target
    position_distance = torch.linalg.vector_norm(difference[..., :2], dim=-1)
    ade = position_distance[mask].mean()
    step_indexes = torch.arange(prediction.shape[1], device=prediction.device)
    last_indexes = torch.where(
        mask, step_indexes.unsqueeze(0), torch.full_like(mask, -1, dtype=torch.long)
    ).max(dim=1).values
    batch_indexes = torch.arange(prediction.shape[0], device=prediction.device)
    fde = position_distance[batch_indexes, last_indexes].mean()

    heading_difference = torch.atan2(
        torch.sin(difference[..., 4]), torch.cos(difference[..., 4])
    ).abs()
    metrics = {
        "ade": ade,
        "fde": fde,
        "x_rmse": torch.sqrt(torch.square(difference[..., 0][mask]).mean()),
        "y_rmse": torch.sqrt(torch.square(difference[..., 1][mask]).mean()),
        "speed_mae": difference[..., 2][mask].abs().mean(),
        "heading_mae": heading_difference[mask].mean(),
    }
    return {
        **{name: float(value.detach().cpu()) for name, value in metrics.items()},
        "samples": int(prediction.shape[0]),
        "valid_points": int(mask.sum().item()),
    }
