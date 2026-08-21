from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from pathlib import Path

import torch

from driving_algorithm.evaluation.metrics import trajectory_metrics

from .losses import LossWeights, trajectory_loss


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_weights: LossWeights | None = None,
    max_batches: int | None = None,
    gradient_clip_norm: float = 5.0,
) -> dict[str, float]:
    model.train()
    totals = defaultdict(float)
    samples = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(
            batch["image"], batch["state_history"], batch["history_mask"]
        )
        losses = trajectory_loss(
            prediction, batch["future_target"], batch["future_mask"], loss_weights
        )
        losses["total"].backward()
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

        batch_size = int(prediction.shape[0])
        samples += batch_size
        for name, value in losses.items():
            totals[name] += float(value.detach().cpu()) * batch_size
    if samples == 0:
        raise ValueError("training loader produced no batches")
    return {name: value / samples for name, value in totals.items()}


def evaluate_model(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict:
    model.eval()
    predictions = []
    targets = []
    masks = []
    sources = []
    scene_types = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = _move_batch(batch, device)
            prediction = model(
                batch["image"], batch["state_history"], batch["history_mask"]
            )
            predictions.append(prediction.cpu())
            targets.append(batch["future_target"].cpu())
            masks.append(batch["future_mask"].cpu())
            sources.extend(list(batch.get("source", ["unknown"] * len(prediction))))
            scene_types.extend(
                list(batch.get("scene_type", ["unknown"] * len(prediction)))
            )
    if not predictions:
        raise ValueError("evaluation loader produced no batches")

    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    mask = torch.cat(masks)

    def grouped_metrics(labels):
        groups = {}
        for label in sorted(set(labels)):
            indexes = torch.tensor(
                [index for index, value in enumerate(labels) if value == label],
                dtype=torch.long,
            )
            groups[label] = trajectory_metrics(
                prediction[indexes], target[indexes], mask[indexes]
            )
        return groups

    return {
        "overall": trajectory_metrics(prediction, target, mask),
        "by_source": grouped_metrics(sources),
        "by_scene_type": grouped_metrics(scene_types),
    }


def manifest_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_checkpoint(path: Path, map_location="cpu") -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    manifest_fingerprint: str,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "model_config": model.config.as_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "manifest_fingerprint": manifest_fingerprint,
        "state_statistics": {
            "mean": model.state_mean.detach().cpu().tolist(),
            "std": model.state_std.detach().cpu().tolist(),
        },
    }
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location="cpu",
) -> dict:
    payload = _read_checkpoint(Path(path), map_location=map_location)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def checkpoint_metadata(path: Path) -> dict:
    return _read_checkpoint(Path(path), map_location="cpu")
