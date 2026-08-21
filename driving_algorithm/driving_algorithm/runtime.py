from __future__ import annotations

import torch


def select_device(preference: str = "auto") -> torch.device:
    preference = preference.lower()
    if preference == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if preference == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return torch.device("mps")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if preference == "cpu":
        return torch.device("cpu")
    raise ValueError("device preference must be auto, cpu, mps or cuda")
