from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from driving_algorithm.data.contracts import FUTURE_STEPS, STATE_DIM, TARGET_DIM


@dataclass(frozen=True)
class CNNLSTMConfig:
    image_feature_dim: int = 128
    state_feature_dim: int = 64
    lstm_hidden_dim: int = 128
    lstm_layers: int = 1
    fusion_hidden_dim: int = 256
    dropout: float = 0.1
    pretrained_backbone: bool = False
    freeze_backbone: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


class CNNLSTMTrajectoryPredictor(nn.Module):
    """Fuse current visual context with recurrent ego-motion history."""

    def __init__(
        self,
        config: CNNLSTMConfig | None = None,
        state_mean=None,
        state_std=None,
    ) -> None:
        super().__init__()
        self.config = config or CNNLSTMConfig()
        if self.config.lstm_layers < 1:
            raise ValueError("lstm_layers must be positive")
        if not 0.0 <= self.config.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        weights = (
            ResNet18_Weights.DEFAULT if self.config.pretrained_backbone else None
        )
        backbone = resnet18(weights=weights)
        backbone_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.visual_backbone = backbone
        self.visual_projection = nn.Sequential(
            nn.Linear(backbone_features, self.config.image_feature_dim),
            nn.ReLU(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(STATE_DIM, self.config.state_feature_dim),
            nn.ReLU(),
        )
        self.state_lstm = nn.LSTM(
            input_size=self.config.state_feature_dim,
            hidden_size=self.config.lstm_hidden_dim,
            num_layers=self.config.lstm_layers,
            batch_first=True,
            dropout=self.config.dropout if self.config.lstm_layers > 1 else 0.0,
        )
        self.trajectory_head = nn.Sequential(
            nn.Linear(
                self.config.image_feature_dim + self.config.lstm_hidden_dim,
                self.config.fusion_hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(
                self.config.fusion_hidden_dim, FUTURE_STEPS * TARGET_DIM
            ),
        )

        mean = torch.as_tensor(
            torch.zeros(STATE_DIM) if state_mean is None else state_mean,
            dtype=torch.float32,
        )
        std = torch.as_tensor(
            torch.ones(STATE_DIM) if state_std is None else state_std,
            dtype=torch.float32,
        )
        if mean.shape != (STATE_DIM,) or std.shape != (STATE_DIM,):
            raise ValueError(f"state statistics must have shape ({STATE_DIM},)")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("state statistics must be finite")
        if torch.any(std <= 0):
            raise ValueError("state standard deviations must be positive")
        self.register_buffer("state_mean", mean.clone())
        self.register_buffer("state_std", std.clone())

        if self.config.freeze_backbone:
            for parameter in self.visual_backbone.parameters():
                parameter.requires_grad = False
            self.visual_backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_backbone:
            self.visual_backbone.eval()
        return self

    def forward(
        self,
        image: torch.Tensor,
        state_history: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [B, 3, H, W]")
        if state_history.ndim != 3 or state_history.shape[1:] != (16, STATE_DIM):
            raise ValueError("state_history must have shape [B, 16, 8]")
        if image.shape[0] != state_history.shape[0]:
            raise ValueError("image and state history batch sizes must match")
        if history_mask is None:
            history_mask = torch.ones(
                state_history.shape[:2], dtype=torch.bool, device=state_history.device
            )
        if history_mask.shape != state_history.shape[:2]:
            raise ValueError("history_mask must have shape [B, 16]")
        history_mask = history_mask.to(dtype=torch.bool)
        if torch.any(history_mask.sum(dim=1) == 0):
            raise ValueError("every sample must contain a valid history step")

        if self.config.freeze_backbone:
            with torch.no_grad():
                visual_backbone_features = self.visual_backbone(image)
        else:
            visual_backbone_features = self.visual_backbone(image)
        visual_features = self.visual_projection(visual_backbone_features)

        normalised_state = (state_history - self.state_mean) / self.state_std
        normalised_state = normalised_state * history_mask.unsqueeze(-1)
        encoded_state = self.state_encoder(normalised_state)
        recurrent_state, _ = self.state_lstm(encoded_state)
        step_indexes = torch.arange(16, device=state_history.device).unsqueeze(0)
        last_valid_indexes = torch.where(
            history_mask, step_indexes, torch.full_like(step_indexes, -1)
        ).max(dim=1).values
        batch_indexes = torch.arange(state_history.shape[0], device=state_history.device)
        history_features = recurrent_state[batch_indexes, last_valid_indexes]

        fused = torch.cat((visual_features, history_features), dim=-1)
        prediction = self.trajectory_head(fused)
        return prediction.view(-1, FUTURE_STEPS, TARGET_DIM)
