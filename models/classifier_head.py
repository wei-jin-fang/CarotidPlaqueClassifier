from __future__ import annotations

from torch import nn


def _dropout_or_identity(dropout: float) -> nn.Module:
    return nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()


class CommonClassifierHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"classifier_hidden_dim must be > 0, got {hidden_dim}.")

        self.layers = nn.Sequential(
            nn.BatchNorm1d(in_features),
            _dropout_or_identity(dropout),
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            _dropout_or_identity(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.layers(x)
