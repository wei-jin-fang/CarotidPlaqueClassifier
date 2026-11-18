"""Lightweight multilayer perceptron for numerical tabular features."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class TinyMLPBinaryClassifier(nn.Module):
    """Small MLP intended for low-sample clinical tabular data."""

    def __init__(
        self,
        num_features: int,
        hidden_dims: Sequence[int] = (32, 16),
        dropouts: Sequence[float] = (0.15, 0.10),
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if num_features < 1:
            raise ValueError("num_features must be positive")
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")
        if len(hidden_dims) != len(dropouts):
            raise ValueError("hidden_dims and dropouts must have equal length")
        if any(hidden_dim < 1 for hidden_dim in hidden_dims):
            raise ValueError("Every hidden dimension must be positive")
        if any(not 0 <= dropout < 1 for dropout in dropouts):
            raise ValueError("Every dropout must be in [0, 1)")

        layers: list[nn.Module] = []
        input_dim = num_features
        for hidden_dim, dropout in zip(hidden_dims, dropouts):
            layers.append(nn.Linear(input_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)
        self.num_features = num_features
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(
                f"Expected [batch, features], received shape={tuple(features.shape)}"
            )
        if features.shape[-1] != self.num_features:
            raise ValueError(
                f"Model expects {self.num_features} features, "
                f"received {features.shape[-1]}"
            )
        return self.network(features).squeeze(-1)
