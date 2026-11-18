"""A compact Transformer for numerical tabular features."""

from __future__ import annotations

import torch
from torch import nn


class NumericalFeatureTokenizer(nn.Module):
    """Convert each scalar feature into a feature-specific token."""

    def __init__(self, num_features: int, d_model: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_features, d_model))
        self.bias = nn.Parameter(torch.empty(num_features, d_model))
        nn.init.xavier_uniform_(self.weight)
        nn.init.normal_(self.bias, mean=0.0, std=0.02)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(
                f"Expected [batch, features], received shape={tuple(features.shape)}"
            )
        return features.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class TabularTransformerBinaryClassifier(nn.Module):
    """Transformer encoder with one token per standardized numerical feature."""

    def __init__(
        self,
        num_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_features < 1:
            raise ValueError("num_features must be positive")
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})"
            )
        if num_layers < 1:
            raise ValueError("num_layers must be positive")

        self.num_features = num_features
        self.tokenizer = NumericalFeatureTokenizer(num_features, d_model)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        hidden_dim = max(16, d_model // 2)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.num_features:
            raise ValueError(
                f"Model expects {self.num_features} features, "
                f"received {features.shape[-1]}"
            )
        tokens = self.tokenizer(features)
        cls_token = self.cls_token.expand(features.shape[0], -1, -1)
        encoded = self.encoder(torch.cat([cls_token, tokens], dim=1))
        return self.head(encoded[:, 0]).squeeze(-1)
