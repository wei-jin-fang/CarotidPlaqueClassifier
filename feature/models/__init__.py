"""Models used only by the independent tabular-feature task."""

from .tabular_transformer import TabularTransformerBinaryClassifier
from .tiny_mlp import TinyMLPBinaryClassifier


def build_feature_model(
    model_name: str,
    *,
    num_features: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float,
    mlp_hidden_dims: tuple[int, ...] = (32, 16),
    mlp_dropouts: tuple[float, ...] = (0.15, 0.10),
    mlp_use_layer_norm: bool = True,
) -> TabularTransformerBinaryClassifier | TinyMLPBinaryClassifier:
    normalized_name = model_name.lower()
    if normalized_name == "tabular_transformer":
        return TabularTransformerBinaryClassifier(
            num_features=num_features,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
    if normalized_name == "tiny_mlp":
        return TinyMLPBinaryClassifier(
            num_features=num_features,
            hidden_dims=mlp_hidden_dims,
            dropouts=mlp_dropouts,
            use_layer_norm=mlp_use_layer_norm,
        )
    raise ValueError(
        f"Unsupported feature model: {model_name!r}. "
        "Available models: tabular_transformer, tiny_mlp"
    )


__all__ = [
    "TabularTransformerBinaryClassifier",
    "TinyMLPBinaryClassifier",
    "build_feature_model",
]
