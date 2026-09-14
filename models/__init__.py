from __future__ import annotations

import argparse

from .ablation_models import ResNet2D, ResNet2DAttention, ResNet3D
from .resnet3d_attention import ResNet3DAttention
from .resnet3d_attention_scaling_l2 import ResNet3DAttentionScalingL2

MODEL_REGISTRY = {
    "resnet3d": ResNet3D,
    "resnet2d_attention": ResNet2DAttention,
    "resnet2d": ResNet2D,
    "resnet3d_attention": ResNet3DAttention,
    "resnet3d_attention_scalingL2": ResNet3DAttentionScalingL2,
}


def build_model(model_name: str, args: argparse.Namespace):
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unsupported model '{model_name}'. Available models: {available}")
    model_class = MODEL_REGISTRY[model_name]
    return model_class(args)
