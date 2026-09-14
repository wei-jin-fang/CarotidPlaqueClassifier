from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .resnet3d_attention import ResNet3DAttention


class LayerRelationModuleScalingL2(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 4) -> None:
        super().__init__()
        reduced_channels = max(in_channels // reduction, 1)
        self.reduced_channels = reduced_channels
        self.query_conv = nn.Conv3d(in_channels, reduced_channels, kernel_size=1, bias=False)
        self.key_conv = nn.Conv3d(in_channels, reduced_channels, kernel_size=1, bias=False)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, channels, num_slices, height, width = x.shape

        proj_query = self.query_conv(x).mean(dim=(-1, -2)).permute(0, 2, 1)
        proj_key = self.key_conv(x).mean(dim=(-1, -2)).permute(0, 2, 1)

        proj_query = F.normalize(proj_query, p=2, dim=-1)
        proj_key = F.normalize(proj_key, p=2, dim=-1)

        energy = torch.bmm(proj_query, proj_key.transpose(1, 2)) / math.sqrt(self.reduced_channels)
        relation_matrix = F.softmax(energy, dim=-1)

        proj_value = self.value_conv(x).mean(dim=(-1, -2))
        context = torch.bmm(proj_value, relation_matrix.transpose(1, 2)).unsqueeze(-1).unsqueeze(-1)
        out = x + self.gamma * context.expand(-1, channels, num_slices, height, width)
        return out, relation_matrix


class ResNet3DAttentionScalingL2(ResNet3DAttention):
    def __init__(self, args) -> None:
        super().__init__(args)
        stage_channels = {
            1: args.resnet3d_base_channels,
            2: args.resnet3d_base_channels * 2,
            3: args.resnet3d_base_channels * 4,
            4: args.resnet3d_base_channels * 8,
        }
        self.relation_module = LayerRelationModuleScalingL2(
            stage_channels[args.resnet3d_attention_stage],
            reduction=args.resnet3d_attention_reduction,
        )
