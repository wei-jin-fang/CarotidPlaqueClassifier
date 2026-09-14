from __future__ import annotations

import argparse
from typing import List, Type

import torch
import torch.nn.functional as F
from torch import nn

from .classifier_head import CommonClassifierHead


def _norm3d(num_channels: int) -> nn.Module:
    num_groups = 8
    while num_channels % num_groups != 0 and num_groups > 1:
        num_groups //= 2
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


def _conv3x3x3(in_channels: int, out_channels: int, stride: tuple[int, int, int] = (1, 1, 1)) -> nn.Conv3d:
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int, int] = (1, 1, 1),
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = _conv3x3x3(in_channels, out_channels, stride=stride)
        self.bn1 = _norm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3x3(out_channels, out_channels)
        self.bn2 = _norm3d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class LayerRelationModule(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 4) -> None:
        super().__init__()
        reduced_channels = max(in_channels // reduction, 1)
        self.query_conv = nn.Conv3d(in_channels, reduced_channels, kernel_size=1, bias=False)
        self.key_conv = nn.Conv3d(in_channels, reduced_channels, kernel_size=1, bias=False)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, num_slices, height, width = x.shape

        proj_query = self.query_conv(x).mean(dim=(-1, -2)).permute(0, 2, 1) #
        proj_key = self.key_conv(x).mean(dim=(-1, -2))#最终 $K$ 的维度：[1, 16, 64]
        energy = torch.bmm(proj_query, proj_key)
        relation_matrix = F.softmax(energy, dim=-1)

        proj_value = self.value_conv(x).mean(dim=(-1, -2))
        context = torch.bmm(proj_value, relation_matrix.transpose(1, 2)).unsqueeze(-1).unsqueeze(-1)
        out = x + self.gamma * context.expand(-1, -1, -1, height, width)
        return out, relation_matrix


class ResNet3DAttention(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        in_channels = 1
        num_classes = 2
        base_channels = args.resnet3d_base_channels
        layers = tuple(args.resnet3d_layers)
        classifier_hidden_dim = args.classifier_hidden_dim
        classifier_dropout = args.classifier_dropout
        attention_reduction = args.resnet3d_attention_reduction
        attention_stage = args.resnet3d_attention_stage

        self.inplanes = base_channels
        self.attention_stage = attention_stage
        self.last_relation_matrix: torch.Tensor | None = None
        self.last_feature_sequence: torch.Tensor | None = None
        self.last_feature_metadata: dict[str, int] | None = None

        self.stem = nn.Sequential(
            nn.Conv3d(
                in_channels,
                base_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            _norm3d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(BasicBlock3D, base_channels, layers[0], stride=(1, 1, 1))
        self.layer2 = self._make_layer(BasicBlock3D, base_channels * 2, layers[1], stride=(1, 2, 2))
        self.layer3 = self._make_layer(BasicBlock3D, base_channels * 4, layers[2], stride=(1, 2, 2))
        self.layer4 = self._make_layer(BasicBlock3D, base_channels * 8, layers[3], stride=(2, 2, 2))

        stage_channels = {
            1: base_channels,
            2: base_channels * 2,
            3: base_channels * 4,
            4: base_channels * 8,
        }
        self.relation_module = LayerRelationModule(stage_channels[attention_stage], reduction=attention_reduction)

        self.avgpool = nn.AdaptiveAvgPool3d(output_size=1)
        self.classifier = CommonClassifierHead(
            in_features=base_channels * 8 * BasicBlock3D.expansion,
            num_classes=num_classes,
            hidden_dim=classifier_hidden_dim,
            dropout=classifier_dropout,
        )

    def _make_layer(
        self,
        block: Type[BasicBlock3D],
        planes: int,
        blocks: int,
        stride: tuple[int, int, int],
    ) -> nn.Sequential:
        downsample = None
        if stride != (1, 1, 1) or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                _norm3d(planes * block.expansion),
            )

        layers: List[nn.Module] = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _apply_attention(self, x: torch.Tensor, stage_idx: int) -> torch.Tensor:
        if stage_idx == self.attention_stage:
            x, relation_matrix = self.relation_module(x)
            self.last_relation_matrix = relation_matrix.detach()
            self.last_feature_sequence = x.detach()
            self.last_feature_metadata = {
                "stage": stage_idx,
                "num_slices": int(x.shape[2]),
                "channels": int(x.shape[1]),
                "height": int(x.shape[3]),
                "width": int(x.shape[4]),
            }
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_relation_matrix = None
        self.last_feature_sequence = None
        self.last_feature_metadata = None

        x = self.stem(x)
        x = self.layer1(x)
        x = self._apply_attention(x, 1)
        x = self.layer2(x)
        x = self._apply_attention(x, 2)
        x = self.layer3(x)
        x = self._apply_attention(x, 3)
        x = self.layer4(x)
        x = self._apply_attention(x, 4)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_attention_outputs(self) -> dict[str, torch.Tensor | dict[str, int] | None]:
        return {
            "relation_matrix": self.last_relation_matrix,
            "feature_sequence": self.last_feature_sequence,
            "feature_metadata": self.last_feature_metadata,
        }
