from __future__ import annotations

import argparse
import math
from typing import List, Type

import torch
import torch.nn.functional as F
from torch import nn

from .classifier_head import CommonClassifierHead
from .resnet3d_attention import BasicBlock3D, _norm3d


def _norm2d(num_channels: int) -> nn.Module:
    num_groups = 8
    while num_channels % num_groups != 0 and num_groups > 1:
        num_groups //= 2
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class ResNet3D(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        in_channels = 1
        num_classes = 2
        base_channels = args.resnet3d_base_channels
        layers = tuple(args.resnet3d_layers)

        self.inplanes = base_channels
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
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=1)
        self.classifier = CommonClassifierHead(
            in_features=base_channels * 8 * BasicBlock3D.expansion,
            num_classes=num_classes,
            hidden_dim=args.classifier_hidden_dim,
            dropout=args.classifier_dropout,
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

        layers: List[nn.Module] = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class BasicBlock2D(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = _norm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_channels, out_channels)
        self.bn2 = _norm2d(out_channels)
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


class SliceRelationModuleScalingL2(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 4) -> None:
        super().__init__()
        reduced_channels = max(in_channels // reduction, 1)
        self.reduced_channels = reduced_channels
        self.query_conv = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False)
        self.key_conv = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_slices, channels, height, width = x.shape
        flat = x.reshape(batch_size * num_slices, channels, height, width)

        query = self.query_conv(flat).mean(dim=(-1, -2)).reshape(batch_size, num_slices, -1)
        key = self.key_conv(flat).mean(dim=(-1, -2)).reshape(batch_size, num_slices, -1)
        value = self.value_conv(flat).mean(dim=(-1, -2)).reshape(batch_size, num_slices, channels)

        query = F.normalize(query, p=2, dim=-1)
        key = F.normalize(key, p=2, dim=-1)

        energy = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(self.reduced_channels)
        relation_matrix = F.softmax(energy, dim=-1)
        context = torch.bmm(relation_matrix, value).reshape(batch_size, num_slices, channels, 1, 1)
        out = x + self.gamma * context.expand(-1, -1, -1, height, width)
        return out, relation_matrix


class ResNet2D(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        in_channels = 1
        num_classes = 2
        base_channels = args.resnet3d_base_channels
        layers = tuple(args.resnet3d_layers)

        self.inplanes = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                base_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            _norm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(BasicBlock2D, base_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock2D, base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock2D, base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock2D, base_channels * 8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = CommonClassifierHead(
            in_features=base_channels * 8 * BasicBlock2D.expansion,
            num_classes=num_classes,
            hidden_dim=args.classifier_hidden_dim,
            dropout=args.classifier_dropout,
        )

    def _make_layer(
        self,
        block: Type[BasicBlock2D],
        planes: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                _norm2d(planes * block.expansion),
            )

        layers: List[nn.Module] = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _forward_slices(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, num_slices, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * num_slices, channels, height, width)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        _, feature_channels, feature_height, feature_width = x.shape
        return x.reshape(batch_size, num_slices, feature_channels, feature_height, feature_width)

    def _aggregate_features(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_slices, feature_channels, feature_height, feature_width = x.shape
        x = x.reshape(batch_size * num_slices, feature_channels, feature_height, feature_width)
        x = self.avgpool(x).reshape(batch_size, num_slices, feature_channels)
        return x.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_slices(x)
        x = self._aggregate_features(x)
        return self.classifier(x)


class ResNet2DAttention(ResNet2D):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        base_channels = args.resnet3d_base_channels
        self.relation_module = SliceRelationModuleScalingL2(
            base_channels * 8 * BasicBlock2D.expansion,
            reduction=args.resnet3d_attention_reduction,
        )
        self.last_relation_matrix: torch.Tensor | None = None
        self.last_feature_sequence: torch.Tensor | None = None
        self.last_feature_metadata: dict[str, int] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_relation_matrix = None
        self.last_feature_sequence = None
        self.last_feature_metadata = None

        x = self._forward_slices(x)
        x, relation_matrix = self.relation_module(x)
        self.last_relation_matrix = relation_matrix.detach()
        self.last_feature_sequence = x.permute(0, 2, 1, 3, 4).detach()
        self.last_feature_metadata = {
            "stage": 4,
            "num_slices": int(x.shape[1]),
            "channels": int(x.shape[2]),
            "height": int(x.shape[3]),
            "width": int(x.shape[4]),
        }
        x = self._aggregate_features(x)
        return self.classifier(x)

    def get_attention_outputs(self) -> dict[str, torch.Tensor | dict[str, int] | None]:
        return {
            "relation_matrix": self.last_relation_matrix,
            "feature_sequence": self.last_feature_sequence,
            "feature_metadata": self.last_feature_metadata,
        }
