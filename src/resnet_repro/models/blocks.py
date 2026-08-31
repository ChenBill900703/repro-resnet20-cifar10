"""Shared Plain/Residual building blocks for the CIFAR-10 depth-20 models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from resnet_repro.batch_norm import CaffeCompatibleBatchNorm2d


def make_conv3x3(
    config: Mapping[str, Any],
    in_channels: int,
    out_channels: int,
    *,
    stride: int,
) -> nn.Conv2d:
    stem = config["architecture"]["stem"]
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=stem["kernel_size"],
        stride=stride,
        padding=stem["padding"],
        bias=stem["bias"],
    )


def make_batch_norm(
    config: Mapping[str, Any], num_features: int
) -> CaffeCompatibleBatchNorm2d:
    batch_norm = config["batch_normalization"]
    return CaffeCompatibleBatchNorm2d(
        num_features,
        eps=batch_norm["epsilon"],
        momentum=batch_norm["pytorch_momentum"],
        affine=batch_norm["affine"],
    )


class OptionAShortcut(nn.Module):
    """Parameter-free option A shortcut.

    The exact ``::2`` indexing, symmetric channel padding, and odd-size error
    are approved project assumptions rather than paper-explicit tensor code.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
    ) -> None:
        super().__init__()
        shortcut = config["architecture"]["shortcut"]
        approved_semantics = (
            shortcut["type"] == "option_a"
            and not shortcut["trainable_parameters"]
            and shortcut["spatial_downsample"] == "even_index_stride_2"
            and shortcut["channel_padding"] == "symmetric_zero"
            and shortcut["odd_size_policy"] == "error"
        )
        if not approved_semantics:
            raise ValueError("Option A parameter-free shortcut is required")
        if stride == 1:
            if in_channels != out_channels:
                raise ValueError("identity shortcut requires equal channel counts")
        elif stride == 2:
            if out_channels <= in_channels:
                raise ValueError("downsample shortcut must increase channels")
            if (out_channels - in_channels) % 2:
                raise ValueError("symmetric channel padding requires an even difference")
        else:
            raise ValueError("Option A shortcut stride must be 1 or 2")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4:
            raise ValueError(f"Option A expects 4D NCHW input, got {value.ndim}D")
        if value.shape[1] != self.in_channels:
            raise ValueError(
                f"Option A expected {self.in_channels} channels, got {value.shape[1]}"
            )
        if self.stride == 1:
            return value
        if value.shape[2] % 2 or value.shape[3] % 2:
            raise ValueError("Option A downsample requires even spatial dimensions")
        downsampled = value[:, :, ::2, ::2]
        padding = (self.out_channels - self.in_channels) // 2
        before = value.new_zeros(
            value.shape[0], padding, downsampled.shape[2], downsampled.shape[3]
        )
        after = value.new_zeros(
            value.shape[0], padding, downsampled.shape[2], downsampled.shape[3]
        )
        return torch.cat((before, downsampled, after), dim=1)


class ResidualAdd(nn.Module):
    """Observable, parameter-free residual addition operation."""

    def forward(self, residual: torch.Tensor, shortcut: torch.Tensor) -> torch.Tensor:
        return residual + shortcut


class PlainBlock(nn.Module):
    """Two-convolution plain counterpart with no shortcut or addition."""

    def __init__(
        self,
        config: Mapping[str, Any],
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
    ) -> None:
        super().__init__()
        self.conv1 = make_conv3x3(config, in_channels, out_channels, stride=stride)
        self.bn1 = make_batch_norm(config, out_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = make_conv3x3(config, out_channels, out_channels, stride=1)
        self.bn2 = make_batch_norm(config, out_channels)
        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.conv1(value)
        value = self.bn1(value)
        value = self.relu1(value)
        value = self.conv2(value)
        value = self.bn2(value)
        return self.relu2(value)


class PostActivationResidualBlock(nn.Module):
    """Approved ``conv-BN-ReLU-conv-BN-add-ReLU`` residual block."""

    def __init__(
        self,
        config: Mapping[str, Any],
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
    ) -> None:
        super().__init__()
        self.conv1 = make_conv3x3(config, in_channels, out_channels, stride=stride)
        self.bn1 = make_batch_norm(config, out_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = make_conv3x3(config, out_channels, out_channels, stride=1)
        self.bn2 = make_batch_norm(config, out_channels)
        self.shortcut = OptionAShortcut(
            config, in_channels, out_channels, stride=stride
        )
        self.add = ResidualAdd()
        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        identity = value
        value = self.conv1(value)
        value = self.bn1(value)
        value = self.relu1(value)
        value = self.conv2(value)
        value = self.bn2(value)
        value = self.add(value, self.shortcut(identity))
        return self.relu2(value)
