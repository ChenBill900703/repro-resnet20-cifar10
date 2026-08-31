"""Approved Phase 1 initialization helpers.

The convolution fan-in choice and FC standard deviation are project decisions
recorded in the frozen config; they are not implicit PyTorch defaults.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from resnet_repro.batch_norm import CaffeCompatibleBatchNorm2d


def convolution_fan_in(layer: nn.Conv2d) -> int:
    kernel_height, kernel_width = layer.kernel_size
    return (layer.in_channels // layer.groups) * kernel_height * kernel_width


def convolution_target_std(layer: nn.Conv2d) -> float:
    return math.sqrt(2.0 / convolution_fan_in(layer))


def initialize_weighted_module(module: nn.Module, config: Mapping[str, Any]) -> None:
    initialization = config["initialization"]
    if isinstance(module, nn.Conv2d):
        convolution = initialization["convolution"]
        if (
            convolution["distribution"] != "normal"
            or convolution["fan_mode"] != "fan_in"
            or convolution["gain"] != "sqrt_2"
        ):
            raise ValueError("unsupported convolution initialization recipe")
        if module.bias is not None:
            raise ValueError("approved Conv2d layers must not have bias")
        torch.nn.init.normal_(
            module.weight,
            mean=convolution["mean"],
            std=convolution_target_std(module),
        )
    elif isinstance(module, nn.Linear):
        fully_connected = initialization["fully_connected"]
        if fully_connected["distribution"] != "normal":
            raise ValueError("unsupported fully connected initialization recipe")
        if module.bias is None:
            raise ValueError("approved classifier Linear layer must have bias")
        torch.nn.init.normal_(
            module.weight,
            mean=fully_connected["mean"],
            std=fully_connected["std"],
        )
        torch.nn.init.constant_(
            module.bias, initialization["bias"]["fully_connected"]
        )
    elif isinstance(module, CaffeCompatibleBatchNorm2d):
        if not module.affine or module.weight is None or module.bias is None:
            raise ValueError("approved BatchNorm layers must be affine")
        module.reset_running_stats()
        torch.nn.init.constant_(
            module.weight, initialization["batch_norm"]["gamma"]
        )
        torch.nn.init.constant_(
            module.bias, initialization["batch_norm"]["beta"]
        )


def initialize_module_tree(module: nn.Module, config: Mapping[str, Any]) -> None:
    """Replace every relevant PyTorch default with the frozen initialization."""

    with torch.no_grad():
        for child in module.modules():
            initialize_weighted_module(child, config)
