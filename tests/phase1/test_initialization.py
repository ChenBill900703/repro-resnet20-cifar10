from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from resnet_repro.batch_norm import CaffeCompatibleBatchNorm2d
from resnet_repro.config import FrozenConfig
from resnet_repro.models import Plain20, ResNet20
from resnet_repro.models.initialization import (
    convolution_target_std,
    initialize_weighted_module,
)


def test_init_001_convolution_uses_normal_distribution_path(
    frozen_config: FrozenConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    layer = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
    original = torch.nn.init.normal_
    calls: list[tuple[float, float]] = []

    def recording_normal_(tensor: torch.Tensor, mean: float, std: float):
        calls.append((mean, std))
        return original(tensor, mean=mean, std=std)

    monkeypatch.setattr(torch.nn.init, "normal_", recording_normal_)
    initialize_weighted_module(layer, frozen_config.data)

    assert calls == [(0.0, convolution_target_std(layer))]


@pytest.mark.parametrize("fixture_name", ["plain20", "resnet20"])
def test_init_002_all_convolution_stds_match_fan_in_recipe(
    request: pytest.FixtureRequest, fixture_name: str
) -> None:
    model = request.getfixturevalue(fixture_name)
    convolutions = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    for convolution in convolutions:
        target = convolution_target_std(convolution)
        empirical = float(convolution.weight.std(unbiased=False).item())
        assert math.isclose(empirical, target, rel_tol=0.20, abs_tol=0.0)


@pytest.mark.parametrize("fixture_name", ["plain20", "resnet20"])
def test_init_003_all_convolutions_have_no_bias(
    request: pytest.FixtureRequest, fixture_name: str
) -> None:
    model = request.getfixturevalue(fixture_name)
    assert all(
        module.bias is None
        for module in model.modules()
        if isinstance(module, nn.Conv2d)
    )


@pytest.mark.parametrize("fixture_name", ["plain20", "resnet20"])
def test_init_004_fc_normal_std_and_zero_bias(
    request: pytest.FixtureRequest, fixture_name: str
) -> None:
    model = request.getfixturevalue(fixture_name)
    weight = model.classifier.weight
    assert abs(float(weight.mean().item())) < 0.0015
    assert math.isclose(
        float(weight.std(unbiased=False).item()), 0.01, rel_tol=0.15, abs_tol=0.0
    )
    torch.testing.assert_close(
        model.classifier.bias, torch.zeros_like(model.classifier.bias), rtol=0, atol=0
    )


@pytest.mark.parametrize("fixture_name", ["plain20", "resnet20"])
def test_bn_gamma_one_beta_zero(
    request: pytest.FixtureRequest, fixture_name: str
) -> None:
    model = request.getfixturevalue(fixture_name)
    batch_norms = [
        module
        for module in model.modules()
        if isinstance(module, CaffeCompatibleBatchNorm2d)
    ]
    assert len(batch_norms) == 19
    for batch_norm in batch_norms:
        torch.testing.assert_close(
            batch_norm.weight, torch.ones_like(batch_norm.weight), rtol=0, atol=0
        )
        torch.testing.assert_close(
            batch_norm.bias, torch.zeros_like(batch_norm.bias), rtol=0, atol=0
        )
