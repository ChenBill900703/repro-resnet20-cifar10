from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from resnet_repro.batch_norm import CaffeCompatibleBatchNorm2d
from resnet_repro.config import FrozenConfig
from resnet_repro.models import (
    OptionAShortcut,
    Plain20,
    PlainBlock,
    PostActivationResidualBlock,
    ResidualAdd,
    ResNet20,
)


EXPECTED_PARAMETER_COUNT = 269_722


@pytest.mark.parametrize("fixture_name", ["plain20", "resnet20"])
def test_arch_001_002_weighted_layer_and_module_counts(
    request: pytest.FixtureRequest, fixture_name: str
) -> None:
    model = request.getfixturevalue(fixture_name)
    convolutions = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    linears = [module for module in model.modules() if isinstance(module, nn.Linear)]

    assert model.weighted_layers == 20
    assert len(convolutions) == 19
    assert len(linears) == 1
    assert len(convolutions) + len(linears) == 20


@pytest.mark.parametrize(
    ("fixture_name", "block_type"),
    [("plain20", PlainBlock), ("resnet20", PostActivationResidualBlock)],
)
def test_arch_003_stage_block_counts(
    request: pytest.FixtureRequest, fixture_name: str, block_type: type[nn.Module]
) -> None:
    model = request.getfixturevalue(fixture_name)
    assert model.stage_block_counts == (3, 3, 3)
    assert [len(model.stage1), len(model.stage2), len(model.stage3)] == [3, 3, 3]
    assert sum(isinstance(module, block_type) for module in model.modules()) == 9


@pytest.mark.parametrize("fixture_name", ["plain20", "resnet20"])
def test_arch_004_forward_shapes(
    request: pytest.FixtureRequest, fixture_name: str
) -> None:
    model = request.getfixturevalue(fixture_name)
    value = torch.randn(2, 3, 32, 32)
    features, intermediates = model.forward_features(value)
    logits = model(value)

    assert [tuple(tensor.shape) for tensor in intermediates] == [
        (2, 16, 32, 32),
        (2, 16, 32, 32),
        (2, 32, 16, 16),
        (2, 64, 8, 8),
    ]
    assert tuple(features.shape) == (2, 64, 8, 8)
    assert tuple(logits.shape) == (2, 10)


def test_arch_005_plain_resnet_parameter_count_exact_equality(
    plain20: Plain20, resnet20: ResNet20
) -> None:
    plain_count = sum(parameter.numel() for parameter in plain20.parameters())
    resnet_count = sum(parameter.numel() for parameter in resnet20.parameters())
    assert plain_count == resnet_count == EXPECTED_PARAMETER_COUNT


def test_arch_006_parameter_count_rounds_to_paper_027m(resnet20: ResNet20) -> None:
    parameter_count = sum(parameter.numel() for parameter in resnet20.parameters())
    assert parameter_count == EXPECTED_PARAMETER_COUNT
    assert round(parameter_count / 1_000_000, 2) == 0.27


@pytest.mark.parametrize("fixture_name", ["plain20", "resnet20"])
def test_all_convolutions_are_bias_free_and_classifier_has_bias(
    request: pytest.FixtureRequest, fixture_name: str
) -> None:
    model = request.getfixturevalue(fixture_name)
    convolutions = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    assert all(module.bias is None for module in convolutions)
    assert model.classifier.bias is not None
    assert model.classifier.in_features == 64
    assert model.classifier.out_features == 10


@pytest.mark.parametrize("fixture_name", ["plain20", "resnet20"])
def test_every_convolution_has_caffe_compatible_batch_norm(
    request: pytest.FixtureRequest, fixture_name: str
) -> None:
    model = request.getfixturevalue(fixture_name)
    batch_norms = [
        module
        for module in model.modules()
        if isinstance(module, CaffeCompatibleBatchNorm2d)
    ]
    assert len(batch_norms) == 19
    assert isinstance(model.stem_bn, CaffeCompatibleBatchNorm2d)
    for stage in (model.stage1, model.stage2, model.stage3):
        for block in stage:
            assert isinstance(block.bn1, CaffeCompatibleBatchNorm2d)
            assert isinstance(block.bn2, CaffeCompatibleBatchNorm2d)


def test_plain_has_no_add_while_resnet_has_nine_adds(
    plain20: Plain20, resnet20: ResNet20
) -> None:
    assert not any(isinstance(module, ResidualAdd) for module in plain20.modules())
    assert not any(isinstance(module, OptionAShortcut) for module in plain20.modules())
    assert sum(isinstance(module, ResidualAdd) for module in resnet20.modules()) == 9
    assert sum(isinstance(module, OptionAShortcut) for module in resnet20.modules()) == 9


def test_residual_block_operation_order(resnet20: ResNet20) -> None:
    block = resnet20.stage1[0]
    names = ["conv1", "bn1", "relu1", "conv2", "bn2", "shortcut", "add", "relu2"]
    observed: list[str] = []
    handles = [
        getattr(block, name).register_forward_hook(
            lambda _module, _inputs, _output, name=name: observed.append(name)
        )
        for name in names
    ]
    try:
        block(torch.randn(2, 16, 32, 32))
    finally:
        for handle in handles:
            handle.remove()
    assert observed == names


def test_transition_blocks_use_stride_two_and_option_a(resnet20: ResNet20) -> None:
    for block in (resnet20.stage2[0], resnet20.stage3[0]):
        assert block.conv1.stride == (2, 2)
        assert block.conv2.stride == (1, 1)
        assert isinstance(block.shortcut, OptionAShortcut)
        assert block.shortcut.stride == 2
    for stage in (resnet20.stage1, resnet20.stage2, resnet20.stage3):
        for index, block in enumerate(stage):
            if block not in (resnet20.stage2[0], resnet20.stage3[0]):
                assert block.conv1.stride == (1, 1)


@pytest.mark.parametrize("model_class", [Plain20, ResNet20])
def test_cpu_forward_backward_smoke(
    frozen_config: FrozenConfig, model_class: type[Plain20] | type[ResNet20]
) -> None:
    torch.manual_seed(1)
    model = model_class(frozen_config).cpu().train()
    value = torch.randn(2, 3, 32, 32)
    logits = model(value)
    loss = logits.square().mean()
    loss.backward()

    assert tuple(logits.shape) == (2, 10)
    assert torch.isfinite(loss)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


@pytest.mark.parametrize("model_class", [Plain20, ResNet20])
def test_state_dict_roundtrip(
    frozen_config: FrozenConfig, model_class: type[Plain20] | type[ResNet20]
) -> None:
    torch.manual_seed(1)
    source = model_class(frozen_config).train()
    value = torch.randn(2, 3, 32, 32)
    source(value)
    state = copy.deepcopy(source.state_dict())
    source.eval()
    expected = source(value)

    restored = model_class(frozen_config)
    restored.load_state_dict(state)
    restored.eval()
    actual = restored(value)

    assert state.keys() == restored.state_dict().keys()
    for name, expected_value in state.items():
        torch.testing.assert_close(
            restored.state_dict()[name], expected_value, rtol=0, atol=0
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
