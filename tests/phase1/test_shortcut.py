from __future__ import annotations

import pytest
import torch

from resnet_repro.config import FrozenConfig
from resnet_repro.models import OptionAShortcut, ResNet20


def test_short_002_identity_returns_exact_same_tensor(frozen_config: FrozenConfig) -> None:
    shortcut = OptionAShortcut(frozen_config.data, 16, 16, stride=1)
    value = torch.randn(2, 16, 32, 32, dtype=torch.float64)
    output = shortcut(value)
    assert output is value
    torch.testing.assert_close(output, value, rtol=0, atol=0)
    assert output.dtype == value.dtype
    assert output.device == value.device


@pytest.mark.parametrize(
    ("in_channels", "out_channels", "spatial_size", "padding"),
    [(16, 32, 32, 8), (32, 64, 16, 16)],
)
def test_short_003_004_downsample_values_and_symmetric_padding(
    frozen_config: FrozenConfig,
    in_channels: int,
    out_channels: int,
    spatial_size: int,
    padding: int,
) -> None:
    shortcut = OptionAShortcut(
        frozen_config.data, in_channels, out_channels, stride=2
    )
    value = torch.arange(
        2 * in_channels * spatial_size * spatial_size, dtype=torch.float64
    ).reshape(2, in_channels, spatial_size, spatial_size)
    output = shortcut(value)
    expected_center = value[:, :, ::2, ::2]

    assert tuple(output.shape) == (
        2,
        out_channels,
        spatial_size // 2,
        spatial_size // 2,
    )
    torch.testing.assert_close(output[:, :padding], torch.zeros_like(output[:, :padding]), rtol=0, atol=0)
    torch.testing.assert_close(
        output[:, padding : padding + in_channels], expected_center, rtol=0, atol=0
    )
    torch.testing.assert_close(output[:, -padding:], torch.zeros_like(output[:, -padding:]), rtol=0, atol=0)
    assert output.dtype == value.dtype
    assert output.device == value.device


@pytest.mark.parametrize("shape", [(2, 16, 31, 32), (2, 16, 32, 31)])
def test_short_005_odd_spatial_size_is_rejected(
    frozen_config: FrozenConfig, shape: tuple[int, int, int, int]
) -> None:
    shortcut = OptionAShortcut(frozen_config.data, 16, 32, stride=2)
    with pytest.raises(ValueError, match="even spatial"):
        shortcut(torch.randn(*shape))


def test_short_001_shortcuts_have_no_parameters_or_buffers(
    frozen_config: FrozenConfig, resnet20: ResNet20
) -> None:
    standalone = OptionAShortcut(frozen_config.data, 16, 32, stride=2)
    assert list(standalone.parameters()) == []
    assert list(standalone.buffers()) == []
    for shortcut in (
        module for module in resnet20.modules() if isinstance(module, OptionAShortcut)
    ):
        assert list(shortcut.parameters()) == []
        assert list(shortcut.buffers()) == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_option_a_preserves_cuda_device_and_dtype(frozen_config: FrozenConfig) -> None:
    shortcut = OptionAShortcut(frozen_config.data, 16, 32, stride=2).cuda()
    value = torch.randn(2, 16, 32, 32, device="cuda", dtype=torch.float32)
    output = shortcut(value)
    assert output.device.type == "cuda"
    assert output.dtype == torch.float32
