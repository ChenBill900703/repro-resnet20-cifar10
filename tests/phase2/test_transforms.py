from __future__ import annotations

import random

import numpy as np
import torch
from torch.nn import functional as F

from resnet_repro.config import FrozenConfig
from resnet_repro.data.transforms import TestingTransform, TrainingTransform, to_float_0_255
from resnet_repro.reproducibility import capture_rng_state, restore_rng_state, sample_rng_scope


def _mean() -> torch.Tensor:
    return torch.arange(3 * 32 * 32, dtype=torch.float32).reshape(3, 32, 32) / 16.0


def test_to_float_preserves_0_255_scale() -> None:
    image = np.array([[[0, 127, 255]]], dtype=np.uint8)
    output = to_float_0_255(image)
    assert output.dtype == torch.float32
    assert output.shape == (3, 1, 1)
    torch.testing.assert_close(output[:, 0, 0], torch.tensor([0.0, 127.0, 255.0]), rtol=0, atol=0)


def test_training_transform_order_and_reference(frozen_config: FrozenConfig) -> None:
    mean = _mean()
    image = torch.arange(3 * 32 * 32, dtype=torch.uint8).reshape(3, 32, 32)
    transform = TrainingTransform(frozen_config.data, mean, base_seed=1)
    assert transform.operation_order == tuple(frozen_config.data["preprocessing"]["train_order"])

    with sample_rng_scope(1, 3, 17):
        top = int(torch.randint(0, 9, ()).item())
        left = int(torch.randint(0, 9, ()).item())
        flip = bool(torch.rand(()) < 0.5)
    expected = F.pad(to_float_0_255(image) - mean, (4, 4, 4, 4), value=0.0)
    expected = expected[:, top : top + 32, left : left + 32]
    if flip:
        expected = torch.flip(expected, dims=(2,))

    actual = transform(image, epoch=3, official_index=17)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_padding_after_mean_subtraction_is_centered_zero(frozen_config: FrozenConfig) -> None:
    mean = _mean()
    transform = TrainingTransform(frozen_config.data, mean, base_seed=1)
    padded = transform.center_and_pad(mean.clone())
    assert padded.shape == (3, 40, 40)
    torch.testing.assert_close(padded, torch.zeros_like(padded), rtol=0, atol=0)


def test_same_seed_epoch_index_augmentation_is_exact(frozen_config: FrozenConfig) -> None:
    transform = TrainingTransform(frozen_config.data, _mean(), base_seed=1)
    image = torch.arange(3 * 32 * 32, dtype=torch.uint8).reshape(3, 32, 32)
    first = transform(image, epoch=4, official_index=123)
    random.seed(999)
    np.random.seed(888)
    torch.manual_seed(777)
    second = transform(image, epoch=4, official_index=123)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_testing_transform_uses_same_mean_and_consumes_no_rng(frozen_config: FrozenConfig) -> None:
    mean = _mean()
    transform = TestingTransform(frozen_config.data, mean)
    image = torch.arange(3 * 32 * 32, dtype=torch.uint8).reshape(3, 32, 32)
    original = capture_rng_state()
    try:
        before = capture_rng_state()
        first = transform(image)
        after = capture_rng_state()
        second = transform(image)
        torch.testing.assert_close(first, to_float_0_255(image) - mean, rtol=0, atol=0)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        assert before.python == after.python
        np.testing.assert_array_equal(before.numpy[1], after.numpy[1])
        torch.testing.assert_close(before.torch_cpu, after.torch_cpu, rtol=0, atol=0)
        assert transform.operation_order == tuple(
            frozen_config.data["preprocessing"]["test_order"]
        )
    finally:
        restore_rng_state(original)
