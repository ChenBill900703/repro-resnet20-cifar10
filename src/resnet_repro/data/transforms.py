"""Frozen CIFAR-10 preprocessing and identity-derived augmentation."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
from torch.nn import functional as F

from ..reproducibility import sample_rng_scope


_TRAIN_ORDER = (
    "convert_to_float_0_255",
    "subtract_per_pixel_mean",
    "constant_zero_pad_4",
    "random_crop_32",
    "random_horizontal_flip",
)
_TEST_ORDER = ("convert_to_float_0_255", "subtract_per_pixel_mean")


def to_float_0_255(image: Any) -> torch.Tensor:
    """Convert CIFAR image input to contiguous CHW float32 without rescaling."""

    if isinstance(image, torch.Tensor):
        tensor = image.detach()
    else:
        array = np.asarray(image)
        if not array.flags.writeable:
            array = array.copy()
        tensor = torch.as_tensor(array)
    if tensor.ndim != 3:
        raise ValueError("CIFAR-10 image must have exactly three dimensions")
    if tensor.shape[0] == 3:
        chw = tensor
    elif tensor.shape[-1] == 3:
        chw = tensor.permute(2, 0, 1)
    else:
        raise ValueError("CIFAR-10 image must have exactly three channels")
    output = chw.to(dtype=torch.float32).contiguous()
    if not torch.isfinite(output).all():
        raise ValueError("CIFAR-10 image contains non-finite values")
    if output.min().item() < 0.0 or output.max().item() > 255.0:
        raise ValueError("CIFAR-10 image values must stay in [0,255]")
    return output


def _validated_mean(config: Mapping[str, Any], mean: torch.Tensor) -> torch.Tensor:
    preprocessing = config["preprocessing"]
    if tuple(preprocessing["input_scale"]) != (0.0, 255.0):
        raise ValueError("frozen preprocessing must use the [0,255] input scale")
    if preprocessing["standard_deviation_normalization"]:
        raise ValueError("standard-deviation normalization is forbidden")
    expected_shape = tuple(preprocessing["mean"]["shape"])
    result = torch.as_tensor(mean, dtype=torch.float32).detach().clone()
    if tuple(result.shape) != expected_shape:
        raise ValueError(f"mean shape must be {expected_shape}, got {tuple(result.shape)}")
    if not torch.isfinite(result).all():
        raise ValueError("mean contains non-finite values")
    return result.contiguous()


class TrainingTransform:
    """Mean-first zero-padding, random crop, and horizontal flip."""

    def __init__(
        self,
        config: Mapping[str, Any],
        mean: torch.Tensor,
        *,
        base_seed: int,
    ) -> None:
        order = tuple(config["preprocessing"]["train_order"])
        if order != _TRAIN_ORDER:
            raise ValueError(f"unsupported frozen training transform order: {order}")
        self.operation_order = order
        self.mean = _validated_mean(config, mean)
        self.base_seed = base_seed

    def center_and_pad(self, image: Any) -> torch.Tensor:
        centered = to_float_0_255(image) - self.mean
        return F.pad(centered, (4, 4, 4, 4), mode="constant", value=0.0)

    def __call__(
        self,
        image: Any,
        *,
        epoch: int,
        official_index: int,
    ) -> torch.Tensor:
        padded = self.center_and_pad(image)
        with sample_rng_scope(self.base_seed, epoch, official_index):
            top = int(torch.randint(0, 9, ()).item())
            left = int(torch.randint(0, 9, ()).item())
            flip = bool(torch.rand(()) < 0.5)
        output = padded[:, top : top + 32, left : left + 32]
        if flip:
            output = torch.flip(output, dims=(2,))
        return output.contiguous()


class TestingTransform:
    """Deterministic test transform using the training-set mean."""

    __test__ = False

    def __init__(self, config: Mapping[str, Any], mean: torch.Tensor) -> None:
        order = tuple(config["preprocessing"]["test_order"])
        if order != _TEST_ORDER:
            raise ValueError(f"unsupported frozen test transform order: {order}")
        if not config["preprocessing"]["mean"]["reuse_for_test"]:
            raise ValueError("frozen config must reuse the training mean for test")
        self.operation_order = order
        self.mean = _validated_mean(config, mean)

    def __call__(self, image: Any) -> torch.Tensor:
        return (to_float_0_255(image) - self.mean).contiguous()
