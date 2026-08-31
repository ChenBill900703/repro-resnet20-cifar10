"""Caffe-compatible BatchNorm running-statistics semantics.

PyTorch's update coefficient ``momentum=0.001`` matches Caffe's
``moving_average_fraction=0.999``, but the default buffer initialization does
not implement Caffe's scale-factor normalization during early updates.  This
module keeps scaled accumulator buffers plus the corresponding scale, then
uses de-biased effective statistics for evaluation.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CaffeCompatibleBatchNorm2d(nn.Module):
    """BatchNorm2d with Caffe b590 accumulation and evaluation semantics."""

    def __init__(
        self,
        num_features: int,
        *,
        eps: float = 1.0e-5,
        momentum: float = 0.001,
        affine: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if type(num_features) is not int or num_features <= 0:
            raise ValueError("num_features must be a positive integer")
        if not 0.0 < momentum <= 1.0:
            raise ValueError("momentum must be in (0, 1]")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_features = num_features
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.empty(num_features, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(num_features, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.register_buffer("running_mean", torch.empty(num_features, **factory_kwargs))
        self.register_buffer("running_var", torch.empty(num_features, **factory_kwargs))
        self.register_buffer("running_scale", torch.empty((), **factory_kwargs))
        self.register_buffer(
            "num_batches_tracked",
            torch.tensor(0, dtype=torch.long, device=device),
        )
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        self.running_mean.zero_()
        self.running_var.zero_()
        self.running_scale.zero_()
        self.num_batches_tracked.zero_()

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def _check_input(self, value: torch.Tensor) -> None:
        if value.ndim != 4:
            raise ValueError(f"expected 4D NCHW input, got {value.ndim}D")
        if value.shape[1] != self.num_features:
            raise ValueError(
                f"expected {self.num_features} channels, got {value.shape[1]}"
            )

    def effective_running_statistics(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the Caffe TEST-phase mean and variance."""

        if float(self.running_scale.item()) <= 0.0:
            raise RuntimeError("running statistics are undefined before the first training batch")
        return self.running_mean / self.running_scale, self.running_var / self.running_scale

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self._check_input(value)
        if self.training:
            output = F.batch_norm(
                value,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                True,
                self.momentum,
                self.eps,
            )
            self.num_batches_tracked.add_(1)
            self.running_scale.mul_(1.0 - self.momentum).add_(self.momentum)
            return output
        effective_mean, effective_variance = self.effective_running_statistics()
        return F.batch_norm(
            value,
            effective_mean,
            effective_variance,
            self.weight,
            self.bias,
            False,
            0.0,
            self.eps,
        )

    def extra_repr(self) -> str:
        return (
            f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, "
            f"affine={self.affine}, caffe_scale_factor=True"
        )
