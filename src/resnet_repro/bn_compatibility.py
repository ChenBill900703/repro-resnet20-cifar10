"""Mandatory Caffe/PyTorch BatchNorm semantics compatibility harness.

The harness uses four fixed float64 batches and tolerances declared in code
before execution.  It does not train a model.  A non-zero CLI exit means the
approved candidate cannot pass the formal-training gate without decision
review.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch

from .batch_norm import CaffeCompatibleBatchNorm2d
from .config import load_frozen_config


PRE_REGISTERED_BATCH_COUNT = 4
PRE_REGISTERED_DTYPE = torch.float64
PRE_REGISTERED_ATOL = 1.0e-10
PRE_REGISTERED_RTOL = 1.0e-8


@dataclass(frozen=True)
class BNStepComparison:
    step: int
    caffe_scale_factor: float
    implementation_running_scale: float
    batch_mean: list[float]
    batch_variance_biased: list[float]
    batch_variance_unbiased: list[float]
    caffe_effective_mean: list[float]
    implementation_effective_mean: list[float]
    caffe_effective_variance: list[float]
    implementation_effective_variance: list[float]
    running_mean_max_abs_diff: float
    running_variance_max_abs_diff: float
    running_mean_close: bool
    running_variance_close: bool
    running_scale_close: bool


@dataclass(frozen=True)
class BNCompatibilityReport:
    schema_version: int
    compatible: bool
    batch_count: int
    dtype: str
    caffe_moving_average_fraction: float
    pytorch_momentum: float
    epsilon: float
    atol: float
    rtol: float
    steps: tuple[BNStepComparison, ...]
    evaluation_max_abs_diff: float
    evaluation_close: bool
    conclusion: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _fixed_batches() -> tuple[torch.Tensor, ...]:
    base = torch.arange(24, dtype=PRE_REGISTERED_DTYPE).reshape(2, 3, 2, 2)
    return (
        (base - 11.5) / 4.0,
        base.flip(0) / 3.0 + 2.0,
        base.flip(-1) * -0.25 + 1.5,
        torch.sin(base / 5.0) * 3.0 - 0.75,
    )


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left - right)).item())


def _as_floats(value: torch.Tensor) -> list[float]:
    return [float(item) for item in value.tolist()]


def run_bn_compatibility(
    *,
    pytorch_momentum: float = 0.001,
    epsilon: float = 1.0e-5,
    caffe_moving_average_fraction: float = 0.999,
    batches: Sequence[torch.Tensor] | None = None,
    atol: float = PRE_REGISTERED_ATOL,
    rtol: float = PRE_REGISTERED_RTOL,
) -> BNCompatibilityReport:
    """Compare Caffe b590 accumulated/scale semantics with torch BatchNorm2d."""

    if batches is None:
        batches = _fixed_batches()
    if len(batches) != PRE_REGISTERED_BATCH_COUNT:
        raise ValueError(f"BN gate requires exactly {PRE_REGISTERED_BATCH_COUNT} batches")
    if abs((1.0 - caffe_moving_average_fraction) - pytorch_momentum) > 1.0e-15:
        raise ValueError("candidate requires pytorch_momentum == 1 - Caffe fraction")

    channel_count = batches[0].shape[1]
    bn = CaffeCompatibleBatchNorm2d(
        channel_count,
        eps=epsilon,
        momentum=pytorch_momentum,
        affine=True,
        dtype=PRE_REGISTERED_DTYPE,
    )
    with torch.no_grad():
        bn.weight.fill_(1.0)
        bn.bias.zero_()

    caffe_mean_blob = torch.zeros(channel_count, dtype=PRE_REGISTERED_DTYPE)
    caffe_variance_blob = torch.zeros(channel_count, dtype=PRE_REGISTERED_DTYPE)
    caffe_scale_factor = 0.0
    steps: list[BNStepComparison] = []

    bn.train()
    for step, batch in enumerate(batches, start=1):
        if batch.dtype != PRE_REGISTERED_DTYPE or batch.ndim != 4:
            raise ValueError("all BN gate batches must be float64 NCHW tensors")
        if batch.shape[1] != channel_count:
            raise ValueError("all BN gate batches must have the same channel count")
        reduce_dims = (0, 2, 3)
        batch_mean = batch.mean(dim=reduce_dims)
        variance_biased = batch.var(dim=reduce_dims, unbiased=False)
        sample_count = batch.numel() // channel_count
        correction = sample_count / (sample_count - 1) if sample_count > 1 else 1.0
        variance_unbiased = variance_biased * correction

        caffe_scale_factor = (
            caffe_moving_average_fraction * caffe_scale_factor + 1.0
        )
        caffe_mean_blob = (
            caffe_moving_average_fraction * caffe_mean_blob + batch_mean
        )
        caffe_variance_blob = (
            caffe_moving_average_fraction * caffe_variance_blob + variance_unbiased
        )
        with torch.no_grad():
            bn(batch)

        caffe_mean = caffe_mean_blob / caffe_scale_factor
        caffe_variance = caffe_variance_blob / caffe_scale_factor
        implementation_mean, implementation_variance = (
            bn.effective_running_statistics()
        )
        mean_close = torch.allclose(
            caffe_mean, implementation_mean, atol=atol, rtol=rtol
        )
        variance_close = torch.allclose(
            caffe_variance, implementation_variance, atol=atol, rtol=rtol
        )
        expected_running_scale = pytorch_momentum * caffe_scale_factor
        scale_close = abs(float(bn.running_scale.item()) - expected_running_scale) <= atol
        steps.append(
            BNStepComparison(
                step=step,
                caffe_scale_factor=caffe_scale_factor,
                implementation_running_scale=float(bn.running_scale.item()),
                batch_mean=_as_floats(batch_mean),
                batch_variance_biased=_as_floats(variance_biased),
                batch_variance_unbiased=_as_floats(variance_unbiased),
                caffe_effective_mean=_as_floats(caffe_mean),
                implementation_effective_mean=_as_floats(implementation_mean),
                caffe_effective_variance=_as_floats(caffe_variance),
                implementation_effective_variance=_as_floats(implementation_variance),
                running_mean_max_abs_diff=_max_abs(caffe_mean, implementation_mean),
                running_variance_max_abs_diff=_max_abs(caffe_variance, implementation_variance),
                running_mean_close=mean_close,
                running_variance_close=variance_close,
                running_scale_close=scale_close,
            )
        )

    probe = torch.tensor(
        [[[[0.5, -1.0], [2.0, 3.5]], [[-2.0, 0.0], [1.0, 4.0]], [[3.0, 2.0], [-1.0, 0.25]]]],
        dtype=PRE_REGISTERED_DTYPE,
    )
    bn.eval()
    with torch.no_grad():
        pytorch_output = bn(probe)
    caffe_mean = caffe_mean_blob / caffe_scale_factor
    caffe_variance = caffe_variance_blob / caffe_scale_factor
    caffe_output = (
        (probe - caffe_mean.reshape(1, -1, 1, 1))
        / torch.sqrt(caffe_variance.reshape(1, -1, 1, 1) + epsilon)
    )
    evaluation_close = torch.allclose(
        caffe_output, pytorch_output, atol=atol, rtol=rtol
    )
    compatible = evaluation_close and all(
        step.running_mean_close
        and step.running_variance_close
        and step.running_scale_close
        for step in steps
    )
    conclusion = (
        "PASS: candidate implements the pre-registered Caffe-like semantics."
        if compatible
        else "FAIL: the compatibility implementation differs from the pre-registered "
        "Caffe reference; decision review is required."
    )
    return BNCompatibilityReport(
        schema_version=1,
        compatible=compatible,
        batch_count=len(batches),
        dtype=str(PRE_REGISTERED_DTYPE),
        caffe_moving_average_fraction=caffe_moving_average_fraction,
        pytorch_momentum=pytorch_momentum,
        epsilon=epsilon,
        atol=atol,
        rtol=rtol,
        steps=tuple(steps),
        evaluation_max_abs_diff=_max_abs(caffe_output, pytorch_output),
        evaluation_close=evaluation_close,
        conclusion=conclusion,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cifar10_plain20_resnet20_frozen.yaml"),
    )
    parser.add_argument("--json", action="store_true", help="emit the complete JSON report")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = load_frozen_config(args.config)
    report = run_bn_compatibility(
        pytorch_momentum=config.data["batch_normalization"]["pytorch_momentum"],
        epsilon=config.data["batch_normalization"]["epsilon"],
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(report.conclusion)
        print(f"evaluation_max_abs_diff={report.evaluation_max_abs_diff:.12g}")
    return 0 if report.compatible else 2


if __name__ == "__main__":
    raise SystemExit(main())
