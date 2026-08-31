from __future__ import annotations

import json

import pytest
import torch

from resnet_repro.batch_norm import CaffeCompatibleBatchNorm2d
from resnet_repro.bn_compatibility import (
    PRE_REGISTERED_ATOL,
    PRE_REGISTERED_BATCH_COUNT,
    PRE_REGISTERED_DTYPE,
    PRE_REGISTERED_RTOL,
    run_bn_compatibility,
)


def test_bn_harness_uses_pre_registered_protocol() -> None:
    report = run_bn_compatibility()

    assert report.batch_count == PRE_REGISTERED_BATCH_COUNT == 4
    assert report.dtype == str(PRE_REGISTERED_DTYPE)
    assert report.atol == PRE_REGISTERED_ATOL
    assert report.rtol == PRE_REGISTERED_RTOL
    assert report.pytorch_momentum == pytest.approx(0.001)
    assert report.caffe_moving_average_fraction == pytest.approx(0.999)
    assert [step.step for step in report.steps] == [1, 2, 3, 4]


def test_bn_harness_records_caffe_scale_factor_recurrence() -> None:
    report = run_bn_compatibility()
    expected = 0.0
    for step in report.steps:
        expected = 0.999 * expected + 1.0
        assert step.caffe_scale_factor == pytest.approx(expected)
        assert step.implementation_running_scale == pytest.approx(0.001 * expected)
        assert step.running_scale_close


def test_bn_harness_records_biased_and_unbiased_variance() -> None:
    report = run_bn_compatibility()
    first = report.steps[0]
    # Fixed batches contain N*H*W=8 values per channel.
    for biased, unbiased in zip(
        first.batch_variance_biased, first.batch_variance_unbiased, strict=True
    ):
        assert unbiased == pytest.approx(biased * 8.0 / 7.0)


def test_bn_003_compatibility_gate_passes() -> None:
    report = run_bn_compatibility()

    assert report.compatible
    assert report.evaluation_close
    assert report.evaluation_max_abs_diff <= report.atol
    assert all(step.running_mean_close for step in report.steps)
    assert all(step.running_variance_close for step in report.steps)
    assert report.conclusion.startswith("PASS")


def test_bn_report_is_json_serializable() -> None:
    payload = json.loads(json.dumps(run_bn_compatibility().to_dict()))
    assert payload["schema_version"] == 1
    assert payload["compatible"] is True
    assert len(payload["steps"]) == 4


def test_bn_harness_rejects_unregistered_batch_count() -> None:
    batch = torch.zeros((2, 3, 2, 2), dtype=torch.float64)
    with pytest.raises(ValueError, match="exactly 4"):
        run_bn_compatibility(batches=[batch])


def test_bn_harness_rejects_noncomplementary_momentum() -> None:
    with pytest.raises(ValueError, match="1 - Caffe"):
        run_bn_compatibility(pytorch_momentum=0.01)


def test_caffe_bn_state_dict_restores_scale_and_eval_exactly() -> None:
    source = CaffeCompatibleBatchNorm2d(3, dtype=torch.float64)
    source.train()
    source(torch.arange(24, dtype=torch.float64).reshape(2, 3, 2, 2))
    source.eval()
    probe = torch.linspace(-2, 2, 12, dtype=torch.float64).reshape(1, 3, 2, 2)
    expected = source(probe)

    restored = CaffeCompatibleBatchNorm2d(3, dtype=torch.float64)
    restored.load_state_dict(source.state_dict())
    restored.eval()

    torch.testing.assert_close(restored(probe), expected, rtol=0, atol=0)
    assert restored.running_scale.item() == source.running_scale.item()


def test_caffe_bn_rejects_eval_before_running_statistics_exist() -> None:
    layer = CaffeCompatibleBatchNorm2d(3)
    layer.eval()
    with pytest.raises(RuntimeError, match="before the first"):
        layer(torch.zeros(2, 3, 2, 2))


def test_caffe_bn_train_output_matches_batch_reference() -> None:
    layer = CaffeCompatibleBatchNorm2d(3, dtype=torch.float64)
    value = torch.arange(48, dtype=torch.float64).reshape(4, 3, 2, 2) / 7.0
    mean = value.mean(dim=(0, 2, 3), keepdim=True)
    variance = value.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
    expected = (value - mean) / torch.sqrt(variance + layer.eps)

    actual = layer(value)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_caffe_bn_affine_initialization() -> None:
    layer = CaffeCompatibleBatchNorm2d(5)
    torch.testing.assert_close(layer.weight, torch.ones(5), rtol=0, atol=0)
    torch.testing.assert_close(layer.bias, torch.zeros(5), rtol=0, atol=0)
    assert layer.affine


def test_caffe_bn_effective_statistics_match_unbiased_batch_after_first_update() -> None:
    layer = CaffeCompatibleBatchNorm2d(3, dtype=torch.float64)
    value = torch.arange(24, dtype=torch.float64).reshape(2, 3, 2, 2)
    layer(value)
    mean, variance = layer.effective_running_statistics()

    torch.testing.assert_close(mean, value.mean(dim=(0, 2, 3)), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        variance,
        value.var(dim=(0, 2, 3), unbiased=True),
        rtol=1e-12,
        atol=1e-12,
    )


def test_caffe_bn_eval_does_not_mutate_any_buffer() -> None:
    layer = CaffeCompatibleBatchNorm2d(3, dtype=torch.float64)
    value = torch.arange(24, dtype=torch.float64).reshape(2, 3, 2, 2)
    layer(value)
    before = {name: buffer.clone() for name, buffer in layer.named_buffers()}

    layer.eval()
    layer(value[:1])

    after = dict(layer.named_buffers())
    assert before.keys() == after.keys()
    for name in before:
        torch.testing.assert_close(after[name], before[name], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_caffe_bn_preserves_cpu_dtype_and_device(dtype: torch.dtype) -> None:
    layer = CaffeCompatibleBatchNorm2d(3, dtype=dtype, device="cpu")
    value = torch.randn(2, 3, 2, 2, dtype=dtype, device="cpu")
    output = layer(value)

    assert output.dtype == dtype
    assert output.device.type == "cpu"
    assert layer.running_mean.dtype == dtype
    assert layer.running_scale.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_caffe_bn_cuda_device_behavior() -> None:
    layer = CaffeCompatibleBatchNorm2d(3, dtype=torch.float32, device="cuda")
    value = torch.randn(2, 3, 2, 2, device="cuda")
    output = layer(value)
    assert output.device.type == "cuda"
    assert layer.running_scale.device.type == "cuda"


def test_caffe_bn_num_batches_tracked_counts_only_training_forwards() -> None:
    layer = CaffeCompatibleBatchNorm2d(3)
    value = torch.randn(2, 3, 2, 2)
    assert layer.num_batches_tracked.item() == 0
    layer(value)
    layer(value)
    assert layer.num_batches_tracked.item() == 2
    layer.eval()
    layer(value)
    assert layer.num_batches_tracked.item() == 2


def test_caffe_bn_reset_restores_all_initial_state() -> None:
    layer = CaffeCompatibleBatchNorm2d(3, dtype=torch.float64)
    layer(torch.arange(24, dtype=torch.float64).reshape(2, 3, 2, 2))
    with torch.no_grad():
        layer.weight.fill_(2.0)
        layer.bias.fill_(3.0)

    layer.reset_parameters()

    torch.testing.assert_close(layer.running_mean, torch.zeros(3, dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(layer.running_var, torch.zeros(3, dtype=torch.float64), rtol=0, atol=0)
    assert layer.running_scale.item() == 0.0
    assert layer.num_batches_tracked.item() == 0
    torch.testing.assert_close(layer.weight, torch.ones(3, dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(layer.bias, torch.zeros(3, dtype=torch.float64), rtol=0, atol=0)
    layer.eval()
    with pytest.raises(RuntimeError, match="before the first"):
        layer(torch.zeros(2, 3, 2, 2, dtype=torch.float64))


def test_caffe_bn_dtype_migration_moves_parameters_and_buffers() -> None:
    layer = CaffeCompatibleBatchNorm2d(3, dtype=torch.float32)
    layer(torch.randn(2, 3, 2, 2, dtype=torch.float32))

    migrated = layer.to(dtype=torch.float64)
    output = migrated(torch.randn(2, 3, 2, 2, dtype=torch.float64))

    assert output.dtype == torch.float64
    for parameter in migrated.parameters():
        assert parameter.dtype == torch.float64
    for name, buffer in migrated.named_buffers():
        expected = torch.long if name == "num_batches_tracked" else torch.float64
        assert buffer.dtype == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_caffe_bn_device_migration_moves_parameters_and_buffers() -> None:
    layer = CaffeCompatibleBatchNorm2d(3).cuda()
    output = layer(torch.randn(2, 3, 2, 2, device="cuda"))
    assert output.device.type == "cuda"
    assert all(parameter.device.type == "cuda" for parameter in layer.parameters())
    assert all(buffer.device.type == "cuda" for buffer in layer.buffers())

    restored = layer.cpu()
    assert all(parameter.device.type == "cpu" for parameter in restored.parameters())
    assert all(buffer.device.type == "cpu" for buffer in restored.buffers())
