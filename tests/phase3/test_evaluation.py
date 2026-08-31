from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from resnet_repro.batch_norm import CaffeCompatibleBatchNorm2d
from resnet_repro.config import FrozenConfig
from resnet_repro.evaluation import evaluate


def test_eval_002_known_test_error_and_total_loss(frozen_config: FrozenConfig) -> None:
    model = nn.Identity()
    logits = torch.tensor(
        [
            [3.0, 1.0, 0.0],
            [0.0, 4.0, 1.0],
            [0.0, 2.0, 3.0],
            [0.0, 2.0, 1.0],
        ]
    )
    targets = torch.tensor([0, 1, 2, 0])
    expected_loss = float(nn.functional.cross_entropy(logits, targets, reduction="sum"))
    result = evaluate(model, [{"image": logits, "target": targets}], frozen_config)
    assert result.correct_count == 3
    assert result.sample_count == 4
    assert result.test_error_percent == 25.0
    assert result.total_loss == pytest.approx(expected_loss)
    assert result.mean_loss == pytest.approx(expected_loss / 4)


class _BNEvaluationProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = CaffeCompatibleBatchNorm2d(3)
        self.classifier = nn.Linear(3, 2)
        self.observed_training_modes: list[bool] = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.observed_training_modes.append(self.training)
        value = self.bn(value)
        return self.classifier(value.mean(dim=(2, 3)))


def test_evaluation_uses_effective_bn_stats_and_changes_no_state(
    frozen_config: FrozenConfig,
) -> None:
    torch.manual_seed(1)
    model = _BNEvaluationProbe().train()
    training_batch = torch.arange(48, dtype=torch.float32).reshape(4, 3, 2, 2)
    model(training_batch)
    model.observed_training_modes.clear()
    model_state = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    batch = {
        "image": torch.flip(training_batch, dims=(0,)),
        "target": torch.tensor([0, 1, 0, 1]),
    }

    result = evaluate(model, [batch], frozen_config)

    assert result.sample_count == 4
    assert model.observed_training_modes == [False]
    assert model.training is True
    for name, before in model_state.items():
        torch.testing.assert_close(model.state_dict()[name], before, rtol=0, atol=0)
    assert optimizer.state_dict() == optimizer_state
    effective_mean, effective_variance = model.bn.effective_running_statistics()
    assert torch.isfinite(effective_mean).all()
    assert torch.isfinite(effective_variance).all()


def test_evaluation_rejects_empty_input(frozen_config: FrozenConfig) -> None:
    with pytest.raises(ValueError, match="at least one"):
        evaluate(nn.Identity(), [], frozen_config)
