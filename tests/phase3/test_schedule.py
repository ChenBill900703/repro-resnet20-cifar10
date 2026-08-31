from __future__ import annotations

import pytest
from torch import nn

from resnet_repro.config import FrozenConfig
from resnet_repro.training.optimizer import build_sgd_optimizer
from resnet_repro.training.schedule import ExactUpdateLrController, TrainingCompleteError


def _controller(frozen_config: FrozenConfig) -> ExactUpdateLrController:
    model = nn.Linear(2, 1)
    optimizer = build_sgd_optimizer(model, frozen_config)
    return ExactUpdateLrController(optimizer, frozen_config)


@pytest.mark.parametrize(
    ("update_number", "expected_lr"),
    [
        (1, 0.1),
        (32_000, 0.1),
        (32_001, 0.01),
        (48_000, 0.01),
        (48_001, 0.001),
        (64_000, 0.001),
    ],
)
def test_lr_001_through_006_exact_boundaries(
    frozen_config: FrozenConfig, update_number: int, expected_lr: float
) -> None:
    controller = _controller(frozen_config)
    assert controller.lr_for_update(update_number) == expected_lr


def test_lr_006_rejects_update_64001(frozen_config: FrozenConfig) -> None:
    controller = _controller(frozen_config)
    controller.completed_updates = 64_000
    with pytest.raises(TrainingCompleteError, match="64,001"):
        controller.prepare_next_update()
    with pytest.raises(TrainingCompleteError, match="64,001"):
        controller.lr_for_update(64_001)


@pytest.mark.parametrize(
    ("completed", "expected_next_lr"),
    [(31_999, 0.1), (32_000, 0.01), (47_999, 0.01), (48_000, 0.001)],
)
def test_schedule_state_roundtrip_at_boundaries(
    frozen_config: FrozenConfig, completed: int, expected_next_lr: float
) -> None:
    source = _controller(frozen_config)
    state = source.state_dict()
    state["completed_updates"] = completed
    state["current_lr"] = expected_next_lr
    restored = _controller(frozen_config)
    restored.load_state_dict(state)
    assert restored.completed_updates == completed
    assert restored.current_lr == expected_next_lr
    assert restored.prepare_next_update() == (completed + 1, expected_next_lr)
