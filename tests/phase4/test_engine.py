from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from resnet_repro.config import FrozenConfig
from resnet_repro.training.engine import (
    EXPECTED_SOURCE_COMMIT,
    _atomic_write_json,
    make_run_paths,
)
from resnet_repro.training.optimizer import build_sgd_optimizer
from resnet_repro.training.schedule import ExactUpdateLrController
from resnet_repro.training.step import train_one_update


class _Acknowledger:
    def __init__(self) -> None:
        self.indices: tuple[int, ...] | None = None

    def mark_batch_consumed(self, indices) -> None:
        self.indices = tuple(indices)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value)


def test_phase4_source_commit_pin_is_optional_for_public_clones() -> None:
    assert EXPECTED_SOURCE_COMMIT is None


def test_run_paths_are_confined_to_requested_root(tmp_path: Path) -> None:
    paths = make_run_paths(tmp_path / "run")
    assert paths.root == (tmp_path / "run").resolve()
    assert paths.manifest.parent == paths.root
    assert paths.events.parent == paths.root
    assert paths.checkpoints.parent == paths.root
    assert paths.evaluations.parent == paths.root


def test_atomic_manifest_write_retries_transient_windows_reader_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    destination = tmp_path / "run_manifest.json"
    real_replace = os.replace
    attempts = 0

    def transiently_locked(source, target):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("simulated Windows sharing violation")
        return real_replace(source, target)

    monkeypatch.setattr("resnet_repro.training.engine.os.replace", transiently_locked)
    _atomic_write_json(destination, {"status": "running", "completed_updates": 7})
    assert attempts == 3
    assert destination.read_text(encoding="utf-8") == (
        '{\n  "completed_updates": 7,\n  "status": "running"\n}\n'
    )


def test_training_step_reports_metrics_and_acknowledges_exact_indices(
    frozen_config: FrozenConfig,
) -> None:
    torch.manual_seed(1)
    model = _Model()
    optimizer = build_sgd_optimizer(model, frozen_config)
    controller = ExactUpdateLrController(optimizer, frozen_config)
    acknowledger = _Acknowledger()
    images = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([0, 1])
    result = train_one_update(
        model,
        optimizer,
        controller,
        images=images,
        targets=targets,
        indices=(7, 3),
        batch_acknowledger=acknowledger,
    )
    assert result.update_number == 1
    assert result.learning_rate == 0.1
    assert result.sample_count == 2
    assert 0 <= result.correct_count <= 2
    assert result.indices == acknowledger.indices == (7, 3)
    assert result.loss == pytest.approx(result.loss)
