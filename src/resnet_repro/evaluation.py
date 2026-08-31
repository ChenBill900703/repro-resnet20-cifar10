"""Side-effect-free, single-view evaluation for the frozen test protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .config import FrozenConfig


@dataclass(frozen=True)
class EvaluationResult:
    total_loss: float
    mean_loss: float
    correct_count: int
    sample_count: int
    test_error_percent: float


def _validate_policy(config: FrozenConfig) -> None:
    config.assert_source_unchanged()
    evaluation = config.data["evaluation"]
    if (
        evaluation["test_view"] != "original_32x32_single_view"
        or evaluation["metric"] != "test_error_percent"
        or evaluation["primary_checkpoint"] != "update_64000_final"
        or evaluation["best_test_checkpoint_selection"]
        or not config.data["experiment"]["test_selection_forbidden"]
    ):
        raise ValueError("unsupported or test-selecting evaluation policy")


def evaluate(
    model: nn.Module,
    batches: Iterable[Mapping[str, Any]],
    config: FrozenConfig,
) -> EvaluationResult:
    """Evaluate once without gradients, parameter updates, or buffer mutation."""

    _validate_policy(config)
    was_training = model.training
    total_loss = 0.0
    correct_count = 0
    sample_count = 0
    model.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                images = batch["image"]
                targets = batch["target"]
                logits = model(images)
                if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
                    raise ValueError("evaluation expects [N,C] logits and [N] targets")
                loss = F.cross_entropy(logits, targets, reduction="sum")
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError("evaluation loss must be finite")
                batch_size = int(targets.numel())
                total_loss += float(loss.item())
                correct_count += int((logits.argmax(dim=1) == targets).sum().item())
                sample_count += batch_size
    finally:
        model.train(was_training)
    if sample_count == 0:
        raise ValueError("evaluation requires at least one sample")
    return EvaluationResult(
        total_loss=total_loss,
        mean_loss=total_loss / sample_count,
        correct_count=correct_count,
        sample_count=sample_count,
        test_error_percent=100.0 * (sample_count - correct_count) / sample_count,
    )
