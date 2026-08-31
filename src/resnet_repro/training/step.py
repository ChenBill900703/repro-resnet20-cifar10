"""One successful optimizer update, without Phase 4 run orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch
from torch import nn
from torch.optim import Optimizer

from .schedule import ExactUpdateLrController


class _BatchAcknowledger(Protocol):
    def mark_batch_consumed(self, indices: Sequence[int]) -> None: ...


@dataclass(frozen=True)
class TrainingStepResult:
    update_number: int
    learning_rate: float
    loss: float
    indices: tuple[int, ...]
    correct_count: int
    sample_count: int


def train_one_update(
    model: nn.Module,
    optimizer: Optimizer,
    lr_controller: ExactUpdateLrController,
    *,
    images: torch.Tensor,
    targets: torch.Tensor,
    indices: Sequence[int],
    batch_acknowledger: _BatchAcknowledger,
    criterion: nn.Module | None = None,
) -> TrainingStepResult:
    """Perform one update and acknowledge sampler progress only after it succeeds."""

    update_number, learning_rate = lr_controller.prepare_next_update()
    criterion = nn.CrossEntropyLoss() if criterion is None else criterion
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = criterion(logits, targets)
    if loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("training loss must be a finite scalar")
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not gradients or any(gradient is None for gradient in gradients):
        raise RuntimeError("every learnable parameter must receive a gradient")
    if any(not bool(torch.isfinite(gradient).all().item()) for gradient in gradients):
        raise FloatingPointError("training gradients must be finite")
    optimizer.step()
    consumed_indices = tuple(int(index) for index in indices)
    correct_count = int((logits.detach().argmax(dim=1) == targets).sum().item())
    sample_count = int(targets.numel())
    batch_acknowledger.mark_batch_consumed(consumed_indices)
    lr_controller.mark_update_completed(update_number)
    return TrainingStepResult(
        update_number=update_number,
        learning_rate=learning_rate,
        loss=float(loss.detach().item()),
        indices=consumed_indices,
        correct_count=correct_count,
        sample_count=sample_count,
    )
