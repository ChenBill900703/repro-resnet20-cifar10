"""Learning-rate control indexed by completed optimizer updates, never epochs."""

from __future__ import annotations

from typing import Any, Mapping

from torch.optim import Optimizer

from ..config import FrozenConfig


class TrainingCompleteError(RuntimeError):
    """An update beyond the frozen 64,000-update budget was requested."""


class ExactUpdateLrController:
    """Own exact LR boundaries and the authoritative completed-update counter."""

    SCHEMA_VERSION = 1

    def __init__(self, optimizer: Optimizer, config: FrozenConfig) -> None:
        config.assert_source_unchanged()
        learning_rate = config.data["learning_rate"]
        if learning_rate["schedule_unit"] != "completed_optimizer_updates":
            raise ValueError("learning-rate schedule must use completed optimizer updates")
        self.optimizer = optimizer
        self.ranges = tuple(
            (
                int(item["first_update"]),
                int(item["last_update"]),
                float(item["value"]),
            )
            for item in learning_rate["ranges"]
        )
        self.max_updates = int(config.data["experiment"]["max_updates"])
        self.completed_updates = 0
        self._set_lr(self.lr_for_update(1))

    @property
    def current_lr(self) -> float:
        learning_rates = {float(group["lr"]) for group in self.optimizer.param_groups}
        if len(learning_rates) != 1:
            raise ValueError("all optimizer parameter groups must share one learning rate")
        return learning_rates.pop()

    @property
    def next_update_number(self) -> int:
        if self.completed_updates >= self.max_updates:
            raise TrainingCompleteError(
                f"update #{self.max_updates + 1:,} is forbidden; training ends at "
                f"#{self.max_updates:,}"
            )
        return self.completed_updates + 1

    def lr_for_update(self, update_number: int) -> float:
        if type(update_number) is not int or update_number < 1:
            raise ValueError("update_number must be a positive integer")
        if update_number > self.max_updates:
            raise TrainingCompleteError(
                f"update #{update_number:,} is forbidden; training ends at "
                f"#{self.max_updates:,}"
            )
        for first, last, value in self.ranges:
            if first <= update_number <= last:
                return value
        raise ValueError(f"no frozen learning rate covers update #{update_number:,}")

    def _set_lr(self, value: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(value)

    def prepare_next_update(self) -> tuple[int, float]:
        update_number = self.next_update_number
        learning_rate = self.lr_for_update(update_number)
        self._set_lr(learning_rate)
        return update_number, learning_rate

    def mark_update_completed(self, update_number: int) -> None:
        expected = self.next_update_number
        if update_number != expected:
            raise ValueError(
                f"completed update must be #{expected:,}, got #{update_number:,}"
            )
        expected_lr = self.lr_for_update(update_number)
        if self.current_lr != expected_lr:
            raise ValueError("optimizer learning rate changed during the update")
        self.completed_updates = update_number
        if self.completed_updates < self.max_updates:
            self._set_lr(self.lr_for_update(self.completed_updates + 1))

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "completed_updates": self.completed_updates,
            "current_lr": self.current_lr,
            "max_updates": self.max_updates,
            "ranges": self.ranges,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        required = {
            "schema_version",
            "completed_updates",
            "current_lr",
            "max_updates",
            "ranges",
        }
        if set(state) != required:
            raise ValueError("learning-rate controller state fields do not match schema")
        if state["schema_version"] != self.SCHEMA_VERSION:
            raise ValueError("unsupported learning-rate controller schema")
        if state["max_updates"] != self.max_updates or tuple(state["ranges"]) != self.ranges:
            raise ValueError("learning-rate controller state differs from frozen schedule")
        completed = state["completed_updates"]
        if type(completed) is not int or not 0 <= completed <= self.max_updates:
            raise ValueError("completed_updates is out of range")
        expected_lr = self.lr_for_update(
            completed + 1 if completed < self.max_updates else self.max_updates
        )
        if float(state["current_lr"]) != expected_lr:
            raise ValueError("saved current_lr disagrees with completed_updates")
        self.completed_updates = completed
        self._set_lr(expected_lr)
