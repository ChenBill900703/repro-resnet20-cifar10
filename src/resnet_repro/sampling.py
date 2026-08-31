"""Checkpointable, non-replacement sampling with prefetch-safe progress."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch
from torch.utils.data import Sampler

from .reproducibility import make_dataloader_generator


class StatefulBatchSampler(Sampler[list[int]]):
    """Random batch sampler whose checkpoint cursor means *consumed* samples.

    DataLoader workers may request batches ahead of the optimizer.  Merely
    serializing a RandomSampler generator therefore skips prefetched batches
    after resume.  This sampler tracks issued batches separately and advances
    its checkpoint cursor only when the training loop acknowledges a batch via
    :meth:`mark_batch_consumed`.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        *,
        base_seed: int,
        drop_last: bool = False,
    ) -> None:
        if type(dataset_size) is not int or dataset_size <= 0:
            raise ValueError("dataset_size must be a positive integer")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if type(drop_last) is not bool:
            raise ValueError("drop_last must be bool")
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.base_seed = base_seed
        self.generator = make_dataloader_generator(base_seed)
        self.epoch = 0
        self._permutation: torch.Tensor | None = None
        self._consumed_position = 0
        self._issued_position = 0
        self._outstanding: deque[tuple[int, ...]] = deque()

    @property
    def consumed_position(self) -> int:
        return self._consumed_position

    @property
    def outstanding_batch_count(self) -> int:
        return len(self._outstanding)

    def _usable_size(self) -> int:
        if self.drop_last:
            return self.dataset_size - (self.dataset_size % self.batch_size)
        return self.dataset_size

    def _start_epoch(self) -> None:
        self._permutation = torch.randperm(self.dataset_size, generator=self.generator)
        self._consumed_position = 0
        self._issued_position = 0

    def __iter__(self) -> Iterator[list[int]]:
        if self._outstanding:
            raise RuntimeError("cannot create a second iterator with unconsumed prefetched batches")
        if self._permutation is None:
            self._start_epoch()
        assert self._permutation is not None
        self._issued_position = self._consumed_position
        usable_size = self._usable_size()
        while self._issued_position < usable_size:
            end = min(self._issued_position + self.batch_size, usable_size)
            batch = tuple(
                int(item)
                for item in self._permutation[self._issued_position:end].tolist()
            )
            self._issued_position = end
            self._outstanding.append(batch)
            yield list(batch)
            if self._permutation is None:
                return

    def mark_batch_consumed(self, indices: Sequence[int]) -> None:
        """Acknowledge one yielded batch after its optimizer update succeeds."""

        if not self._outstanding:
            raise RuntimeError("no issued batch is waiting to be acknowledged")
        expected = self._outstanding[0]
        actual = tuple(int(index) for index in indices)
        if actual != expected:
            raise RuntimeError(
                f"consumed batch does not match next issued batch: expected {expected}, got {actual}"
            )
        self._outstanding.popleft()
        self._consumed_position += len(expected)
        if self._consumed_position == self._usable_size():
            if self._outstanding:
                raise RuntimeError("epoch ended while prefetched batches remain unacknowledged")
            self.epoch += 1
            self._permutation = None
            self._consumed_position = 0
            self._issued_position = 0

    def state_dict(self) -> dict[str, Any]:
        """Serialize only acknowledged progress, deliberately rewinding prefetch."""

        return {
            "schema_version": self.SCHEMA_VERSION,
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "drop_last": self.drop_last,
            "base_seed": self.base_seed,
            "epoch": self.epoch,
            "consumed_cursor": self._consumed_position,
            "permutation": self._permutation.clone() if self._permutation is not None else None,
            "generator_state": self.generator.get_state().clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        required = {
            "schema_version",
            "dataset_size",
            "batch_size",
            "drop_last",
            "base_seed",
            "epoch",
            "consumed_cursor",
            "permutation",
            "generator_state",
        }
        if set(state) != required:
            raise ValueError("sampler state fields do not match schema version 1")
        if state["schema_version"] != self.SCHEMA_VERSION:
            raise ValueError(f"unsupported sampler state schema: {state['schema_version']}")
        for field in ("dataset_size", "batch_size", "drop_last", "base_seed"):
            if state[field] != getattr(self, field):
                raise ValueError(f"sampler state {field} does not match current loader")
        epoch = state["epoch"]
        position = state["consumed_cursor"]
        permutation = state["permutation"]
        if type(epoch) is not int or epoch < 0:
            raise ValueError("sampler epoch must be a non-negative integer")
        if type(position) is not int or not 0 <= position <= self._usable_size():
            raise ValueError("sampler consumed_cursor is out of range")
        if position % self.batch_size and position != self._usable_size():
            raise ValueError("sampler consumed_cursor is not a batch boundary")
        if permutation is None:
            if position != 0:
                raise ValueError("sampler state without a permutation must be at epoch start")
        else:
            if not isinstance(permutation, torch.Tensor) or permutation.dtype != torch.int64:
                raise ValueError("sampler permutation must be an int64 tensor")
            if permutation.shape != (self.dataset_size,):
                raise ValueError("sampler permutation length differs from dataset size")
            expected = torch.arange(self.dataset_size, dtype=torch.int64)
            if not torch.equal(torch.sort(permutation.cpu()).values, expected):
                raise ValueError("sampler permutation is not a bijection of dataset indices")
        generator_state = state["generator_state"]
        if not isinstance(generator_state, torch.Tensor):
            raise ValueError("sampler generator_state must be a tensor")

        self.generator.set_state(generator_state)
        self.epoch = epoch
        self._permutation = permutation.clone() if permutation is not None else None
        self._consumed_position = position
        self._issued_position = position
        self._outstanding.clear()

    def __len__(self) -> int:
        remaining = self._usable_size() - self._consumed_position
        if self.drop_last:
            return remaining // self.batch_size
        return (remaining + self.batch_size - 1) // self.batch_size
