"""DataLoader assembly with sampler/dataset epoch synchronization."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..reproducibility import WorkerSeeder
from ..sampling import StatefulBatchSampler
from .cifar10 import IndexedDataset


class EpochSynchronizedDataLoader:
    """Bind explicit dataset epoch state to acknowledged sampler progress."""

    def __init__(
        self,
        dataset: IndexedDataset,
        *,
        batch_sampler: StatefulBatchSampler,
        num_workers: int,
        worker_init_fn: Any = None,
        worker_generator: torch.Generator | None = None,
    ) -> None:
        if len(dataset) != batch_sampler.dataset_size:
            raise ValueError("dataset size and stateful sampler size must match")
        if type(num_workers) is not int or num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer")
        if dataset.split != "train":
            raise ValueError("epoch-synchronized loader is only valid for training")
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.num_workers = num_workers
        self.dataset.set_epoch(self.batch_sampler.epoch)
        self._loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            worker_init_fn=WorkerSeeder() if worker_init_fn is None else worker_init_fn,
            generator=worker_generator,
            persistent_workers=False,
        )

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        self.dataset.set_epoch(self.batch_sampler.epoch)
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self.batch_sampler)

    def mark_batch_consumed(self, indices: Sequence[int]) -> None:
        self.batch_sampler.mark_batch_consumed(indices)
        self.dataset.set_epoch(self.batch_sampler.epoch)


def make_test_loader(
    dataset: IndexedDataset,
    *,
    batch_size: int,
    num_workers: int,
    worker_generator: torch.Generator | None = None,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Build a deterministic sequential test loader with no random sampler."""

    if dataset.split != "test":
        raise ValueError("test loader requires the official test split wrapper")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=WorkerSeeder() if num_workers else None,
        generator=worker_generator,
        persistent_workers=False,
    )
