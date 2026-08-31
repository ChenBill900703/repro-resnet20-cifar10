from __future__ import annotations

import gc

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from resnet_repro.config import FrozenConfig
from resnet_repro.data.cifar10 import Cifar10Metadata, IndexedDataset
from resnet_repro.data.loaders import EpochSynchronizedDataLoader, make_test_loader
from resnet_repro.data.transforms import TestingTransform, TrainingTransform
from resnet_repro.reproducibility import make_worker_generator
from resnet_repro.sampling import StatefulBatchSampler


class _TwentyImages(Dataset):
    def __len__(self) -> int:
        return 20

    def __getitem__(self, index: int):
        image = np.full((32, 32, 3), index, dtype=np.uint8)
        return image, index % 10


def _metadata(frozen_config: FrozenConfig) -> Cifar10Metadata:
    metadata = Cifar10Metadata.from_config(frozen_config.data)
    return metadata.with_sizes(train_size=20, test_size=20, official_split=False)


def _training_loader(
    frozen_config: FrozenConfig, sampler: StatefulBatchSampler
) -> EpochSynchronizedDataLoader:
    mean = torch.zeros(3, 32, 32)
    dataset = IndexedDataset(
        _TwentyImages(),
        metadata=_metadata(frozen_config),
        split="train",
        transform=TrainingTransform(frozen_config.data, mean, base_seed=1),
    )
    return EpochSynchronizedDataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=4,
        worker_init_fn=None,
        worker_generator=make_worker_generator(1),
    )


def _consume(iterator, loader: EpochSynchronizedDataLoader, count: int):
    batches = []
    for _ in range(count):
        batch = next(iterator)
        loader.mark_batch_consumed(batch["index"].tolist())
        batches.append((batch["index"].clone(), batch["image"].clone()))
    return batches


def test_training_nonreplacement_full_epoch_indices_and_epoch_sync(
    frozen_config: FrozenConfig,
) -> None:
    sampler = StatefulBatchSampler(20, 4, base_seed=1)
    loader = _training_loader(frozen_config, sampler)
    assert loader.dataset.epoch == 0
    batches = _consume(iter(loader), loader, 5)
    indices = torch.cat([batch[0] for batch in batches]).tolist()
    assert sorted(indices) == list(range(20))
    assert len(indices) == len(set(indices)) == 20
    assert sampler.epoch == loader.dataset.epoch == 1


def test_batch_preserves_official_indices(frozen_config: FrozenConfig) -> None:
    loader = _training_loader(
        frozen_config, StatefulBatchSampler(20, 4, base_seed=1)
    )
    batch = next(iter(loader))
    assert "index" in batch
    assert batch["index"].shape == (4,)


def test_test_loader_is_official_sequential_order(frozen_config: FrozenConfig) -> None:
    dataset = IndexedDataset(
        _TwentyImages(),
        metadata=_metadata(frozen_config),
        split="test",
        transform=TestingTransform(frozen_config.data, torch.zeros(3, 32, 32)),
    )
    loader = make_test_loader(dataset, batch_size=4, num_workers=0)
    indices = torch.cat([batch["index"] for batch in loader]).tolist()
    assert indices == list(range(20))


def test_four_worker_exact_resume_indices_and_augmentation(
    frozen_config: FrozenConfig,
) -> None:
    reference_sampler = StatefulBatchSampler(20, 4, base_seed=1)
    reference_loader = _training_loader(frozen_config, reference_sampler)
    reference_iterator = iter(reference_loader)
    _consume(reference_iterator, reference_loader, 2)
    checkpoint = reference_sampler.state_dict()
    reference_tail = _consume(reference_iterator, reference_loader, 3)
    with pytest.raises(StopIteration):
        next(reference_iterator)

    resumed_sampler = StatefulBatchSampler(20, 4, base_seed=1)
    resumed_sampler.load_state_dict(checkpoint)
    resumed_loader = _training_loader(frozen_config, resumed_sampler)
    resumed_iterator = iter(resumed_loader)
    resumed_tail = _consume(resumed_iterator, resumed_loader, 3)
    with pytest.raises(StopIteration):
        next(resumed_iterator)

    assert reference_sampler.epoch == resumed_sampler.epoch == 1
    assert reference_loader.dataset.epoch == resumed_loader.dataset.epoch == 1
    for (reference_indices, reference_images), (resumed_indices, resumed_images) in zip(
        reference_tail, resumed_tail, strict=True
    ):
        torch.testing.assert_close(reference_indices, resumed_indices, rtol=0, atol=0)
        torch.testing.assert_close(reference_images, resumed_images, rtol=0, atol=0)
    del reference_iterator, resumed_iterator
    gc.collect()
