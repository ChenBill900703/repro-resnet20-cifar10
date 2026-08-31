from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from resnet_repro.reproducibility import (
    WorkerSeeder,
    make_worker_generator,
    sample_rng_scope,
)
from resnet_repro.sampling import StatefulBatchSampler


class _SyntheticAugmentedDataset(Dataset):
    def __init__(self, size: int, *, base_seed: int, epoch: int = 0) -> None:
        self.size = size
        self.base_seed = base_seed
        self.epoch = epoch

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        with sample_rng_scope(self.base_seed, self.epoch, index):
            signature = torch.tensor(
                [random.random(), float(np.random.rand()), float(torch.rand(()))],
                dtype=torch.float64,
            )
        return {"index": index, "augmentation": signature}


def _loader(dataset: Dataset, sampler: StatefulBatchSampler) -> DataLoader:
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=False,
        worker_init_fn=WorkerSeeder(),
        generator=make_worker_generator(1),
    )


def _consume(
    iterator, sampler: StatefulBatchSampler, count: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(count):
        batch = next(iterator)
        indices = batch["index"]
        augmentation = batch["augmentation"]
        sampler.mark_batch_consumed(indices.tolist())
        batches.append((indices.clone(), augmentation.clone()))
    return batches


def _consume_sampler_epoch(sampler: StatefulBatchSampler) -> list[list[int]]:
    batches: list[list[int]] = []
    for batch in sampler:
        batches.append(batch)
        sampler.mark_batch_consumed(batch)
    return batches


def test_same_seed_produces_identical_permutation() -> None:
    first = _consume_sampler_epoch(StatefulBatchSampler(31, 5, base_seed=1))
    second = _consume_sampler_epoch(StatefulBatchSampler(31, 5, base_seed=1))
    assert first == second


def test_different_seed_produces_different_permutation() -> None:
    first = _consume_sampler_epoch(StatefulBatchSampler(31, 5, base_seed=1))
    second = _consume_sampler_epoch(StatefulBatchSampler(31, 5, base_seed=2))
    assert first != second


def test_sampler_epoch_is_nonreplacement_bijection() -> None:
    batches = _consume_sampler_epoch(StatefulBatchSampler(31, 5, base_seed=1))
    indices = [index for batch in batches for index in batch]
    assert len(indices) == len(set(indices)) == 31
    assert sorted(indices) == list(range(31))


def test_sampler_checkpoint_cursor_ignores_unconsumed_prefetch() -> None:
    sampler = StatefulBatchSampler(20, 4, base_seed=1)
    iterator = iter(sampler)
    first = next(iterator)
    next(iterator)
    next(iterator)
    sampler.mark_batch_consumed(first)

    state = sampler.state_dict()
    assert state["consumed_cursor"] == 4
    assert sampler.outstanding_batch_count == 2

    resumed = StatefulBatchSampler(20, 4, base_seed=1)
    resumed.load_state_dict(state)
    assert resumed.consumed_position == 4
    assert resumed.outstanding_batch_count == 0
    assert next(iter(resumed)) == state["permutation"][4:8].tolist()


def test_resume_starts_at_exact_first_unconsumed_batch() -> None:
    sampler = StatefulBatchSampler(23, 4, base_seed=1)
    iterator = iter(sampler)
    consumed = next(iterator)
    first_unconsumed = next(iterator)
    next(iterator)
    sampler.mark_batch_consumed(consumed)
    checkpoint = sampler.state_dict()

    resumed = StatefulBatchSampler(23, 4, base_seed=1)
    resumed.load_state_dict(checkpoint)
    assert next(iter(resumed)) == first_unconsumed


def test_acknowledgment_must_follow_issued_batch_order() -> None:
    sampler = StatefulBatchSampler(12, 4, base_seed=1)
    iterator = iter(sampler)
    first = next(iterator)
    second = next(iterator)

    with pytest.raises(RuntimeError, match="does not match"):
        sampler.mark_batch_consumed(second)
    assert sampler.consumed_position == 0
    assert sampler.outstanding_batch_count == 2

    sampler.mark_batch_consumed(first)
    sampler.mark_batch_consumed(second)
    assert sampler.consumed_position == 8
    with pytest.raises(RuntimeError, match="no issued batch"):
        fresh = StatefulBatchSampler(12, 4, base_seed=1)
        fresh.mark_batch_consumed(first)


def test_data_011_ckpt_003_exact_resume_with_prefetch_and_augmentation() -> None:
    dataset = _SyntheticAugmentedDataset(37, base_seed=1)
    reference_sampler = StatefulBatchSampler(len(dataset), 4, base_seed=1)
    reference_iterator = iter(_loader(dataset, reference_sampler))

    _consume(reference_iterator, reference_sampler, 3)
    checkpoint = reference_sampler.state_dict()
    reference_tail = _consume(reference_iterator, reference_sampler, 7)
    with pytest.raises(StopIteration):
        next(reference_iterator)

    resumed_sampler = StatefulBatchSampler(len(dataset), 4, base_seed=1)
    resumed_sampler.load_state_dict(checkpoint)
    resumed_iterator = iter(_loader(dataset, resumed_sampler))
    resumed_tail = _consume(resumed_iterator, resumed_sampler, 7)
    with pytest.raises(StopIteration):
        next(resumed_iterator)

    assert reference_sampler.epoch == resumed_sampler.epoch == 1
    for (reference_indices, reference_aug), (resumed_indices, resumed_aug) in zip(
        reference_tail, resumed_tail, strict=True
    ):
        torch.testing.assert_close(reference_indices, resumed_indices, rtol=0, atol=0)
        torch.testing.assert_close(reference_aug, resumed_aug, rtol=0, atol=0)


def test_sampler_state_rejects_loader_mismatch_and_corrupt_permutation() -> None:
    sampler = StatefulBatchSampler(10, 4, base_seed=1)
    iterator = iter(sampler)
    first = next(iterator)
    sampler.mark_batch_consumed(first)
    state = sampler.state_dict()

    wrong_batch_size = StatefulBatchSampler(10, 5, base_seed=1)
    with pytest.raises(ValueError, match="batch_size"):
        wrong_batch_size.load_state_dict(state)

    state["permutation"][0] = state["permutation"][1]
    with pytest.raises(ValueError, match="not a bijection"):
        StatefulBatchSampler(10, 4, base_seed=1).load_state_dict(state)


def test_sampler_state_rejects_unknown_schema_missing_field_and_bad_cursor() -> None:
    sampler = StatefulBatchSampler(10, 4, base_seed=1)
    iterator = iter(sampler)
    sampler.mark_batch_consumed(next(iterator))
    state = sampler.state_dict()

    bad_schema = dict(state, schema_version=999)
    with pytest.raises(ValueError, match="unsupported"):
        StatefulBatchSampler(10, 4, base_seed=1).load_state_dict(bad_schema)

    missing = dict(state)
    del missing["epoch"]
    with pytest.raises(ValueError, match="fields"):
        StatefulBatchSampler(10, 4, base_seed=1).load_state_dict(missing)

    bad_cursor = dict(state, consumed_cursor=3)
    with pytest.raises(ValueError, match="batch boundary"):
        StatefulBatchSampler(10, 4, base_seed=1).load_state_dict(bad_cursor)


def test_sampler_no_replacement_and_next_epoch_reshuffles() -> None:
    sampler = StatefulBatchSampler(17, 4, base_seed=1)
    first_epoch: list[int] = []
    for batch in sampler:
        first_epoch.extend(batch)
        sampler.mark_batch_consumed(batch)
    second_epoch: list[int] = []
    for batch in sampler:
        second_epoch.extend(batch)
        sampler.mark_batch_consumed(batch)

    assert sorted(first_epoch) == list(range(17))
    assert sorted(second_epoch) == list(range(17))
    assert first_epoch != second_epoch
    assert sampler.epoch == 2


def test_final_partial_batch_is_preserved() -> None:
    sampler = StatefulBatchSampler(10, 4, base_seed=1, drop_last=False)
    batches = _consume_sampler_epoch(sampler)
    assert [len(batch) for batch in batches] == [4, 4, 2]
    assert sampler.epoch == 1


def test_drop_last_omits_only_final_partial_batch() -> None:
    sampler = StatefulBatchSampler(10, 4, base_seed=1, drop_last=True)
    batches = _consume_sampler_epoch(sampler)
    indices = [index for batch in batches for index in batch]
    assert [len(batch) for batch in batches] == [4, 4]
    assert len(indices) == len(set(indices)) == 8
    assert sampler.epoch == 1
