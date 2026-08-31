from __future__ import annotations

import copy
import gc
import random
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from resnet_repro.reproducibility import (
    RNGState,
    WorkerSeeder,
    capture_rng_state,
    configure_global_rng,
    make_worker_generator,
    restore_rng_state,
    sample_rng_scope,
)
from resnet_repro.sampling import StatefulBatchSampler


class _TwentySampleDataset(Dataset):
    """Synthetic-only dataset carrying explicit epoch and official indices."""

    def __init__(self, *, base_seed: int, epoch: int = 0) -> None:
        self.base_seed = base_seed
        self.epoch = epoch

    def __len__(self) -> int:
        return 20

    def __getitem__(self, official_index: int) -> dict[str, torch.Tensor | int]:
        with sample_rng_scope(self.base_seed, self.epoch, official_index):
            augmentation = torch.tensor(
                [random.random(), float(np.random.rand()), float(torch.rand(()))],
                dtype=torch.float64,
            )
        base = torch.tensor(
            [official_index / 20.0, (official_index % 5) / 5.0, 1.0],
            dtype=torch.float64,
        )
        features = base + 0.01 * augmentation
        target = (0.3 * features[0] - 0.2 * features[1] + 0.1).reshape(1)
        return {
            "index": official_index,
            "augmentation": augmentation,
            "features": features,
            "target": target,
        }


def _loader(
    dataset: Dataset,
    sampler: StatefulBatchSampler,
    worker_generator: torch.Generator,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=2,
        prefetch_factor=2,
        persistent_workers=False,
        worker_init_fn=WorkerSeeder(),
        generator=worker_generator,
    )


def _consume_updates(
    iterator: Any,
    sampler: StatefulBatchSampler,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    count: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    trace: list[tuple[torch.Tensor, torch.Tensor]] = []
    model.train()
    for _ in range(count):
        batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch["features"])
        loss = torch.nn.functional.mse_loss(prediction, batch["target"])
        loss.backward()
        optimizer.step()
        indices = batch["index"]
        sampler.mark_batch_consumed(indices.tolist())
        trace.append((indices.clone(), batch["augmentation"].clone()))
    return trace


def _new_model_and_optimizer() -> tuple[nn.Module, torch.optim.Optimizer]:
    model = nn.Linear(3, 1, bias=True, dtype=torch.float64)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    return model, optimizer


def _assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, Mapping):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def _assert_rng_equal(left: RNGState, right: RNGState) -> None:
    assert left.schema_version == right.schema_version
    assert left.python == right.python
    assert left.numpy[0] == right.numpy[0]
    np.testing.assert_array_equal(left.numpy[1], right.numpy[1])
    assert left.numpy[2:] == right.numpy[2:]
    torch.testing.assert_close(left.torch_cpu, right.torch_cpu, rtol=0, atol=0)
    assert len(left.torch_cuda) == len(right.torch_cuda)
    for left_cuda, right_cuda in zip(left.torch_cuda, right.torch_cuda, strict=True):
        torch.testing.assert_close(left_cuda, right_cuda, rtol=0, atol=0)
    assert left.dataloader_generator is right.dataloader_generator is None


def _shutdown_unfinished_iterator(iterator: Any) -> None:
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if shutdown is not None:
        shutdown()
    del iterator
    gc.collect()


def test_fake_data_uninterrupted_and_checkpoint_resume_are_exact() -> None:
    # Uninterrupted five-update reference.
    configure_global_rng(123, deterministic_algorithms=True, cudnn_benchmark=False, tf32=False)
    reference_model, reference_optimizer = _new_model_and_optimizer()
    reference_sampler = StatefulBatchSampler(20, 4, base_seed=1)
    reference_worker_generator = make_worker_generator(1)
    reference_iterator = iter(
        _loader(_TwentySampleDataset(base_seed=1), reference_sampler, reference_worker_generator)
    )
    reference_trace = _consume_updates(
        reference_iterator, reference_sampler, reference_model, reference_optimizer, 5
    )
    reference_model_state = copy.deepcopy(reference_model.state_dict())
    reference_optimizer_state = copy.deepcopy(reference_optimizer.state_dict())
    reference_rng_state = capture_rng_state()
    reference_sampler_state = reference_sampler.state_dict()
    reference_worker_state = reference_worker_generator.get_state().clone()

    # Identical start, checkpoint after two successful optimizer updates.
    configure_global_rng(123, deterministic_algorithms=True, cudnn_benchmark=False, tf32=False)
    interrupted_model, interrupted_optimizer = _new_model_and_optimizer()
    interrupted_sampler = StatefulBatchSampler(20, 4, base_seed=1)
    interrupted_worker_generator = make_worker_generator(1)
    interrupted_iterator = iter(
        _loader(_TwentySampleDataset(base_seed=1), interrupted_sampler, interrupted_worker_generator)
    )
    interrupted_trace = _consume_updates(
        interrupted_iterator,
        interrupted_sampler,
        interrupted_model,
        interrupted_optimizer,
        2,
    )
    checkpoint = {
        "model": copy.deepcopy(interrupted_model.state_dict()),
        "optimizer": copy.deepcopy(interrupted_optimizer.state_dict()),
        "rng": capture_rng_state(),
        "sampler": interrupted_sampler.state_dict(),
    }
    _shutdown_unfinished_iterator(interrupted_iterator)

    # Recreate runtime objects and resume from the consumed cursor.
    resumed_model, resumed_optimizer = _new_model_and_optimizer()
    resumed_model.load_state_dict(checkpoint["model"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    resumed_sampler = StatefulBatchSampler(20, 4, base_seed=1)
    resumed_sampler.load_state_dict(checkpoint["sampler"])
    resumed_worker_generator = make_worker_generator(1)
    restore_rng_state(checkpoint["rng"])
    resumed_iterator = iter(
        _loader(_TwentySampleDataset(base_seed=1), resumed_sampler, resumed_worker_generator)
    )
    resumed_trace = _consume_updates(
        resumed_iterator, resumed_sampler, resumed_model, resumed_optimizer, 3
    )

    combined_trace = interrupted_trace + resumed_trace
    assert len(reference_trace) == len(combined_trace) == 5
    for (reference_indices, reference_aug), (resumed_indices, resumed_aug) in zip(
        reference_trace, combined_trace, strict=True
    ):
        torch.testing.assert_close(reference_indices, resumed_indices, rtol=0, atol=0)
        torch.testing.assert_close(reference_aug, resumed_aug, rtol=0, atol=0)

    _assert_nested_equal(reference_model_state, resumed_model.state_dict())
    _assert_nested_equal(reference_optimizer_state, resumed_optimizer.state_dict())
    _assert_rng_equal(reference_rng_state, capture_rng_state())
    _assert_nested_equal(reference_sampler_state, resumed_sampler.state_dict())
    torch.testing.assert_close(
        reference_worker_state,
        resumed_worker_generator.get_state(),
        rtol=0,
        atol=0,
    )
