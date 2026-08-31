from __future__ import annotations

import copy
import gc
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from resnet_repro.config import FrozenConfig
from resnet_repro.data import Cifar10Metadata, EpochSynchronizedDataLoader, IndexedDataset
from resnet_repro.data.transforms import TrainingTransform
from resnet_repro.reproducibility import (
    capture_rng_state,
    configure_global_rng,
    make_worker_generator,
    restore_rng_state,
)
from resnet_repro.sampling import StatefulBatchSampler
from resnet_repro.training.checkpoint import load_checkpoint, save_checkpoint
from resnet_repro.training.optimizer import build_sgd_optimizer
from resnet_repro.training.schedule import ExactUpdateLrController
from resnet_repro.training.step import train_one_update


class _Images(Dataset):
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        grid = np.arange(32 * 32 * 3, dtype=np.uint16).reshape(32, 32, 3)
        image = ((grid + index * 17) % 256).astype(np.uint8)
        return image, index % 10


class _TinyClassifier(nn.Module):
    model_name = "synthetic_resume"

    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(3 * 32 * 32, 10)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.flatten(value / 255.0, 1))


@dataclass
class _RunObjects:
    model: _TinyClassifier
    optimizer: torch.optim.Optimizer
    controller: ExactUpdateLrController
    sampler: StatefulBatchSampler
    dataset: IndexedDataset
    loader: EpochSynchronizedDataLoader
    worker_generator: torch.Generator


def _build(frozen_config: FrozenConfig) -> _RunObjects:
    metadata = Cifar10Metadata.from_config(frozen_config.data).with_sizes(
        train_size=48, test_size=12, official_split=False
    )
    dataset = IndexedDataset(
        _Images(48),
        metadata=metadata,
        split="train",
        transform=TrainingTransform(
            frozen_config.data, torch.zeros(3, 32, 32), base_seed=1
        ),
    )
    sampler = StatefulBatchSampler(48, 4, base_seed=1)
    worker_generator = make_worker_generator(1)
    loader = EpochSynchronizedDataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=4,
        worker_generator=worker_generator,
    )
    model = _TinyClassifier()
    optimizer = build_sgd_optimizer(model, frozen_config)
    controller = ExactUpdateLrController(optimizer, frozen_config)
    return _RunObjects(
        model, optimizer, controller, sampler, dataset, loader, worker_generator
    )


def _run_updates(objects: _RunObjects, count: int):
    iterator = iter(objects.loader)
    records = []
    for _ in range(count):
        batch = next(iterator)
        rng_factor = 1.0 + 0.001 * (
            random.random() + float(np.random.rand()) + float(torch.rand(()).item())
        )
        images = batch["image"] * rng_factor
        result = train_one_update(
            objects.model,
            objects.optimizer,
            objects.controller,
            images=images,
            targets=batch["target"],
            indices=batch["index"].tolist(),
            batch_acknowledger=objects.loader,
        )
        records.append(
            (
                result.update_number,
                result.learning_rate,
                result.loss,
                batch["index"].clone(),
                batch["image"].clone(),
            )
        )
    del iterator
    gc.collect()
    return records


def _snapshot(objects: _RunObjects):
    momentum = [
        objects.optimizer.state[parameter]["momentum_buffer"].detach().clone()
        for parameter in objects.model.parameters()
    ]
    return {
        "model": copy.deepcopy(objects.model.state_dict()),
        "momentum": momentum,
        "controller": copy.deepcopy(objects.controller.state_dict()),
        "sampler": copy.deepcopy(objects.sampler.state_dict()),
        "rng": capture_rng_state(),
    }


def test_synthetic_uninterrupted_vs_checkpoint_resume_is_exact(
    tmp_path, frozen_config: FrozenConfig
) -> None:
    original_rng = capture_rng_state()
    try:
        configure_global_rng(123)
        uninterrupted = _build(frozen_config)
        uninterrupted_records = _run_updates(uninterrupted, 10)
        uninterrupted_state = _snapshot(uninterrupted)
        del uninterrupted
        gc.collect()

        configure_global_rng(123)
        interrupted = _build(frozen_config)
        resumed_records = _run_updates(interrupted, 4)
        checkpoint = tmp_path / "exact-resume.pt"
        save_checkpoint(
            checkpoint,
            model=interrupted.model,
            optimizer=interrupted.optimizer,
            lr_controller=interrupted.controller,
            sampler=interrupted.sampler,
            dataset=interrupted.dataset,
            dataloader_generator=interrupted.worker_generator,
            config=frozen_config,
            mean_artifact_sha256="SYNTHETIC-PHASE3-MEAN",
        )
        del interrupted
        gc.collect()

        resumed = _build(frozen_config)
        load_checkpoint(
            checkpoint,
            model=resumed.model,
            optimizer=resumed.optimizer,
            lr_controller=resumed.controller,
            sampler=resumed.sampler,
            dataset=resumed.dataset,
            dataloader_generator=resumed.worker_generator,
            config=frozen_config,
            mean_artifact_sha256="SYNTHETIC-PHASE3-MEAN",
        )
        resumed_records.extend(_run_updates(resumed, 6))
        resumed_state = _snapshot(resumed)

        assert len(uninterrupted_records) == len(resumed_records) == 10
        for expected, actual in zip(uninterrupted_records, resumed_records, strict=True):
            assert expected[:3] == actual[:3]
            torch.testing.assert_close(expected[3], actual[3], rtol=0, atol=0)
            torch.testing.assert_close(expected[4], actual[4], rtol=0, atol=0)
        assert uninterrupted_state["controller"] == resumed_state["controller"]
        for name, value in uninterrupted_state["model"].items():
            torch.testing.assert_close(resumed_state["model"][name], value, rtol=0, atol=0)
        for expected, actual in zip(
            uninterrupted_state["momentum"], resumed_state["momentum"], strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected_sampler = uninterrupted_state["sampler"]
        actual_sampler = resumed_state["sampler"]
        assert expected_sampler.keys() == actual_sampler.keys()
        for key in expected_sampler:
            if isinstance(expected_sampler[key], torch.Tensor):
                torch.testing.assert_close(actual_sampler[key], expected_sampler[key], rtol=0, atol=0)
            else:
                assert actual_sampler[key] == expected_sampler[key]
        expected_rng = uninterrupted_state["rng"]
        actual_rng = resumed_state["rng"]
        assert expected_rng.python == actual_rng.python
        np.testing.assert_array_equal(expected_rng.numpy[1], actual_rng.numpy[1])
        torch.testing.assert_close(actual_rng.torch_cpu, expected_rng.torch_cpu, rtol=0, atol=0)
        assert len(actual_rng.torch_cuda) == len(expected_rng.torch_cuda)
        for expected, actual in zip(
            expected_rng.torch_cuda, actual_rng.torch_cuda, strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        restore_rng_state(original_rng)
