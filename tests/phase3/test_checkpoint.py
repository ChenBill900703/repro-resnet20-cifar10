from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import nn

from resnet_repro.config import FrozenConfig
from resnet_repro.reproducibility import make_worker_generator
from resnet_repro.sampling import StatefulBatchSampler
from resnet_repro.training.checkpoint import CheckpointError, load_checkpoint, save_checkpoint
from resnet_repro.training.optimizer import build_sgd_optimizer
from resnet_repro.training.schedule import ExactUpdateLrController


MEAN_HASH = "SYNTHETIC-PHASE3-MEAN"


class _TinyModel(nn.Module):
    model_name = "synthetic"

    def __init__(self, out_features: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(3, out_features)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value)


@dataclass
class _DatasetState:
    epoch: int = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def _objects(frozen_config: FrozenConfig, *, out_features: int = 2):
    model = _TinyModel(out_features)
    optimizer = build_sgd_optimizer(model, frozen_config)
    controller = ExactUpdateLrController(optimizer, frozen_config)
    sampler = StatefulBatchSampler(8, 2, base_seed=1)
    dataset = _DatasetState()
    generator = make_worker_generator(1)
    return model, optimizer, controller, sampler, dataset, generator


def _save(tmp_path, frozen_config: FrozenConfig):
    objects = _objects(frozen_config)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model=objects[0],
        optimizer=objects[1],
        lr_controller=objects[2],
        sampler=objects[3],
        dataset=objects[4],
        dataloader_generator=objects[5],
        config=frozen_config,
        mean_artifact_sha256=MEAN_HASH,
    )
    return path, objects


def test_checkpoint_contains_every_required_state_domain(
    tmp_path, frozen_config: FrozenConfig
) -> None:
    path, _ = _save(tmp_path, frozen_config)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert {
        "model_state",
        "optimizer_state",
        "completed_updates",
        "current_lr",
        "scheduler_state",
        "sampler_state",
        "dataset_epoch",
        "python_rng_state",
        "numpy_rng_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_states",
        "dataloader_generator_state",
        "frozen_config_payload",
        "frozen_config_sha256",
        "source_commit",
        "source_dirty",
        "environment_fingerprint",
    } <= set(payload)
    assert payload["frozen_config_payload"] == frozen_config.raw_bytes
    assert payload["frozen_config_sha256"] == frozen_config.sha256


def test_checkpoint_roundtrip_restores_lr_sampler_dataset_and_model(
    tmp_path, frozen_config: FrozenConfig
) -> None:
    path, source = _save(tmp_path, frozen_config)
    restored = _objects(frozen_config)
    metadata = load_checkpoint(
        path,
        model=restored[0],
        optimizer=restored[1],
        lr_controller=restored[2],
        sampler=restored[3],
        dataset=restored[4],
        dataloader_generator=restored[5],
        config=frozen_config,
        mean_artifact_sha256=MEAN_HASH,
    )
    assert metadata["completed_updates"] == 0
    assert metadata["current_lr"] == 0.1
    assert restored[4].epoch == restored[3].epoch == 0
    torch.testing.assert_close(
        restored[5].get_state(), source[5].get_state(), rtol=0, atol=0
    )
    for name, value in source[0].state_dict().items():
        torch.testing.assert_close(restored[0].state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("frozen_config_sha256", "BAD", "config SHA-256"),
        ("mean_artifact_sha256", "BAD", "mean artifact"),
        ("source_commit", "BAD", "source commit"),
    ],
)
def test_checkpoint_rejects_incompatible_metadata(
    tmp_path,
    frozen_config: FrozenConfig,
    field: str,
    replacement: str,
    message: str,
) -> None:
    path, _ = _save(tmp_path, frozen_config)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload[field] = replacement
    torch.save(payload, path)
    restored = _objects(frozen_config)
    with pytest.raises(CheckpointError, match=message):
        load_checkpoint(
            path,
            model=restored[0],
            optimizer=restored[1],
            lr_controller=restored[2],
            sampler=restored[3],
            dataset=restored[4],
            dataloader_generator=restored[5],
            config=frozen_config,
            mean_artifact_sha256=MEAN_HASH,
        )


def test_checkpoint_rejects_missing_field_corruption_and_model_structure(
    tmp_path, frozen_config: FrozenConfig
) -> None:
    path, _ = _save(tmp_path, frozen_config)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    del payload["python_rng_state"]
    torch.save(payload, path)
    restored = _objects(frozen_config)
    with pytest.raises(CheckpointError, match="fields do not match"):
        load_checkpoint(
            path,
            model=restored[0], optimizer=restored[1], lr_controller=restored[2],
            sampler=restored[3], dataset=restored[4], dataloader_generator=restored[5],
            config=frozen_config, mean_artifact_sha256=MEAN_HASH,
        )

    path.write_bytes(b"not a checkpoint")
    with pytest.raises(CheckpointError, match="cannot read"):
        load_checkpoint(
            path,
            model=restored[0], optimizer=restored[1], lr_controller=restored[2],
            sampler=restored[3], dataset=restored[4], dataloader_generator=restored[5],
            config=frozen_config, mean_artifact_sha256=MEAN_HASH,
        )

    path, _ = _save(tmp_path, frozen_config)
    wrong = _objects(frozen_config, out_features=3)
    with pytest.raises(CheckpointError, match="state restoration"):
        load_checkpoint(
            path,
            model=wrong[0], optimizer=wrong[1], lr_controller=wrong[2],
            sampler=wrong[3], dataset=wrong[4], dataloader_generator=wrong[5],
            config=frozen_config, mean_artifact_sha256=MEAN_HASH,
        )


def test_checkpoint_rejects_sampler_dataset_epoch_disagreement(
    tmp_path, frozen_config: FrozenConfig
) -> None:
    objects = _objects(frozen_config)
    objects[4].epoch = 1
    with pytest.raises(CheckpointError, match="does not match sampler"):
        save_checkpoint(
            tmp_path / "bad.pt",
            model=objects[0], optimizer=objects[1], lr_controller=objects[2],
            sampler=objects[3], dataset=objects[4], dataloader_generator=objects[5],
            config=frozen_config, mean_artifact_sha256=MEAN_HASH,
        )
