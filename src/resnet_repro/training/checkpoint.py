"""Validated, atomic Phase 3 checkpoint save and restore."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Protocol

import torch
from torch import nn
from torch.optim import Optimizer

from ..config import FrozenConfig
from ..reproducibility import RNGState, capture_rng_state, environment_fingerprint, restore_rng_state
from ..sampling import StatefulBatchSampler
from .schedule import ExactUpdateLrController


class CheckpointError(RuntimeError):
    """A checkpoint is corrupt, incomplete, or incompatible with the current run."""


class _EpochDataset(Protocol):
    epoch: int

    def set_epoch(self, epoch: int) -> None: ...


SCHEMA_VERSION = 1
_REQUIRED_FIELDS = {
    "schema_version",
    "model_name",
    "model_state",
    "model_training",
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
    "augmentation_rng_policy",
    "frozen_config_payload",
    "frozen_config_sha256",
    "source_commit",
    "source_dirty",
    "environment_fingerprint",
    "mean_artifact_sha256",
}


def _run_git(repo: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CheckpointError(f"cannot inspect source repository: {exc}") from exc
    return result.stdout.strip()


def source_provenance(config: FrozenConfig) -> tuple[str, bool]:
    repo = Path(_run_git(config.path.parent, "rev-parse", "--show-toplevel"))
    commit = _run_git(repo, "rev-parse", "HEAD")
    dirty = bool(_run_git(repo, "status", "--porcelain"))
    return commit, dirty


def _requirements_lock(config: FrozenConfig) -> Path:
    return config.path.parent.parent / "environment" / "requirements-lock.txt"


def _model_name(model: nn.Module) -> str:
    name = getattr(model, "model_name", None)
    if not isinstance(name, str) or not name:
        raise CheckpointError("checkpointed model must expose a non-empty model_name")
    return name


def _validate_dataset_sampler_epoch(
    dataset: _EpochDataset, sampler: StatefulBatchSampler
) -> None:
    if type(dataset.epoch) is not int or dataset.epoch < 0:
        raise CheckpointError("dataset epoch must be a non-negative integer")
    if dataset.epoch != sampler.epoch:
        raise CheckpointError(
            f"dataset epoch {dataset.epoch} does not match sampler epoch {sampler.epoch}"
        )


def _expected_current_lr(controller: ExactUpdateLrController, completed: int) -> float:
    update = completed + 1 if completed < controller.max_updates else controller.max_updates
    return controller.lr_for_update(update)


def save_checkpoint(
    destination: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    lr_controller: ExactUpdateLrController,
    sampler: StatefulBatchSampler,
    dataset: _EpochDataset,
    dataloader_generator: torch.Generator,
    config: FrozenConfig,
    mean_artifact_sha256: str,
) -> Path:
    """Atomically save all state needed for exact continuation."""

    config.assert_source_unchanged()
    _validate_dataset_sampler_epoch(dataset, sampler)
    if not isinstance(dataloader_generator, torch.Generator):
        raise CheckpointError("a DataLoader generator is required")
    if not isinstance(mean_artifact_sha256, str) or not mean_artifact_sha256:
        raise CheckpointError("mean_artifact_sha256 must be a non-empty string")
    completed = lr_controller.completed_updates
    expected_lr = _expected_current_lr(lr_controller, completed)
    if lr_controller.current_lr != expected_lr:
        raise CheckpointError("current optimizer LR disagrees with completed updates")
    source_commit, source_dirty = source_provenance(config)
    rng = capture_rng_state(dataloader_generator)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_name": _model_name(model),
        "model_state": model.state_dict(),
        "model_training": model.training,
        "optimizer_state": optimizer.state_dict(),
        "completed_updates": completed,
        "current_lr": lr_controller.current_lr,
        "scheduler_state": lr_controller.state_dict(),
        "sampler_state": sampler.state_dict(),
        "dataset_epoch": dataset.epoch,
        "python_rng_state": rng.python,
        "numpy_rng_state": rng.numpy,
        "torch_cpu_rng_state": rng.torch_cpu,
        "torch_cuda_rng_states": rng.torch_cuda,
        "dataloader_generator_state": rng.dataloader_generator,
        "augmentation_rng_policy": {
            "schema_version": 1,
            "base_seed": config.data["reproducibility"]["seed"],
            "scope": config.data["reproducibility"]["augmentation_rng"]["scope"],
            "seed_components": tuple(
                config.data["reproducibility"]["augmentation_rng"]["seed_components"]
            ),
        },
        "frozen_config_payload": config.raw_bytes,
        "frozen_config_sha256": config.sha256,
        "source_commit": source_commit,
        "source_dirty": source_dirty,
        "environment_fingerprint": environment_fingerprint(_requirements_lock(config)),
        "mean_artifact_sha256": mean_artifact_sha256,
    }
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as temporary:
            temporary_name = temporary.name
            torch.save(payload, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    except Exception:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass
        raise
    return path


def _read_checkpoint(source: str | Path) -> dict[str, Any]:
    try:
        payload = torch.load(Path(source), map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CheckpointError(f"cannot read checkpoint: {exc}") from exc
    if not isinstance(payload, dict):
        raise CheckpointError("checkpoint root must be a mapping")
    if set(payload) != _REQUIRED_FIELDS:
        missing = sorted(_REQUIRED_FIELDS - set(payload))
        unknown = sorted(set(payload) - _REQUIRED_FIELDS)
        raise CheckpointError(
            f"checkpoint fields do not match schema; missing={missing}, unknown={unknown}"
        )
    if payload["schema_version"] != SCHEMA_VERSION:
        raise CheckpointError(f"unsupported checkpoint schema: {payload['schema_version']}")
    return payload


def _validate_metadata(
    payload: Mapping[str, Any],
    *,
    model: nn.Module,
    lr_controller: ExactUpdateLrController,
    sampler: StatefulBatchSampler,
    config: FrozenConfig,
    mean_artifact_sha256: str,
) -> None:
    config.assert_source_unchanged()
    if payload["model_name"] != _model_name(model):
        raise CheckpointError("checkpoint model_name does not match current model")
    if payload["frozen_config_payload"] != config.raw_bytes:
        raise CheckpointError("checkpoint frozen config payload does not match")
    if payload["frozen_config_sha256"] != config.sha256:
        raise CheckpointError("checkpoint frozen config SHA-256 does not match")
    if payload["mean_artifact_sha256"] != mean_artifact_sha256:
        raise CheckpointError("checkpoint mean artifact SHA-256 does not match")
    current_commit, current_dirty = source_provenance(config)
    if payload["source_commit"] != current_commit or payload["source_dirty"] != current_dirty:
        raise CheckpointError("checkpoint source commit/dirty state does not match")
    current_environment = environment_fingerprint(_requirements_lock(config))
    if payload["environment_fingerprint"] != current_environment:
        raise CheckpointError("checkpoint environment fingerprint does not match")
    completed = payload["completed_updates"]
    if type(completed) is not int or not 0 <= completed <= lr_controller.max_updates:
        raise CheckpointError("checkpoint completed_updates is out of range")
    expected_lr = _expected_current_lr(lr_controller, completed)
    if float(payload["current_lr"]) != expected_lr:
        raise CheckpointError("checkpoint current_lr disagrees with completed_updates")
    scheduler_state = payload["scheduler_state"]
    if not isinstance(scheduler_state, Mapping):
        raise CheckpointError("checkpoint scheduler_state must be a mapping")
    if scheduler_state.get("completed_updates") != completed:
        raise CheckpointError("checkpoint scheduler and completed update counts differ")
    sampler_state = payload["sampler_state"]
    if not isinstance(sampler_state, Mapping):
        raise CheckpointError("checkpoint sampler_state must be a mapping")
    if payload["dataset_epoch"] != sampler_state.get("epoch"):
        raise CheckpointError("checkpoint dataset and sampler epochs differ")
    if sampler_state.get("dataset_size") != sampler.dataset_size:
        raise CheckpointError("checkpoint sampler dataset size does not match")
    policy = payload["augmentation_rng_policy"]
    expected_policy = {
        "schema_version": 1,
        "base_seed": config.data["reproducibility"]["seed"],
        "scope": config.data["reproducibility"]["augmentation_rng"]["scope"],
        "seed_components": tuple(
            config.data["reproducibility"]["augmentation_rng"]["seed_components"]
        ),
    }
    if policy != expected_policy:
        raise CheckpointError("checkpoint augmentation RNG policy does not match")
    if not isinstance(payload["dataloader_generator_state"], torch.Tensor):
        raise CheckpointError("checkpoint DataLoader generator state is invalid")


def load_checkpoint(
    source: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    lr_controller: ExactUpdateLrController,
    sampler: StatefulBatchSampler,
    dataset: _EpochDataset,
    dataloader_generator: torch.Generator,
    config: FrozenConfig,
    mean_artifact_sha256: str,
) -> dict[str, Any]:
    """Validate compatibility, restore every state domain, and return metadata."""

    if not isinstance(dataloader_generator, torch.Generator):
        raise CheckpointError("a DataLoader generator is required")
    payload = _read_checkpoint(source)
    _validate_metadata(
        payload,
        model=model,
        lr_controller=lr_controller,
        sampler=sampler,
        config=config,
        mean_artifact_sha256=mean_artifact_sha256,
    )
    try:
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        lr_controller.load_state_dict(payload["scheduler_state"])
        sampler.load_state_dict(payload["sampler_state"])
        dataset.set_epoch(payload["dataset_epoch"])
        _validate_dataset_sampler_epoch(dataset, sampler)
        model.train(bool(payload["model_training"]))
        rng = RNGState(
            schema_version=1,
            python=payload["python_rng_state"],
            numpy=payload["numpy_rng_state"],
            torch_cpu=payload["torch_cpu_rng_state"],
            torch_cuda=tuple(payload["torch_cuda_rng_states"]),
            dataloader_generator=payload["dataloader_generator_state"],
        )
        restore_rng_state(rng, dataloader_generator)
    except Exception as exc:
        raise CheckpointError(f"checkpoint state restoration failed: {exc}") from exc
    if lr_controller.completed_updates != payload["completed_updates"]:
        raise CheckpointError("restored completed update count differs from checkpoint")
    return {
        "completed_updates": lr_controller.completed_updates,
        "current_lr": lr_controller.current_lr,
        "model_name": payload["model_name"],
        "source_commit": payload["source_commit"],
        "source_dirty": payload["source_dirty"],
        "environment_fingerprint": payload["environment_fingerprint"],
        "mean_artifact_sha256": payload["mean_artifact_sha256"],
    }
