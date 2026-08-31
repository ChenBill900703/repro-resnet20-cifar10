"""Phase 4 run orchestration for the frozen CIFAR-10 reproduction.

The engine intentionally exposes no hyper-parameter overrides.  The only
runtime choices are the approved model name, output directory, smoke/formal
mode, and an optional compatible checkpoint to resume.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import torch
from torch import nn

from ..config import FrozenConfig, load_frozen_config
from ..data import (
    Cifar10Metadata,
    EpochSynchronizedDataLoader,
    IndexedDataset,
    TestingTransform,
    TrainingTransform,
    load_mean_artifact,
    make_test_loader,
)
from ..evaluation import EvaluationResult, evaluate
from ..models import Plain20, ResNet20
from ..reproducibility import (
    configure_from_frozen_config,
    environment_fingerprint,
    make_worker_generator,
)
from ..sampling import StatefulBatchSampler
from .checkpoint import load_checkpoint, save_checkpoint
from .optimizer import build_sgd_optimizer
from .schedule import ExactUpdateLrController
from .step import TrainingStepResult, train_one_update


# Set RESNET_REPRO_EXPECTED_COMMIT when an experiment must be pinned to one
# exact revision. Public clones remain runnable when the variable is unset;
# formal runs are still required to start from a clean working tree and a
# matching PASS preflight report.
EXPECTED_SOURCE_COMMIT = os.environ.get("RESNET_REPRO_EXPECTED_COMMIT")


@dataclass(frozen=True)
class RunPaths:
    root: Path
    manifest: Path
    events: Path
    checkpoints: Path
    evaluations: Path
    frozen_config_copy: Path


@dataclass
class RunObjects:
    config: FrozenConfig
    mean_sha256: str
    model: nn.Module
    optimizer: torch.optim.Optimizer
    controller: ExactUpdateLrController
    sampler: StatefulBatchSampler
    train_dataset: IndexedDataset
    train_loader: EpochSynchronizedDataLoader
    test_dataset: IndexedDataset
    test_loader: Iterable[Mapping[str, torch.Tensor]]
    worker_generator: torch.Generator
    device: torch.device


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_tensor(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    return sha256(contiguous.numpy().tobytes()).hexdigest().upper()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    # Windows readers (including monitoring tools and virus scanners) may hold
    # a short-lived sharing lock on the destination.  Preserve atomic replace
    # semantics while tolerating that transient condition; persistent failures
    # still stop the run.
    for attempt in range(50):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 49:
                raise
            time.sleep(0.02)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, sort_keys=True, default=_json_default) + "\n")
        stream.flush()


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def source_provenance(repo: Path) -> dict[str, Any]:
    commit = _git(repo, "rev-parse", "HEAD")
    return {
        "commit": commit,
        "dirty": bool(_git(repo, "status", "--porcelain")),
        "status_porcelain": _git(repo, "status", "--porcelain"),
    }


def make_run_paths(root: str | Path) -> RunPaths:
    run_root = Path(root).resolve()
    return RunPaths(
        root=run_root,
        manifest=run_root / "run_manifest.json",
        events=run_root / "events.jsonl",
        checkpoints=run_root / "checkpoints",
        evaluations=run_root / "evaluations",
        frozen_config_copy=run_root / "frozen_config.yaml",
    )


def _load_official_cifar10(data_root: Path, *, train: bool):
    from torchvision.datasets import CIFAR10

    return CIFAR10(root=str(data_root), train=train, download=False)


def build_run_objects(
    *,
    config_path: str | Path,
    mean_path: str | Path,
    data_root: str | Path,
    model_name: str,
) -> RunObjects:
    config = load_frozen_config(config_path)
    config.assert_source_unchanged()
    if model_name not in tuple(config.data["experiment"]["models"]):
        raise ValueError(f"model {model_name!r} is not approved by the frozen config")
    if config.data["hardware"]["device"] != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the frozen run requires an available CUDA device")
    configure_from_frozen_config(config.data)
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    if gpu_name != config.data["hardware"]["expected_gpu"]:
        raise RuntimeError(
            f"frozen GPU mismatch: expected {config.data['hardware']['expected_gpu']!r}, "
            f"got {gpu_name!r}"
        )
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("TF32 must be disabled before model construction")

    mean_artifact = load_mean_artifact(mean_path, config.data)
    metadata = Cifar10Metadata.from_config(config.data)
    data_root = Path(data_root).resolve()
    train_base = _load_official_cifar10(data_root, train=True)
    test_base = _load_official_cifar10(data_root, train=False)
    train_dataset = IndexedDataset(
        train_base,
        metadata=metadata,
        split="train",
        transform=TrainingTransform(
            config.data,
            mean_artifact.mean,
            base_seed=config.data["reproducibility"]["seed"],
        ),
    )
    test_dataset = IndexedDataset(
        test_base,
        metadata=metadata,
        split="test",
        transform=TestingTransform(config.data, mean_artifact.mean),
    )
    batch_size = config.data["optimizer"]["batch_size"]
    sampler = StatefulBatchSampler(
        len(train_dataset),
        batch_size,
        base_seed=config.data["reproducibility"]["seed"],
        drop_last=True,
    )
    worker_generator = make_worker_generator(config.data["reproducibility"]["seed"])
    train_loader = EpochSynchronizedDataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=config.data["reproducibility"]["dataloader_workers"],
        worker_generator=worker_generator,
    )
    test_worker_generator = make_worker_generator(
        config.data["reproducibility"]["seed"]
    )
    test_loader = make_test_loader(
        test_dataset,
        batch_size=batch_size,
        num_workers=config.data["reproducibility"]["dataloader_workers"],
        worker_generator=test_worker_generator,
    )
    model_class = Plain20 if model_name == "plain20" else ResNet20
    model = model_class(config).to(device)
    optimizer = build_sgd_optimizer(model, config)
    controller = ExactUpdateLrController(optimizer, config)
    return RunObjects(
        config=config,
        mean_sha256=mean_artifact.sha256,
        model=model,
        optimizer=optimizer,
        controller=controller,
        sampler=sampler,
        train_dataset=train_dataset,
        train_loader=train_loader,
        test_dataset=test_dataset,
        test_loader=test_loader,
        worker_generator=worker_generator,
        device=device,
    )


def _device_batches(
    batches: Iterable[Mapping[str, torch.Tensor]], device: torch.device
) -> Iterator[dict[str, torch.Tensor]]:
    for batch in batches:
        yield {
            "image": batch["image"].to(device),
            "target": batch["target"].to(device),
            "index": batch["index"],
        }


def run_evaluation(objects: RunObjects) -> EvaluationResult:
    return evaluate(
        objects.model,
        _device_batches(objects.test_loader, objects.device),
        objects.config,
    )


def save_final_predictions(objects: RunObjects, destination: Path) -> dict[str, Any]:
    was_training = objects.model.training
    indices: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    objects.model.eval()
    try:
        with torch.no_grad():
            for batch in objects.test_loader:
                output = objects.model(batch["image"].to(objects.device))
                indices.append(batch["index"].cpu())
                targets.append(batch["target"].cpu())
                logits.append(output.cpu())
    finally:
        objects.model.train(was_training)
    payload = {
        "indices": torch.cat(indices),
        "targets": torch.cat(targets),
        "logits": torch.cat(logits),
    }
    payload["predicted_classes"] = payload["logits"].argmax(dim=1)
    expected = torch.arange(len(objects.test_dataset), dtype=torch.long)
    if not torch.equal(payload["indices"], expected):
        raise RuntimeError("final prediction order differs from official CIFAR-10 test order")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    return {
        "path": str(destination),
        "sample_count": int(payload["indices"].numel()),
        "sha256": sha256(destination.read_bytes()).hexdigest().upper(),
    }


def _validate_formal_preflight(
    report_path: str | Path,
    *,
    config: FrozenConfig,
    mean_sha256: str,
    provenance: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if report.get("overall_pass") is not True:
        raise RuntimeError("formal preflight report is not PASS")
    expected = {
        "config_sha256": config.sha256,
        "mean_artifact_sha256": mean_sha256,
        "source_commit": provenance["commit"],
        "source_dirty": provenance["dirty"],
        "environment_fingerprint_sha256": environment["fingerprint_sha256"],
    }
    for field, value in expected.items():
        if report.get(field) != value:
            raise RuntimeError(f"formal preflight {field} does not match current run")
    return report


def run_training(
    *,
    config_path: str | Path,
    mean_path: str | Path,
    data_root: str | Path,
    model_name: str,
    run_dir: str | Path,
    mode: str,
    target_updates: int | None = None,
    resume_checkpoint: str | Path | None = None,
    preflight_report: str | Path | None = None,
    trace_batches: bool = False,
) -> dict[str, Any]:
    if mode not in {"smoke", "formal"}:
        raise ValueError("mode must be 'smoke' or 'formal'")
    paths = make_run_paths(run_dir)
    repo = Path(config_path).resolve().parents[1]
    provenance = source_provenance(repo)
    if (
        EXPECTED_SOURCE_COMMIT is not None
        and provenance["commit"] != EXPECTED_SOURCE_COMMIT
    ):
        raise RuntimeError(
            f"source commit must be {EXPECTED_SOURCE_COMMIT}, got {provenance['commit']}"
        )
    objects = build_run_objects(
        config_path=config_path,
        mean_path=mean_path,
        data_root=data_root,
        model_name=model_name,
    )
    requirements_lock = repo / "environment" / "requirements-lock.txt"
    environment = environment_fingerprint(requirements_lock)
    max_updates = objects.config.data["experiment"]["max_updates"]
    if mode == "formal":
        if provenance["dirty"]:
            raise RuntimeError("formal runs require a clean Git working tree")
        if target_updates not in (None, max_updates):
            raise ValueError("formal runs always target exactly 64,000 updates")
        target_updates = max_updates
        if preflight_report is None:
            raise RuntimeError("formal runs require a matching PASS preflight report")
        preflight = _validate_formal_preflight(
            preflight_report,
            config=objects.config,
            mean_sha256=objects.mean_sha256,
            provenance=provenance,
            environment=environment,
        )
    else:
        if type(target_updates) is not int or not 1 <= target_updates <= 1000:
            raise ValueError("smoke target_updates must be an integer in [1,1000]")
        preflight = None

    paths.root.mkdir(parents=True, exist_ok=True)
    paths.checkpoints.mkdir(parents=True, exist_ok=True)
    paths.evaluations.mkdir(parents=True, exist_ok=True)
    if not paths.frozen_config_copy.exists():
        paths.frozen_config_copy.write_bytes(objects.config.raw_bytes)
    elif paths.frozen_config_copy.read_bytes() != objects.config.raw_bytes:
        raise RuntimeError("run directory frozen config copy differs from current config")

    resumed_from: str | None = None
    rewind_worker_generator_after_first_iterator = False
    if resume_checkpoint is not None:
        metadata = load_checkpoint(
            resume_checkpoint,
            model=objects.model,
            optimizer=objects.optimizer,
            lr_controller=objects.controller,
            sampler=objects.sampler,
            dataset=objects.train_dataset,
            dataloader_generator=objects.worker_generator,
            config=objects.config,
            mean_artifact_sha256=objects.mean_sha256,
        )
        resumed_from = str(Path(resume_checkpoint).resolve())
        rewind_worker_generator_after_first_iterator = True
        if metadata["completed_updates"] >= target_updates:
            if metadata["completed_updates"] > target_updates:
                raise RuntimeError("resume checkpoint is beyond the requested target")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "mode": mode,
        "model_name": model_name,
        "started_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "target_updates": target_updates,
        "completed_updates": objects.controller.completed_updates,
        "resumed_from": resumed_from,
        "source_commit": provenance["commit"],
        "source_dirty": provenance["dirty"],
        "source_status_porcelain": provenance["status_porcelain"],
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": objects.config.sha256,
        "config_copy": str(paths.frozen_config_copy),
        "mean_artifact_path": str(Path(mean_path).resolve()),
        "mean_artifact_sha256": objects.mean_sha256,
        "data_root": str(Path(data_root).resolve()),
        "official_train_size": len(objects.train_dataset),
        "official_test_size": len(objects.test_dataset),
        "batch_size": objects.config.data["optimizer"]["batch_size"],
        "drop_last": True,
        "drop_last_rationale": "DERIVED: enforce the frozen global batch size of 128 for every optimizer update.",
        "seed": objects.config.data["reproducibility"]["seed"],
        "environment": environment,
        "gpu_name": torch.cuda.get_device_name(objects.device),
        "precision": "fp32",
        "amp": False,
        "tf32": False,
        "torch_compile": False,
        "preflight_report": str(Path(preflight_report).resolve()) if preflight_report else None,
        "preflight_summary": preflight,
        "events_path": str(paths.events),
        "checkpoints_dir": str(paths.checkpoints),
        "evaluations_dir": str(paths.evaluations),
        "test_selection_forbidden": True,
        "official_result_checkpoint": "update_64000_final",
    }
    _atomic_write_json(paths.manifest, manifest)
    _append_jsonl(
        paths.events,
        {
            "event": "run_start" if resume_checkpoint is None else "run_resume",
            "timestamp_utc": utc_now(),
            "completed_updates": objects.controller.completed_updates,
            "target_updates": target_updates,
            "resume_checkpoint": resumed_from,
        },
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(objects.device)
    total_samples = 0
    measured_seconds = 0.0
    window_loss = 0.0
    window_correct = 0
    window_samples = 0
    window_start_update = objects.controller.completed_updates + 1
    run_started = time.perf_counter()
    iterator: Iterator[Mapping[str, torch.Tensor]] | None = None
    latest_checkpoint: Path | None = None
    try:
        while objects.controller.completed_updates < target_updates:
            if iterator is None:
                saved_worker_state = (
                    objects.worker_generator.get_state().clone()
                    if rewind_worker_generator_after_first_iterator
                    else None
                )
                iterator = iter(objects.train_loader)
                if saved_worker_state is not None:
                    objects.worker_generator.set_state(saved_worker_state)
                    rewind_worker_generator_after_first_iterator = False
            update_started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = None
                gc.collect()
                continue
            images_cpu = batch["image"]
            indices = tuple(int(value) for value in batch["index"].tolist())
            if len(indices) != objects.config.data["optimizer"]["batch_size"]:
                raise RuntimeError("every training update must use the frozen batch size")
            images_hash = _sha256_tensor(images_cpu) if trace_batches else None
            images = images_cpu.to(objects.device)
            targets = batch["target"].to(objects.device)
            result: TrainingStepResult = train_one_update(
                objects.model,
                objects.optimizer,
                objects.controller,
                images=images,
                targets=targets,
                indices=indices,
                batch_acknowledger=objects.train_loader,
            )
            torch.cuda.synchronize(objects.device)
            update_seconds = time.perf_counter() - update_started
            measured_seconds += update_seconds
            total_samples += result.sample_count
            window_loss += result.loss * result.sample_count
            window_correct += result.correct_count
            window_samples += result.sample_count
            update_event: dict[str, Any] = {
                "event": "train_update",
                "timestamp_utc": utc_now(),
                "update": result.update_number,
                "learning_rate": result.learning_rate,
                "loss": result.loss,
                "correct_count": result.correct_count,
                "sample_count": result.sample_count,
                "train_error_percent": 100.0 * (result.sample_count - result.correct_count) / result.sample_count,
                "batch_indices_sha256": sha256(
                    b"".join(int(index).to_bytes(4, "little", signed=False) for index in indices)
                ).hexdigest().upper(),
                "update_seconds": update_seconds,
                "samples_per_second": result.sample_count / update_seconds,
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(objects.device),
                "sampler_epoch": objects.sampler.epoch,
                "sampler_consumed_cursor": objects.sampler.consumed_position,
                "dataset_epoch": objects.train_dataset.epoch,
            }
            if trace_batches:
                update_event["indices"] = list(indices)
                update_event["augmented_images_sha256"] = images_hash
            _append_jsonl(paths.events, update_event)

            metrics_interval = objects.config.data["logging"]["train_metrics_interval_updates"]
            if result.update_number % metrics_interval == 0:
                _append_jsonl(
                    paths.events,
                    {
                        "event": "train_metrics",
                        "timestamp_utc": utc_now(),
                        "first_update": window_start_update,
                        "last_update": result.update_number,
                        "mean_loss": window_loss / window_samples,
                        "train_error_percent": 100.0 * (window_samples - window_correct) / window_samples,
                        "sample_count": window_samples,
                    },
                )
                window_loss = 0.0
                window_correct = 0
                window_samples = 0
                window_start_update = result.update_number + 1

            should_evaluate = mode == "formal" and (
                result.update_number % objects.config.data["logging"]["evaluation_interval_updates"] == 0
                or result.update_number == target_updates
            )
            if should_evaluate:
                evaluation = run_evaluation(objects)
                _append_jsonl(
                    paths.events,
                    {
                        "event": "evaluation",
                        "timestamp_utc": utc_now(),
                        "update": result.update_number,
                        **asdict(evaluation),
                        "used_for_selection": False,
                    },
                )
                _atomic_write_json(
                    paths.evaluations / f"evaluation_update_{result.update_number:06d}.json",
                    {
                        "schema_version": 1,
                        "model_name": model_name,
                        "update": result.update_number,
                        **asdict(evaluation),
                        "used_for_selection": False,
                    },
                )

            formal_checkpoints = set(objects.config.data["logging"]["checkpoint_updates"])
            should_checkpoint = (
                mode == "formal" and result.update_number in formal_checkpoints
            ) or (mode == "smoke" and result.update_number == target_updates)
            if should_checkpoint:
                suffix = "_final" if mode == "formal" and result.update_number == max_updates else ""
                latest_checkpoint = paths.checkpoints / (
                    f"checkpoint_update_{result.update_number:06d}{suffix}.pt"
                )
                save_checkpoint(
                    latest_checkpoint,
                    model=objects.model,
                    optimizer=objects.optimizer,
                    lr_controller=objects.controller,
                    sampler=objects.sampler,
                    dataset=objects.train_dataset,
                    dataloader_generator=objects.worker_generator,
                    config=objects.config,
                    mean_artifact_sha256=objects.mean_sha256,
                )
                _append_jsonl(
                    paths.events,
                    {
                        "event": "checkpoint_saved",
                        "timestamp_utc": utc_now(),
                        "update": result.update_number,
                        "path": str(latest_checkpoint),
                        "sha256": sha256(latest_checkpoint.read_bytes()).hexdigest().upper(),
                    },
                )

            manifest["completed_updates"] = result.update_number
            manifest["updated_at_utc"] = utc_now()
            manifest["latest_checkpoint"] = str(latest_checkpoint) if latest_checkpoint else None
            manifest["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(objects.device)
            manifest["measured_update_seconds"] = measured_seconds
            manifest["measured_samples"] = total_samples
            manifest["samples_per_second"] = total_samples / measured_seconds
            _atomic_write_json(paths.manifest, manifest)

        if mode == "formal" and objects.controller.completed_updates == max_updates:
            final_evaluation_path = paths.evaluations / f"evaluation_update_{max_updates:06d}.json"
            if not final_evaluation_path.exists():
                evaluation = run_evaluation(objects)
                _atomic_write_json(
                    final_evaluation_path,
                    {
                        "schema_version": 1,
                        "model_name": model_name,
                        "update": max_updates,
                        **asdict(evaluation),
                        "used_for_selection": False,
                    },
                )
            predictions = save_final_predictions(
                objects, paths.evaluations / "final_predictions.pt"
            )
            manifest["final_evaluation"] = json.loads(
                final_evaluation_path.read_text(encoding="utf-8")
            )
            manifest["final_predictions"] = predictions

        manifest["status"] = "completed"
        manifest["completed_at_utc"] = utc_now()
        manifest["updated_at_utc"] = utc_now()
        manifest["wall_seconds"] = time.perf_counter() - run_started
        manifest["completed_updates"] = objects.controller.completed_updates
        manifest["latest_checkpoint"] = str(latest_checkpoint) if latest_checkpoint else None
        manifest["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(objects.device)
        manifest["measured_update_seconds"] = measured_seconds
        manifest["measured_samples"] = total_samples
        manifest["samples_per_second"] = (
            total_samples / measured_seconds if measured_seconds else None
        )
        _atomic_write_json(paths.manifest, manifest)
        _append_jsonl(
            paths.events,
            {
                "event": "run_complete",
                "timestamp_utc": utc_now(),
                "completed_updates": objects.controller.completed_updates,
                "wall_seconds": manifest["wall_seconds"],
            },
        )
        return manifest
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["updated_at_utc"] = utc_now()
        manifest["failure_type"] = type(exc).__name__
        manifest["failure_message"] = str(exc)
        manifest["completed_updates"] = objects.controller.completed_updates
        _atomic_write_json(paths.manifest, manifest)
        _append_jsonl(
            paths.events,
            {
                "event": "run_failed",
                "timestamp_utc": utc_now(),
                "completed_updates": objects.controller.completed_updates,
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
            },
        )
        raise
