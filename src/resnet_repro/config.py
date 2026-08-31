"""Strict loading and protection for the approved frozen experiment config.

This module intentionally has no fallback defaults.  A run either uses the
approved YAML exactly as written or stops before any experiment code starts.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, NoReturn

import yaml


class ConfigError(ValueError):
    """Base class for frozen-config failures."""


class ConfigSchemaError(ConfigError):
    """The YAML payload does not satisfy schema version 1."""


class DuplicateKeyError(ConfigSchemaError):
    """A YAML mapping contains an ambiguous duplicate key."""


class FrozenConfigMutationError(ConfigError):
    """A caller attempted to alter an approved frozen config."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise DuplicateKeyError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


@dataclass(frozen=True)
class _OptionalValue:
    value_type: type


_SCHEMA_V1: dict[str, Any] = {
    "schema_version": int,
    "config_status": str,
    "approved_date": str,
    "source_commit": str,
    "paper": {
        "title": str,
        "arxiv": str,
        "target_table": str,
        "resnet20_test_error_percent": float,
        "plain20_test_error_percent": _OptionalValue(float),
    },
    "approval": {"decision_ids": [str]},
    "experiment": {
        "models": [str],
        "official_result_checkpoint": str,
        "max_updates": int,
        "test_selection_forbidden": bool,
    },
    "hardware": {
        "device": str,
        "gpu_count": int,
        "expected_gpu": str,
        "precision": str,
        "amp": bool,
        "tf32": bool,
        "torch_compile": bool,
    },
    "reproducibility": {
        "seed": int,
        "deterministic_algorithms": bool,
        "cudnn_benchmark": bool,
        "save_rng_states": bool,
        "dataloader_workers": int,
        "worker_seed_policy": str,
        "training_shuffle": bool,
        "training_replacement_sampling": bool,
        "test_shuffle": bool,
        "test_preserve_official_order": bool,
        "sampler_generator": str,
        "save_sampler_generator_state": bool,
        "augmentation_rng": {
            "scope": str,
            "seed_components": [str],
            "worker_assignment_independent": bool,
            "prefetch_order_independent": bool,
            "dataset_epoch_required": bool,
            "training_batch_indices_required": bool,
        },
        "exact_resume": {
            "sampler": str,
            "progress_cursor": str,
            "mark_consumed_after": str,
            "replay_unconsumed_prefetch": bool,
            "checkpoint_fields": [str],
        },
    },
    "dataset": {
        "name": str,
        "train_size": int,
        "test_size": int,
        "num_classes": int,
        "input_shape": [int],
        "official_split": bool,
        "use_validation_for_main_run": bool,
    },
    "preprocessing": {
        "input_scale": [float],
        "standard_deviation_normalization": bool,
        "mean": {
            "type": str,
            "shape": [int],
            "source": str,
            "reuse_for_test": bool,
        },
        "train_order": [str],
        "test_order": [str],
    },
    "architecture": {
        "depth_formula": str,
        "n": int,
        "weighted_layers": int,
        "convolution_layers": int,
        "fully_connected_layers": int,
        "stem": {
            "kernel_size": int,
            "stride": int,
            "padding": int,
            "out_channels": int,
            "bias": bool,
        },
        "stage_blocks": [int],
        "stage_channels": [int],
        "stage_spatial_sizes": [int],
        "block": {"type": str, "convolutions_per_block": int, "order": [str]},
        "shortcut": {
            "type": str,
            "trainable_parameters": bool,
            "spatial_downsample": str,
            "channel_padding": str,
            "odd_size_policy": str,
        },
        "global_average_pooling": bool,
        "classifier": {"in_features": int, "out_features": int, "bias": bool},
    },
    "initialization": {
        "convolution": {"distribution": str, "mean": float, "fan_mode": str, "gain": str},
        "fully_connected": {
            "distribution": str,
            "mean": float,
            "std": float,
            "evidence_status": str,
        },
        "bias": {"fully_connected": float},
        "batch_norm": {"gamma": float, "beta": float},
    },
    "batch_normalization": {
        "enabled": bool,
        "implementation": str,
        "affine": bool,
        "epsilon": float,
        "pytorch_momentum": float,
        "running_mean_state": str,
        "running_variance_state": str,
        "running_scale_factor": bool,
        "evaluation_statistics": str,
        "checkpoint_buffers": [str],
        "synchronized": bool,
        "evaluation_uses_running_statistics": bool,
        "post_training_recalibration": bool,
        "mandatory_compatibility_test": bool,
        "compatibility_status": str,
    },
    "optimizer": {
        "type": str,
        "batch_size": int,
        "momentum": float,
        "weight_decay": float,
        "weight_decay_scope": str,
        "nesterov": bool,
    },
    "learning_rate": {
        "initial": float,
        "schedule_unit": str,
        "ranges": [{"first_update": int, "last_update": int, "value": float}],
    },
    "loss": {"type": str, "reduction": str, "explicit_softmax_in_model": bool},
    "evaluation": {
        "test_view": str,
        "metric": str,
        "primary_checkpoint": str,
        "best_test_checkpoint_selection": bool,
    },
    "logging": {
        "train_metrics_interval_updates": int,
        "evaluation_interval_updates": int,
        "checkpoint_updates": [int],
        "log_learning_rate_every_update": bool,
    },
    "preflight": {"required": [str], "formal_training_allowed_only_after_all_pass": bool},
    "evidence_notes": {
        "option_a_tensor_semantics": str,
        "fc_initialization": str,
        "bn_momentum": str,
        "weight_decay_scope": str,
        "preprocessing_order": str,
        "dataloader_ordering": str,
        "final_checkpoint_reporting_form": str,
    },
}


_APPROVED_DECISIONS = {
    "DEC-INIT-001", "DEC-INIT-002", "DEC-INIT-003", "DEC-FC-001",
    "DEC-FCBIAS-001", "DEC-SHORT-001", "DEC-CONV-001", "DEC-BN-001A",
    "DEC-BN-001B", "DEC-BN-001C", "DEC-WD-001", "DEC-PRE-001", "DEC-AUG-001",
    "DEC-MEANORDER-001", "DEC-LR-001", "DEC-CAFFE-001", "DEC-SHUFFLE-001",
    "DEC-RNG-002",
}


def _schema_failure(path: str, message: str) -> NoReturn:
    raise ConfigSchemaError(f"{path}: {message}")


def _validate_structure(value: Any, schema: Any, path: str = "config") -> None:
    if isinstance(schema, _OptionalValue):
        if value is not None:
            _validate_structure(value, schema.value_type, path)
        return
    if isinstance(schema, type):
        if type(value) is not schema:
            _schema_failure(path, f"expected {schema.__name__}, got {type(value).__name__}")
        return
    if isinstance(schema, dict):
        if not isinstance(value, dict):
            _schema_failure(path, f"expected mapping, got {type(value).__name__}")
        missing = sorted(set(schema) - set(value))
        unknown = sorted(set(value) - set(schema))
        if missing:
            _schema_failure(path, f"missing required fields: {', '.join(missing)}")
        if unknown:
            _schema_failure(path, f"unknown fields: {', '.join(unknown)}")
        for key, child_schema in schema.items():
            _validate_structure(value[key], child_schema, f"{path}.{key}")
        return
    if isinstance(schema, list) and len(schema) == 1:
        if not isinstance(value, list):
            _schema_failure(path, f"expected list, got {type(value).__name__}")
        for index, item in enumerate(value):
            _validate_structure(item, schema[0], f"{path}[{index}]")
        return
    raise TypeError(f"invalid internal schema at {path}")


def _expect(data: Mapping[str, Any], dotted_path: str, expected: Any) -> None:
    value: Any = data
    for part in dotted_path.split("."):
        value = value[part]
    if value != expected:
        _schema_failure(dotted_path, f"expected {expected!r}, got {value!r}")


def _validate_invariants(data: dict[str, Any]) -> None:
    fixed_values = {
        "schema_version": 1,
        "config_status": "frozen",
        "paper.plain20_test_error_percent": None,
        "experiment.models": ["plain20", "resnet20"],
        "experiment.official_result_checkpoint": "final",
        "experiment.max_updates": 64000,
        "experiment.test_selection_forbidden": True,
        "reproducibility.seed": 1,
        "reproducibility.deterministic_algorithms": True,
        "reproducibility.cudnn_benchmark": False,
        "reproducibility.save_rng_states": True,
        "reproducibility.worker_seed_policy": "derived_from_base_seed",
        "reproducibility.training_shuffle": True,
        "reproducibility.training_replacement_sampling": False,
        "reproducibility.test_shuffle": False,
        "reproducibility.test_preserve_official_order": True,
        "reproducibility.sampler_generator": "derived_from_base_seed",
        "reproducibility.save_sampler_generator_state": True,
        "reproducibility.augmentation_rng.scope": "per_sample",
        "reproducibility.augmentation_rng.seed_components": [
            "base_seed", "epoch", "official_sample_index"
        ],
        "reproducibility.augmentation_rng.worker_assignment_independent": True,
        "reproducibility.augmentation_rng.prefetch_order_independent": True,
        "reproducibility.augmentation_rng.dataset_epoch_required": True,
        "reproducibility.augmentation_rng.training_batch_indices_required": True,
        "reproducibility.exact_resume.sampler": "stateful_batch_sampler",
        "reproducibility.exact_resume.progress_cursor": "consumed_samples",
        "reproducibility.exact_resume.mark_consumed_after": "successful_optimizer_update",
        "reproducibility.exact_resume.replay_unconsumed_prefetch": True,
        "reproducibility.exact_resume.checkpoint_fields": [
            "permutation", "consumed_cursor", "epoch", "generator_state"
        ],
        "dataset.input_shape": [3, 32, 32],
        "preprocessing.input_scale": [0.0, 255.0],
        "preprocessing.standard_deviation_normalization": False,
        "preprocessing.mean.shape": [3, 32, 32],
        "preprocessing.mean.source": "full_official_training_set",
        "preprocessing.mean.reuse_for_test": True,
        "architecture.depth_formula": "6n+2",
        "architecture.n": 3,
        "architecture.weighted_layers": 20,
        "architecture.convolution_layers": 19,
        "architecture.fully_connected_layers": 1,
        "architecture.stage_blocks": [3, 3, 3],
        "architecture.stage_channels": [16, 32, 64],
        "architecture.stage_spatial_sizes": [32, 16, 8],
        "architecture.shortcut.type": "option_a",
        "architecture.shortcut.trainable_parameters": False,
        "batch_normalization.enabled": True,
        "batch_normalization.implementation": "caffe_compatible_scaled_accumulator",
        "batch_normalization.affine": True,
        "batch_normalization.epsilon": 1.0e-5,
        "batch_normalization.pytorch_momentum": 0.001,
        "batch_normalization.running_mean_state": "scaled_accumulator",
        "batch_normalization.running_variance_state": "scaled_unbiased_accumulator",
        "batch_normalization.running_scale_factor": True,
        "batch_normalization.evaluation_statistics": "accumulator_divided_by_scale_factor",
        "batch_normalization.checkpoint_buffers": [
            "running_mean", "running_var", "running_scale", "num_batches_tracked"
        ],
        "batch_normalization.synchronized": False,
        "batch_normalization.evaluation_uses_running_statistics": True,
        "batch_normalization.post_training_recalibration": False,
        "batch_normalization.mandatory_compatibility_test": True,
        "batch_normalization.compatibility_status": "approved_implementation_mandatory_preflight",
        "optimizer.type": "SGD",
        "optimizer.batch_size": 128,
        "optimizer.momentum": 0.9,
        "optimizer.weight_decay": 0.0001,
        "optimizer.weight_decay_scope": "all_learnable_parameters",
        "optimizer.nesterov": False,
        "evaluation.primary_checkpoint": "update_64000_final",
        "evaluation.best_test_checkpoint_selection": False,
        "preflight.formal_training_allowed_only_after_all_pass": True,
    }
    for dotted_path, expected in fixed_values.items():
        _expect(data, dotted_path, expected)

    decisions = data["approval"]["decision_ids"]
    if len(decisions) != len(set(decisions)):
        _schema_failure("approval.decision_ids", "decision IDs must be unique")
    if set(decisions) != _APPROVED_DECISIONS:
        _schema_failure("approval.decision_ids", "must exactly match approved Phase 0 decisions")

    if data["architecture"]["weighted_layers"] != 6 * data["architecture"]["n"] + 2:
        _schema_failure("architecture.weighted_layers", "must equal 6*n+2")
    counted = data["architecture"]["convolution_layers"] + data["architecture"]["fully_connected_layers"]
    if counted != data["architecture"]["weighted_layers"]:
        _schema_failure("architecture", "conv + FC count must equal weighted layer count")
    if data["architecture"]["classifier"]["out_features"] != data["dataset"]["num_classes"]:
        _schema_failure("architecture.classifier.out_features", "must equal dataset.num_classes")

    ranges = data["learning_rate"]["ranges"]
    expected_ranges = [(1, 32000, 0.1), (32001, 48000, 0.01), (48001, 64000, 0.001)]
    actual_ranges = [(item["first_update"], item["last_update"], item["value"]) for item in ranges]
    if actual_ranges != expected_ranges:
        _schema_failure("learning_rate.ranges", f"expected exact update ranges {expected_ranges!r}")
    if ranges[-1]["last_update"] != data["experiment"]["max_updates"]:
        _schema_failure("learning_rate.ranges", "final boundary must equal experiment.max_updates")
    if data["learning_rate"]["initial"] != ranges[0]["value"]:
        _schema_failure("learning_rate.initial", "must equal first range value")

    if data["logging"]["checkpoint_updates"] != [32000, 48000, 64000]:
        _schema_failure("logging.checkpoint_updates", "must be [32000, 48000, 64000]")
    required_preflight = data["preflight"]["required"]
    if len(required_preflight) != len(set(required_preflight)):
        _schema_failure("preflight.required", "entries must be unique")

    evidence = data["evidence_notes"]
    expected_evidence = {
        "option_a_tensor_semantics": "assumption",
        "fc_initialization": "low_confidence_assumption",
        "bn_momentum": "framework_compatibility_assumption",
        "weight_decay_scope": "caffe_default_derived_assumption",
        "preprocessing_order": "assumption",
        "dataloader_ordering": "assumption",
        "final_checkpoint_reporting_form": "project_rule",
    }
    if evidence != expected_evidence:
        _schema_failure("evidence_notes", "approved evidence statuses must be preserved exactly")


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class FrozenConfig:
    """An immutable, validated snapshot of the exact approved YAML bytes."""

    path: Path
    raw_bytes: bytes
    sha256: str
    data: Mapping[str, Any]

    def assert_source_unchanged(self) -> None:
        current = self.path.read_bytes()
        if current != self.raw_bytes:
            raise FrozenConfigMutationError(
                f"frozen config changed after load: {self.path} "
                f"({self.sha256} -> {sha256(current).hexdigest().upper()})"
            )


def load_frozen_config(path: str | Path) -> FrozenConfig:
    """Load schema v1 without defaults, duplicate keys, or mutable config data."""

    source = Path(path).resolve(strict=True)
    raw = source.read_bytes()
    try:
        payload = yaml.load(raw.decode("utf-8"), Loader=_UniqueKeyLoader)
    except UnicodeDecodeError as exc:
        raise ConfigSchemaError("config must be UTF-8") from exc
    except yaml.YAMLError as exc:
        raise ConfigSchemaError(f"invalid YAML: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigSchemaError("config root must be a mapping")
    _validate_structure(payload, _SCHEMA_V1)
    _validate_invariants(payload)
    return FrozenConfig(
        path=source,
        raw_bytes=raw,
        sha256=sha256(raw).hexdigest().upper(),
        data=_deep_freeze(payload),
    )


def resolve_frozen_config(
    path: str | Path, *, overrides: Mapping[str, Any] | None = None
) -> FrozenConfig:
    """Resolve a run config while refusing every CLI/runtime override.

    A changed configuration requires a separately reviewed YAML and therefore
    cannot be created by this runtime API.
    """

    if overrides:
        names = ", ".join(sorted(overrides))
        raise FrozenConfigMutationError(
            f"frozen config overrides are forbidden ({names}); create a new config, "
            "SHA-256, decision entry, and approval instead"
        )
    return load_frozen_config(path)
