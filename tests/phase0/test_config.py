from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from resnet_repro.config import (
    ConfigSchemaError,
    DuplicateKeyError,
    FrozenConfigMutationError,
    load_frozen_config,
    resolve_frozen_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "cifar10_plain20_resnet20_frozen.yaml"


def _payload() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _write_yaml(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "candidate.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_config_001_loads_exact_frozen_schema() -> None:
    config = load_frozen_config(CONFIG)

    assert config.sha256 == "B6E9AA16D049FF5F5C089FB2FBAEDFC5608A220AC56E713FCF8DE2850653F84E"
    assert config.data["schema_version"] == 1
    assert config.data["paper"]["plain20_test_error_percent"] is None
    assert config.data["evidence_notes"]["bn_momentum"] == "framework_compatibility_assumption"
    assert len(config.data["approval"]["decision_ids"]) == 18


def test_phase0_closeout_fields_and_decisions_are_required() -> None:
    config = load_frozen_config(CONFIG).data

    assert config["batch_normalization"]["implementation"] == "caffe_compatible_scaled_accumulator"
    assert config["batch_normalization"]["checkpoint_buffers"] == (
        "running_mean", "running_var", "running_scale", "num_batches_tracked"
    )
    assert config["reproducibility"]["augmentation_rng"]["seed_components"] == (
        "base_seed", "epoch", "official_sample_index"
    )
    assert config["reproducibility"]["exact_resume"]["progress_cursor"] == "consumed_samples"
    assert {"DEC-BN-001C", "DEC-RNG-002"}.issubset(
        config["approval"]["decision_ids"]
    )


def test_old_pre_closeout_config_is_rejected(tmp_path: Path) -> None:
    payload = _payload()
    payload["approval"]["decision_ids"].remove("DEC-BN-001C")
    payload["approval"]["decision_ids"].remove("DEC-RNG-002")
    del payload["reproducibility"]["augmentation_rng"]
    del payload["reproducibility"]["exact_resume"]
    for field in (
        "implementation",
        "running_mean_state",
        "running_variance_state",
        "running_scale_factor",
        "evaluation_statistics",
        "checkpoint_buffers",
    ):
        del payload["batch_normalization"][field]
    payload["batch_normalization"]["compatibility_status"] = (
        "approved_value_pending_mandatory_preflight"
    )

    with pytest.raises(ConfigSchemaError):
        load_frozen_config(_write_yaml(tmp_path, payload))


@pytest.mark.parametrize("decision_id", ["DEC-BN-001C", "DEC-RNG-002"])
def test_new_approved_decisions_cannot_be_removed(
    tmp_path: Path, decision_id: str
) -> None:
    payload = _payload()
    payload["approval"]["decision_ids"].remove(decision_id)
    with pytest.raises(ConfigSchemaError, match="decision"):
        load_frozen_config(_write_yaml(tmp_path, payload))


def test_unknown_bn_implementation_is_rejected(tmp_path: Path) -> None:
    payload = _payload()
    payload["batch_normalization"]["implementation"] = "standard_pytorch"
    with pytest.raises(ConfigSchemaError, match="implementation"):
        load_frozen_config(_write_yaml(tmp_path, payload))


def test_config_data_is_deeply_immutable() -> None:
    config = load_frozen_config(CONFIG)

    with pytest.raises(TypeError):
        config.data["reproducibility"]["seed"] = 2
    assert isinstance(config.data["architecture"]["stage_blocks"], tuple)


@pytest.mark.parametrize("change", ["missing", "unknown", "wrong_type"])
def test_schema_rejects_missing_unknown_and_wrong_type(tmp_path: Path, change: str) -> None:
    payload = _payload()
    if change == "missing":
        del payload["optimizer"]["momentum"]
    elif change == "unknown":
        payload["optimizer"]["hidden_default"] = 123
    else:
        payload["optimizer"]["batch_size"] = "128"

    with pytest.raises(ConfigSchemaError):
        load_frozen_config(_write_yaml(tmp_path, payload))


def test_schema_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("schema_version: 1\nschema_version: 1\n", encoding="utf-8")

    with pytest.raises(DuplicateKeyError, match="duplicate YAML key"):
        load_frozen_config(path)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("reproducibility", "seed", 2),
        ("batch_normalization", "pytorch_momentum", 0.1),
        ("experiment", "max_updates", 63999),
        ("evaluation", "best_test_checkpoint_selection", True),
    ],
)
def test_schema_rejects_cross_field_decision_drift(
    tmp_path: Path, section: str, field: str, value: object
) -> None:
    payload = _payload()
    payload[section][field] = value

    with pytest.raises(ConfigSchemaError):
        load_frozen_config(_write_yaml(tmp_path, payload))


def test_config_002_rejects_every_nonempty_override() -> None:
    with pytest.raises(FrozenConfigMutationError, match="new config"):
        resolve_frozen_config(CONFIG, overrides={"reproducibility.seed": 2})


def test_config_002_accepts_no_override() -> None:
    assert resolve_frozen_config(CONFIG, overrides={}).sha256 == load_frozen_config(CONFIG).sha256


def test_loaded_snapshot_detects_later_file_mutation(tmp_path: Path) -> None:
    copied = tmp_path / "frozen.yaml"
    copied.write_bytes(CONFIG.read_bytes())
    config = load_frozen_config(copied)
    copied.write_bytes(copied.read_bytes() + b"\n")

    with pytest.raises(FrozenConfigMutationError, match="changed after load"):
        config.assert_source_unchanged()
