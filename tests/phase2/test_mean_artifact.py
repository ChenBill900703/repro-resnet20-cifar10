from __future__ import annotations

from pathlib import Path

import pytest
import torch

from resnet_repro.config import FrozenConfig
from resnet_repro.data.mean_artifact import (
    create_mean_artifact,
    load_mean_artifact,
)


def _fifty_thousand_training_images():
    image = torch.stack(
        (
            torch.full((32, 32), 10, dtype=torch.uint8),
            torch.full((32, 32), 20, dtype=torch.uint8),
            torch.full((32, 32), 30, dtype=torch.uint8),
        )
    )
    for _ in range(10):
        yield image.unsqueeze(0).expand(5_000, -1, -1, -1)


def test_complete_50k_training_mean_artifact_provenance_metadata_and_hash(
    frozen_config: FrozenConfig, tmp_path: Path
) -> None:
    path = tmp_path / "cifar10_train_mean.bin"
    artifact = create_mean_artifact(
        _fifty_thousand_training_images(),
        path,
        frozen_config.data,
        source_split="train",
    )

    assert artifact.mean.shape == (3, 32, 32)
    assert artifact.mean.dtype == torch.float32
    torch.testing.assert_close(
        artifact.mean[:, 0, 0], torch.tensor([10.0, 20.0, 30.0]), rtol=0, atol=0
    )
    assert artifact.metadata["dataset"] == "CIFAR10"
    assert artifact.metadata["source_split"] == "train"
    assert artifact.metadata["sample_count"] == 50_000
    assert artifact.metadata["shape"] == [3, 32, 32]
    assert artifact.metadata["source"] == "full_official_training_set"
    assert len(artifact.sha256) == 64
    assert path.is_file()

    restored = load_mean_artifact(path, frozen_config.data)
    assert restored.sha256 == artifact.sha256
    assert restored.metadata == artifact.metadata
    torch.testing.assert_close(restored.mean, artifact.mean, rtol=0, atol=0)


def test_mean_artifact_bytes_and_hash_are_deterministic(
    frozen_config: FrozenConfig, tmp_path: Path
) -> None:
    first = create_mean_artifact(
        _fifty_thousand_training_images(),
        tmp_path / "first.bin",
        frozen_config.data,
        source_split="train",
    )
    second = create_mean_artifact(
        _fifty_thousand_training_images(),
        tmp_path / "second.bin",
        frozen_config.data,
        source_split="train",
    )
    assert first.sha256 == second.sha256
    assert first.path.read_bytes() == second.path.read_bytes()


def test_mean_rejects_test_statistics_and_wrong_sample_count(
    frozen_config: FrozenConfig, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="training split"):
        create_mean_artifact(
            _fifty_thousand_training_images(),
            tmp_path / "test.bin",
            frozen_config.data,
            source_split="test",
        )
    with pytest.raises(ValueError, match="50,000"):
        create_mean_artifact(
            [torch.zeros(1, 3, 32, 32, dtype=torch.uint8)],
            tmp_path / "short.bin",
            frozen_config.data,
            source_split="train",
        )
