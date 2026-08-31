from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from resnet_repro.config import FrozenConfig
from resnet_repro.data.cifar10 import Cifar10Metadata, IndexedDataset


class _LazyCifarDataset(Dataset):
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        image = np.full((32, 32, 3), index % 256, dtype=np.uint8)
        return image, index % 10


def test_official_cifar10_metadata(frozen_config: FrozenConfig) -> None:
    metadata = Cifar10Metadata.from_config(frozen_config.data)
    assert metadata.name == "CIFAR10"
    assert metadata.train_size == 50_000
    assert metadata.test_size == 10_000
    assert metadata.num_classes == 10
    assert metadata.input_shape == (3, 32, 32)
    assert metadata.official_split


@pytest.mark.parametrize(("split", "size"), [("train", 50_000), ("test", 10_000)])
def test_indexed_dataset_returns_image_target_and_official_index(
    frozen_config: FrozenConfig, split: str, size: int
) -> None:
    metadata = Cifar10Metadata.from_config(frozen_config.data)
    dataset = IndexedDataset(
        _LazyCifarDataset(size), metadata=metadata, split=split, transform=None
    )
    sample = dataset[257]
    assert set(sample) == {"image", "target", "index"}
    assert sample["image"].shape == (3, 32, 32)
    assert sample["image"].dtype == torch.float32
    assert sample["image"].min().item() == 1.0
    assert sample["image"].max().item() == 1.0
    assert sample["target"] == 7
    assert sample["index"] == 257


def test_official_split_size_mismatch_is_rejected(frozen_config: FrozenConfig) -> None:
    metadata = Cifar10Metadata.from_config(frozen_config.data)
    with pytest.raises(ValueError, match="official train size"):
        IndexedDataset(
            _LazyCifarDataset(19), metadata=metadata, split="train", transform=None
        )
