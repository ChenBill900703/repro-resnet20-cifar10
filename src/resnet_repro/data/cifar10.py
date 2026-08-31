"""CIFAR-10 metadata and index-preserving dataset wrappers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping

import torch
from torch.utils.data import Dataset

from .transforms import TestingTransform, TrainingTransform, to_float_0_255


@dataclass(frozen=True)
class Cifar10Metadata:
    """The split and tensor metadata fixed by the experiment config."""

    name: str
    train_size: int
    test_size: int
    num_classes: int
    input_shape: tuple[int, int, int]
    official_split: bool

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Cifar10Metadata:
        dataset = config["dataset"]
        return cls(
            name=dataset["name"],
            train_size=dataset["train_size"],
            test_size=dataset["test_size"],
            num_classes=dataset["num_classes"],
            input_shape=tuple(dataset["input_shape"]),
            official_split=dataset["official_split"],
        )

    def with_sizes(
        self,
        *,
        train_size: int,
        test_size: int,
        official_split: bool,
    ) -> Cifar10Metadata:
        """Return metadata for synthetic tests without weakening official checks."""

        return replace(
            self,
            train_size=train_size,
            test_size=test_size,
            official_split=official_split,
        )


class IndexedDataset(Dataset[dict[str, torch.Tensor | int]]):
    """Attach the official sample index and epoch-aware transform to a dataset."""

    def __init__(
        self,
        base_dataset: Dataset[Any],
        *,
        metadata: Cifar10Metadata,
        split: Literal["train", "test"],
        transform: TrainingTransform | TestingTransform | None,
    ) -> None:
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        expected_size = metadata.train_size if split == "train" else metadata.test_size
        actual_size = len(base_dataset)
        if actual_size != expected_size:
            qualifier = "official " if metadata.official_split else ""
            raise ValueError(
                f"{qualifier}{split} size must be {expected_size:,}, got {actual_size:,}"
            )
        if split == "train" and isinstance(transform, TestingTransform):
            raise ValueError("training split cannot use the deterministic test transform")
        if split == "test" and isinstance(transform, TrainingTransform):
            raise ValueError("test split cannot use stochastic training augmentation")
        self.base_dataset = base_dataset
        self.metadata = metadata
        self.split = split
        self.transform = transform
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if type(epoch) is not int or epoch < 0:
            raise ValueError("dataset epoch must be a non-negative integer")
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        image, target = self.base_dataset[index]
        if self.transform is None:
            transformed = to_float_0_255(image)
        elif isinstance(self.transform, TrainingTransform):
            transformed = self.transform(
                image,
                epoch=self.epoch,
                official_index=int(index),
            )
        else:
            transformed = self.transform(image)
        if tuple(transformed.shape) != self.metadata.input_shape:
            raise ValueError(
                "transformed image shape does not match CIFAR-10 metadata: "
                f"{tuple(transformed.shape)} != {self.metadata.input_shape}"
            )
        if isinstance(target, torch.Tensor):
            if target.numel() != 1:
                raise ValueError("CIFAR-10 target tensor must be scalar")
            target = int(target.item())
        return {"image": transformed, "target": int(target), "index": int(index)}
