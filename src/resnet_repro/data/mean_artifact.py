"""Deterministic, auditable serialization for the full training-set mean."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch


_MAGIC = b"RESNET_REPRO_CIFAR10_MEAN_V1\n"


@dataclass(frozen=True)
class MeanArtifact:
    mean: torch.Tensor
    metadata: dict[str, Any]
    sha256: str
    path: Path


def _expected_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    dataset = config["dataset"]
    preprocessing = config["preprocessing"]
    return {
        "dataset": dataset["name"],
        "dtype": "float32",
        "input_scale": list(preprocessing["input_scale"]),
        "sample_count": dataset["train_size"],
        "schema_version": 1,
        "shape": list(preprocessing["mean"]["shape"]),
        "source": preprocessing["mean"]["source"],
        "source_split": "train",
    }


def _as_nchw(batch: Any, expected_shape: tuple[int, int, int]) -> torch.Tensor:
    tensor = torch.as_tensor(batch)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError("mean source batches must be NCHW or NHWC image tensors")
    if tuple(tensor.shape[1:]) == expected_shape:
        result = tensor
    elif tuple(tensor.shape[1:]) == (expected_shape[1], expected_shape[2], expected_shape[0]):
        result = tensor.permute(0, 3, 1, 2)
    else:
        raise ValueError("mean source batch shape does not match CIFAR-10")
    if result.numel() and (result.min().item() < 0 or result.max().item() > 255):
        raise ValueError("mean source values must stay in [0,255]")
    return result


def _serialize(mean: torch.Tensor, metadata: Mapping[str, Any]) -> bytes:
    header = json.dumps(
        dict(metadata), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    values = np.asarray(mean.cpu(), dtype="<f4").tobytes(order="C")
    return _MAGIC + header + b"\n" + values


def create_mean_artifact(
    image_batches: Iterable[Any],
    destination: str | Path,
    config: Mapping[str, Any],
    *,
    source_split: str,
) -> MeanArtifact:
    """Compute the per-pixel mean from exactly the official 50k train images."""

    if source_split != "train":
        raise ValueError("mean statistics must come only from the training split")
    metadata = _expected_metadata(config)
    expected_count = metadata["sample_count"]
    expected_shape = tuple(metadata["shape"])
    total = torch.zeros(expected_shape, dtype=torch.float64)
    sample_count = 0
    for batch in image_batches:
        images = _as_nchw(batch, expected_shape)
        total += images.sum(dim=0, dtype=torch.float64)
        sample_count += int(images.shape[0])
        if sample_count > expected_count:
            raise ValueError(f"mean artifact requires exactly {expected_count:,} images")
    if sample_count != expected_count:
        raise ValueError(f"mean artifact requires exactly {expected_count:,} images")
    mean = (total / sample_count).to(torch.float32)
    payload = _serialize(mean, metadata)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    digest = sha256(payload).hexdigest().upper()
    return MeanArtifact(mean=mean, metadata=metadata, sha256=digest, path=path)


def load_mean_artifact(
    source: str | Path,
    config: Mapping[str, Any],
) -> MeanArtifact:
    """Load and validate an artifact without accepting metadata defaults."""

    path = Path(source)
    payload = path.read_bytes()
    if not payload.startswith(_MAGIC):
        raise ValueError("unrecognized CIFAR-10 mean artifact format")
    remainder = payload[len(_MAGIC) :]
    header, separator, raw_mean = remainder.partition(b"\n")
    if not separator:
        raise ValueError("mean artifact metadata header is missing")
    try:
        metadata = json.loads(header.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("mean artifact metadata is invalid") from exc
    expected = _expected_metadata(config)
    if metadata != expected:
        raise ValueError("mean artifact metadata does not match frozen config")
    expected_values = int(np.prod(expected["shape"]))
    values = np.frombuffer(raw_mean, dtype="<f4")
    if values.size != expected_values:
        raise ValueError("mean artifact payload size is invalid")
    mean = torch.from_numpy(values.copy()).reshape(expected["shape"])
    return MeanArtifact(
        mean=mean,
        metadata=metadata,
        sha256=sha256(payload).hexdigest().upper(),
        path=path,
    )
