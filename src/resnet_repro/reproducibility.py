"""RNG state and environment utilities required by Phase 0."""

from __future__ import annotations

import json
import platform
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def derive_seed(base_seed: int, namespace: str) -> int:
    """Derive a stable non-negative 63-bit seed without Python hash randomization."""

    if type(base_seed) is not int or base_seed < 0:
        raise ValueError("base_seed must be a non-negative integer")
    if not namespace:
        raise ValueError("namespace must not be empty")
    digest = sha256(f"resnet-repro-v1:{base_seed}:{namespace}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def configure_global_rng(
    base_seed: int,
    *,
    deterministic_algorithms: bool = True,
    cudnn_benchmark: bool = False,
    tf32: bool = False,
) -> None:
    """Seed Python, NumPy, torch CPU/CUDA and set deterministic backend policy."""

    if type(base_seed) is not int or base_seed < 0:
        raise ValueError("base_seed must be a non-negative integer")
    random.seed(base_seed)
    np.random.seed(base_seed % (2**32))
    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = deterministic_algorithms
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = tf32
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = tf32


def configure_from_frozen_config(config: Mapping[str, Any]) -> None:
    reproducibility = config["reproducibility"]
    configure_global_rng(
        reproducibility["seed"],
        deterministic_algorithms=reproducibility["deterministic_algorithms"],
        cudnn_benchmark=reproducibility["cudnn_benchmark"],
        tf32=config["hardware"]["tf32"],
    )


def make_dataloader_generator(base_seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(derive_seed(base_seed, "training-dataloader"))
    return generator


def make_worker_generator(base_seed: int) -> torch.Generator:
    """Create the independent generator DataLoader uses for worker base seeds."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(derive_seed(base_seed, "dataloader-workers"))
    return generator


@dataclass(frozen=True)
class WorkerSeeder:
    """Pickle-safe worker_init_fn using DataLoader's per-iterator torch seed.

    The DataLoader ``generator`` must come from :func:`make_worker_generator`.
    This synchronizes Python and NumPy with the worker seed without restarting
    every worker at a fixed ``(base_seed, worker_id)`` stream on each epoch.
    Exact augmentation replay is provided separately by
    :func:`sample_rng_scope` and does not depend on worker scheduling.
    """

    def __call__(self, worker_id: int) -> None:
        if type(worker_id) is not int or worker_id < 0:
            raise ValueError("worker_id must be a non-negative integer")
        worker_seed = torch.initial_seed()
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2**32))
        torch.random.default_generator.manual_seed(worker_seed)


def sample_augmentation_seed(base_seed: int, epoch: int, sample_index: int) -> int:
    """Derive augmentation randomness from sample identity, never worker order."""

    if type(epoch) is not int or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    if type(sample_index) is not int or sample_index < 0:
        raise ValueError("sample_index must be a non-negative integer")
    return derive_seed(base_seed, f"augmentation:epoch={epoch}:sample={sample_index}")


@contextmanager
def sample_rng_scope(base_seed: int, epoch: int, sample_index: int):
    """Temporarily seed Python/NumPy/torch CPU for one sample transform.

    Dataset ``__getitem__`` implementations must wrap every stochastic
    augmentation in this scope.  The same sample in the same epoch then gets
    identical augmentation after resume regardless of worker assignment,
    prefetch depth, or process restart.
    """

    seed = sample_augmentation_seed(base_seed, epoch, sample_index)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.random.default_generator.manual_seed(seed)
    try:
        yield seed
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


@dataclass(frozen=True)
class RNGState:
    schema_version: int
    python: object
    numpy: tuple[Any, ...]
    torch_cpu: torch.Tensor
    torch_cuda: tuple[torch.Tensor, ...]
    dataloader_generator: torch.Tensor | None


def capture_rng_state(
    dataloader_generator: torch.Generator | None = None,
) -> RNGState:
    cuda_states: tuple[torch.Tensor, ...] = ()
    if torch.cuda.is_available():
        cuda_states = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    generator_state = (
        dataloader_generator.get_state().clone()
        if dataloader_generator is not None
        else None
    )
    return RNGState(
        schema_version=1,
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch_cpu=torch.get_rng_state().clone(),
        torch_cuda=cuda_states,
        dataloader_generator=generator_state,
    )


def restore_rng_state(
    state: RNGState,
    dataloader_generator: torch.Generator | None = None,
) -> None:
    if state.schema_version != 1:
        raise ValueError(f"unsupported RNG state schema: {state.schema_version}")
    if (state.dataloader_generator is None) != (dataloader_generator is None):
        raise ValueError("DataLoader generator presence must match the saved RNG state")
    if state.torch_cuda and not torch.cuda.is_available():
        raise RuntimeError("saved CUDA RNG states cannot be restored without CUDA")
    random.setstate(state.python)
    np.random.set_state(state.numpy)
    torch.set_rng_state(state.torch_cpu)
    if state.torch_cuda:
        if len(state.torch_cuda) != torch.cuda.device_count():
            raise RuntimeError("saved CUDA RNG state count differs from visible CUDA devices")
        torch.cuda.set_rng_state_all(list(state.torch_cuda))
    if dataloader_generator is not None and state.dataloader_generator is not None:
        dataloader_generator.set_state(state.dataloader_generator)


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return sha256(path.read_bytes()).hexdigest().upper()


def environment_fingerprint(requirements_lock: str | Path | None = None) -> dict[str, Any]:
    """Return auditable environment facts plus a canonical payload hash."""

    lock_path = Path(requirements_lock).resolve() if requirements_lock else None
    cuda_available = torch.cuda.is_available()
    gpu_names = (
        [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        if cuda_available
        else []
    )
    facts: dict[str, Any] = {
        "schema_version": 1,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_names": gpu_names,
        "requirements_lock_sha256": _file_sha256(lock_path) if lock_path else None,
    }
    canonical = json.dumps(facts, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    facts["fingerprint_sha256"] = sha256(canonical.encode("ascii")).hexdigest().upper()
    return facts
