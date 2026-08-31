from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from resnet_repro.reproducibility import (
    RNGState,
    WorkerSeeder,
    capture_rng_state,
    configure_global_rng,
    derive_seed,
    environment_fingerprint,
    make_dataloader_generator,
    make_worker_generator,
    restore_rng_state,
    sample_augmentation_seed,
    sample_rng_scope,
)


ROOT = Path(__file__).resolve().parents[2]


def test_seed_derivation_is_stable_and_namespaced() -> None:
    assert derive_seed(1, "training-dataloader") == derive_seed(1, "training-dataloader")
    assert derive_seed(1, "training-dataloader") != derive_seed(1, "worker-0")
    assert derive_seed(1, "worker-0") != derive_seed(2, "worker-0")


@pytest.mark.parametrize(("seed", "namespace"), [(-1, "x"), (1, "")])
def test_seed_derivation_rejects_invalid_input(seed: int, namespace: str) -> None:
    with pytest.raises(ValueError):
        derive_seed(seed, namespace)


def test_global_seed_reproduces_python_numpy_and_torch() -> None:
    original = capture_rng_state()
    try:
        configure_global_rng(1, deterministic_algorithms=True, cudnn_benchmark=False, tf32=False)
        first = (random.random(), np.random.rand(4), torch.rand(4))
        configure_global_rng(1, deterministic_algorithms=True, cudnn_benchmark=False, tf32=False)
        second = (random.random(), np.random.rand(4), torch.rand(4))
        assert first[0] == second[0]
        np.testing.assert_array_equal(first[1], second[1])
        torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cudnn.benchmark
    finally:
        restore_rng_state(original)


def test_ckpt_003_rng_and_dataloader_generator_round_trip() -> None:
    original = capture_rng_state()
    generator = make_dataloader_generator(1)
    try:
        configure_global_rng(1)
        state = capture_rng_state(generator)
        expected = (
            random.random(),
            np.random.rand(3),
            torch.rand(3),
            torch.randperm(20, generator=generator),
        )
        restore_rng_state(state, generator)
        actual = (
            random.random(),
            np.random.rand(3),
            torch.rand(3),
            torch.randperm(20, generator=generator),
        )
        assert expected[0] == actual[0]
        np.testing.assert_array_equal(expected[1], actual[1])
        torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)
        torch.testing.assert_close(expected[3], actual[3], rtol=0, atol=0)
    finally:
        restore_rng_state(original)


def test_restore_rejects_generator_presence_mismatch() -> None:
    state = capture_rng_state(make_dataloader_generator(1))
    with pytest.raises(ValueError, match="presence"):
        restore_rng_state(state)


def test_restore_rejects_unknown_schema() -> None:
    current = capture_rng_state()
    bad = RNGState(
        schema_version=999,
        python=current.python,
        numpy=current.numpy,
        torch_cpu=current.torch_cpu,
        torch_cuda=current.torch_cuda,
        dataloader_generator=None,
    )
    with pytest.raises(ValueError, match="unsupported"):
        restore_rng_state(bad)


def test_worker_seeder_is_reproducible() -> None:
    original = capture_rng_state()
    try:
        seeder = WorkerSeeder()
        torch.random.default_generator.manual_seed(12345)
        seeder(3)
        first = (random.random(), np.random.rand(), torch.rand(1))
        torch.random.default_generator.manual_seed(12345)
        seeder(3)
        second = (random.random(), np.random.rand(), torch.rand(1))
        assert first[0] == second[0]
        assert first[1] == second[1]
        torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)
    finally:
        restore_rng_state(original)


def test_worker_generator_is_independent_from_sampler_generator() -> None:
    sampler = make_dataloader_generator(1)
    workers = make_worker_generator(1)
    assert not torch.equal(sampler.get_state(), workers.get_state())


def _augmentation_draw(base_seed: int, epoch: int, sample_index: int):
    with sample_rng_scope(base_seed, epoch, sample_index):
        return (random.random(), np.random.rand(), torch.rand(4))


def test_same_sample_identity_produces_same_augmentation_stream() -> None:
    original = capture_rng_state()
    try:
        first = _augmentation_draw(1, 7, 42)
        random.seed(999)
        np.random.seed(888)
        torch.manual_seed(777)
        second = _augmentation_draw(1, 7, 42)

        assert first[0] == second[0]
        assert first[1] == second[1]
        torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)
    finally:
        restore_rng_state(original)


def test_epoch_or_official_index_changes_augmentation_stream() -> None:
    original = capture_rng_state()
    try:
        baseline = _augmentation_draw(1, 7, 42)
        next_epoch = _augmentation_draw(1, 8, 42)
        next_index = _augmentation_draw(1, 7, 43)
        assert sample_augmentation_seed(1, 7, 42) != sample_augmentation_seed(1, 8, 42)
        assert sample_augmentation_seed(1, 7, 42) != sample_augmentation_seed(1, 7, 43)
        assert baseline[0] != next_epoch[0]
        assert baseline[0] != next_index[0]
        assert not torch.equal(baseline[2], next_epoch[2])
        assert not torch.equal(baseline[2], next_index[2])
    finally:
        restore_rng_state(original)


def test_sample_augmentation_is_independent_of_worker_assignment() -> None:
    original = capture_rng_state()
    try:
        torch.random.default_generator.manual_seed(111)
        WorkerSeeder()(0)
        worker_zero = _augmentation_draw(1, 7, 42)
        torch.random.default_generator.manual_seed(999)
        WorkerSeeder()(3)
        worker_three = _augmentation_draw(1, 7, 42)
        assert worker_zero[0] == worker_three[0]
        assert worker_zero[1] == worker_three[1]
        torch.testing.assert_close(worker_zero[2], worker_three[2], rtol=0, atol=0)
    finally:
        restore_rng_state(original)


def test_sample_rng_scope_restores_worker_rng_stream() -> None:
    original = capture_rng_state()
    try:
        configure_global_rng(123)
        state = capture_rng_state()
        with sample_rng_scope(1, 0, 5):
            random.random()
            np.random.rand()
            torch.rand(1)
        after_scope = (random.random(), np.random.rand(), torch.rand(1))
        restore_rng_state(state)
        expected = (random.random(), np.random.rand(), torch.rand(1))
        assert after_scope[0] == expected[0]
        assert after_scope[1] == expected[1]
        torch.testing.assert_close(after_scope[2], expected[2], rtol=0, atol=0)
    finally:
        restore_rng_state(original)


def test_environment_fingerprint_is_stable_and_tracks_lock() -> None:
    lock = ROOT / "environment" / "requirements-lock.txt"
    first = environment_fingerprint(lock)
    second = environment_fingerprint(lock)

    assert first == second
    assert first["schema_version"] == 1
    assert first["requirements_lock_sha256"]
    assert len(first["fingerprint_sha256"]) == 64
