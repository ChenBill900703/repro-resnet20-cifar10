from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torchvision.datasets import CIFAR10

from resnet_repro.bn_compatibility import run_bn_compatibility
from resnet_repro.config import load_frozen_config
from resnet_repro.data import load_mean_artifact
from resnet_repro.reproducibility import configure_from_frozen_config, environment_fingerprint
from resnet_repro.training.engine import EXPECTED_SOURCE_COMMIT, _atomic_write_json, source_provenance, utc_now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and record every formal Phase 0-4 preflight gate.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = ROOT / "configs" / "cifar10_plain20_resnet20_frozen.yaml"
    mean_path = ROOT / "artifacts" / "cifar10" / "cifar10_train_mean_v1.bin"
    config = load_frozen_config(config_path)
    configure_from_frozen_config(config.data)
    mean = load_mean_artifact(mean_path, config.data)
    provenance = source_provenance(ROOT)
    environment = environment_fingerprint(ROOT / "environment" / "requirements-lock.txt")
    started = time.perf_counter()
    tests = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/phase0",
            "tests/phase1",
            "tests/phase2",
            "tests/phase3",
            "tests/phase4",
            "-q",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    bn = run_bn_compatibility(
        pytorch_momentum=config.data["batch_normalization"]["pytorch_momentum"],
        epsilon=config.data["batch_normalization"]["epsilon"],
    )
    train = CIFAR10(root=str(ROOT / "data"), train=True, download=False)
    test = CIFAR10(root=str(ROOT / "data"), train=False, download=False)
    checks = {
        "source_commit": (
            EXPECTED_SOURCE_COMMIT is None
            or provenance["commit"] == EXPECTED_SOURCE_COMMIT
        ),
        "source_clean": not provenance["dirty"],
        "config_frozen": config.data["config_status"] == "frozen",
        "mean_hash": mean.sha256 == "6DAFA62D5751FB9EAA9537BF61D9485DF692926CEC1B16B4BFF53927C40AA0F1",
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) == config.data["hardware"]["expected_gpu"],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark_disabled": not torch.backends.cudnn.benchmark,
        "tf32_matmul_disabled": not torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn_disabled": not torch.backends.cudnn.allow_tf32,
        "official_train_size": len(train) == config.data["dataset"]["train_size"],
        "official_test_size": len(test) == config.data["dataset"]["test_size"],
        "phase0_through_phase4_tests": tests.returncode == 0,
        "bn_compatibility": bn.compatible,
    }
    report = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "overall_pass": all(checks.values()),
        "checks": checks,
        "config_sha256": config.sha256,
        "mean_artifact_sha256": mean.sha256,
        "source_commit": provenance["commit"],
        "source_dirty": provenance["dirty"],
        "source_status_porcelain": provenance["status_porcelain"],
        "environment_fingerprint_sha256": environment["fingerprint_sha256"],
        "environment": environment,
        "pytest_returncode": tests.returncode,
        "pytest_stdout": tests.stdout,
        "pytest_stderr": tests.stderr,
        "bn_conclusion": bn.conclusion,
        "bn_evaluation_max_abs_diff": bn.evaluation_max_abs_diff,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_write_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
