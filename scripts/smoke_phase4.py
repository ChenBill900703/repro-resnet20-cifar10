from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ordered Phase 4 smoke and exact-resume gates.")
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _run(model: str, updates: int, run_dir: Path, *, resume: Path | None = None) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "--model",
        model,
        "--mode",
        "smoke",
        "--run-dir",
        str(run_dir),
        "--updates",
        str(updates),
        "--trace-batches",
    ]
    if resume is not None:
        command.extend(("--resume", str(resume)))
    log_path = run_dir.parent / f"{run_dir.name}_command.log"
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    log_path.write_text(result.stdout + "\n--- STDERR ---\n" + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"{model} {updates}-update smoke failed; see {log_path}")


def _events(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _updates(path: Path) -> list[dict[str, Any]]:
    return [event for event in _events(path) if event["event"] == "train_update"]


def _assert_nested_equal(left: Any, right: Any, path: str = "root") -> None:
    import numpy as np
    import torch

    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or not torch.equal(left, right):
            raise AssertionError(f"tensor mismatch at {path}")
    elif isinstance(left, np.ndarray):
        if not isinstance(right, np.ndarray) or not np.array_equal(left, right):
            raise AssertionError(f"array mismatch at {path}")
    elif isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise AssertionError(f"mapping keys mismatch at {path}")
        for key in left:
            _assert_nested_equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (tuple, list)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            raise AssertionError(f"sequence mismatch at {path}")
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _assert_nested_equal(left_item, right_item, f"{path}[{index}]")
    elif left != right:
        raise AssertionError(f"value mismatch at {path}: {left!r} != {right!r}")


def _compare_resume(reference_dir: Path, resumed_dir: Path) -> dict[str, Any]:
    import torch

    reference_updates = _updates(reference_dir / "events.jsonl")
    resumed_updates = _updates(resumed_dir / "events.jsonl")
    if len(reference_updates) != len(resumed_updates) or len(reference_updates) != 10:
        raise AssertionError("resume trace must contain exactly updates 1 through 10")
    trace_fields = (
        "update",
        "learning_rate",
        "loss",
        "correct_count",
        "sample_count",
        "batch_indices_sha256",
        "augmented_images_sha256",
        "indices",
        "sampler_epoch",
        "sampler_consumed_cursor",
        "dataset_epoch",
    )
    for expected, actual in zip(reference_updates, resumed_updates, strict=True):
        for field in trace_fields:
            if expected[field] != actual[field]:
                raise AssertionError(f"resume trace mismatch at update {expected['update']} field {field}")

    reference = torch.load(
        reference_dir / "checkpoints" / "checkpoint_update_000010.pt",
        map_location="cpu",
        weights_only=False,
    )
    resumed = torch.load(
        resumed_dir / "checkpoints" / "checkpoint_update_000010.pt",
        map_location="cpu",
        weights_only=False,
    )
    state_fields = (
        "model_state",
        "optimizer_state",
        "completed_updates",
        "current_lr",
        "scheduler_state",
        "sampler_state",
        "dataset_epoch",
        "python_rng_state",
        "numpy_rng_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_states",
        "dataloader_generator_state",
        "augmentation_rng_policy",
    )
    for field in state_fields:
        _assert_nested_equal(reference[field], resumed[field], field)
    return {
        "pass": True,
        "updates_compared": 10,
        "trace_fields": list(trace_fields),
        "checkpoint_state_fields": list(state_fields),
        "reference_checkpoint": str(reference_dir / "checkpoints" / "checkpoint_update_000010.pt"),
        "resumed_checkpoint": str(resumed_dir / "checkpoints" / "checkpoint_update_000010.pt"),
    }


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": 1,
        "overall_pass": False,
        "ordered_gates": [],
        "models": {},
    }
    for updates in (1, 10, 100):
        for model in ("plain20", "resnet20"):
            run_dir = output_root / f"{model}_smoke_{updates:03d}"
            _run(model, updates, run_dir)
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            gate = {
                "gate": f"{updates}_update",
                "model": model,
                "pass": manifest["status"] == "completed" and manifest["completed_updates"] == updates,
                "run_dir": str(run_dir),
                "samples_per_second": manifest["samples_per_second"],
                "peak_cuda_memory_bytes": manifest["peak_cuda_memory_bytes"],
                "wall_seconds": manifest["wall_seconds"],
                "checkpoint": manifest["latest_checkpoint"],
            }
            if not gate["pass"]:
                raise RuntimeError(f"ordered smoke gate failed: {gate}")
            report["ordered_gates"].append(gate)

    for model in ("plain20", "resnet20"):
        reference_dir = output_root / f"{model}_smoke_010"
        resumed_dir = output_root / f"{model}_resume_004_to_010"
        _run(model, 4, resumed_dir)
        checkpoint = resumed_dir / "checkpoints" / "checkpoint_update_000004.pt"
        _run(model, 10, resumed_dir, resume=checkpoint)
        report["models"][model] = {"resume_equivalence": _compare_resume(reference_dir, resumed_dir)}
    report["overall_pass"] = True
    report_path = output_root / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
