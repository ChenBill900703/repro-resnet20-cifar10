from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resnet_repro.training.engine import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen CIFAR-10 Plain-20/ResNet-20 training engine."
    )
    parser.add_argument("--model", choices=("plain20", "resnet20"), required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--trace-batches", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "cifar10_plain20_resnet20_frozen.yaml",
    )
    parser.add_argument(
        "--mean",
        type=Path,
        default=ROOT / "artifacts" / "cifar10" / "cifar10_train_mean_v1.bin",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "smoke" and args.updates is None:
        raise SystemExit("--updates is required for smoke mode")
    if args.mode == "formal" and args.updates not in (None, 64000):
        raise SystemExit("formal mode only accepts the frozen 64,000-update target")
    result = run_training(
        config_path=args.config,
        mean_path=args.mean,
        data_root=args.data_root,
        model_name=args.model,
        run_dir=args.run_dir,
        mode=args.mode,
        target_updates=args.updates,
        resume_checkpoint=args.resume,
        preflight_report=args.preflight_report,
        trace_batches=args.trace_batches,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
