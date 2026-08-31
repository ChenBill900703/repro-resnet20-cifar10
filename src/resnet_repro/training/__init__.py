"""Phase 3 optimization, exact-update scheduling, and checkpoint primitives."""

from .checkpoint import CheckpointError, load_checkpoint, save_checkpoint
from .optimizer import build_sgd_optimizer, validate_optimizer_parameter_coverage
from .schedule import ExactUpdateLrController, TrainingCompleteError
from .step import TrainingStepResult, train_one_update
from .engine import RunPaths, build_run_objects, run_training

__all__ = [
    "CheckpointError",
    "ExactUpdateLrController",
    "TrainingCompleteError",
    "TrainingStepResult",
    "RunPaths",
    "build_run_objects",
    "build_sgd_optimizer",
    "load_checkpoint",
    "save_checkpoint",
    "train_one_update",
    "run_training",
    "validate_optimizer_parameter_coverage",
]
