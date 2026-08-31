"""Config-driven SGD construction with all-learnable weight decay."""

from __future__ import annotations

from collections import Counter

from torch import nn
from torch.optim import Optimizer, SGD

from ..config import FrozenConfig


def _learnable_parameters(model: nn.Module) -> list[nn.Parameter]:
    parameters: list[nn.Parameter] = []
    seen: set[int] = set()
    for parameter in model.parameters():
        if parameter.requires_grad and id(parameter) not in seen:
            parameters.append(parameter)
            seen.add(id(parameter))
    if not parameters:
        raise ValueError("model has no learnable parameters")
    return parameters


def validate_optimizer_parameter_coverage(
    model: nn.Module,
    optimizer: Optimizer,
    *,
    expected_weight_decay: float,
) -> None:
    """Require every unique learnable parameter in exactly one uniform-decay group."""

    expected = _learnable_parameters(model)
    expected_ids = {id(parameter) for parameter in expected}
    grouped = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    counts = Counter(id(parameter) for parameter in grouped)
    duplicates = sorted(identity for identity, count in counts.items() if count != 1)
    missing = expected_ids - set(counts)
    extra = set(counts) - expected_ids
    if duplicates or missing or extra:
        raise ValueError(
            "optimizer parameter coverage must contain every learnable parameter "
            "exactly once"
        )
    for group in optimizer.param_groups:
        if float(group["weight_decay"]) != float(expected_weight_decay):
            raise ValueError("all optimizer parameter groups must use the frozen weight decay")


def build_sgd_optimizer(model: nn.Module, config: FrozenConfig) -> SGD:
    """Build the approved SGD optimizer without parameter-type exceptions."""

    config.assert_source_unchanged()
    recipe = config.data["optimizer"]
    if (
        recipe["type"] != "SGD"
        or recipe["weight_decay_scope"] != "all_learnable_parameters"
        or recipe["nesterov"]
    ):
        raise ValueError("unsupported frozen optimizer recipe")
    parameters = _learnable_parameters(model)
    optimizer = SGD(
        [
            {
                "params": parameters,
                "lr": config.data["learning_rate"]["initial"],
                "momentum": recipe["momentum"],
                "weight_decay": recipe["weight_decay"],
                "nesterov": recipe["nesterov"],
            }
        ],
        lr=config.data["learning_rate"]["initial"],
        momentum=recipe["momentum"],
        weight_decay=recipe["weight_decay"],
        nesterov=recipe["nesterov"],
    )
    validate_optimizer_parameter_coverage(
        model,
        optimizer,
        expected_weight_decay=recipe["weight_decay"],
    )
    return optimizer
