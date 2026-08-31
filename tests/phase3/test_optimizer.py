from __future__ import annotations

import pytest
import torch
from torch import nn

from resnet_repro.batch_norm import CaffeCompatibleBatchNorm2d
from resnet_repro.config import FrozenConfig
from resnet_repro.models import Plain20, ResNet20
from resnet_repro.training.optimizer import (
    build_sgd_optimizer,
    validate_optimizer_parameter_coverage,
)


@pytest.mark.parametrize("model_class", [Plain20, ResNet20])
def test_opt_001_002_sgd_recipe_and_exact_parameter_coverage(
    frozen_config: FrozenConfig,
    model_class: type[Plain20] | type[ResNet20],
) -> None:
    model = model_class(frozen_config)
    optimizer = build_sgd_optimizer(model, frozen_config)

    assert isinstance(optimizer, torch.optim.SGD)
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert group["momentum"] == 0.9
    assert group["weight_decay"] == 0.0001
    assert group["nesterov"] is False
    grouped_ids = [id(parameter) for parameter in group["params"]]
    learnable_ids = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
    assert len(grouped_ids) == len(set(grouped_ids))
    assert set(grouped_ids) == set(learnable_ids)


def test_opt_003_bn_affine_and_fc_bias_receive_same_decay(
    frozen_config: FrozenConfig,
) -> None:
    model = ResNet20(frozen_config)
    optimizer = build_sgd_optimizer(model, frozen_config)
    parameter_to_decay = {
        id(parameter): group["weight_decay"]
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    batch_norms = [
        module for module in model.modules() if isinstance(module, CaffeCompatibleBatchNorm2d)
    ]
    assert batch_norms
    for batch_norm in batch_norms:
        assert parameter_to_decay[id(batch_norm.weight)] == 0.0001
        assert parameter_to_decay[id(batch_norm.bias)] == 0.0001
    assert model.classifier.bias is not None
    assert parameter_to_decay[id(model.classifier.bias)] == 0.0001
    assert all(
        module.bias is None for module in model.modules() if isinstance(module, nn.Conv2d)
    )


def test_optimizer_validator_rejects_duplicate_and_missing_parameters() -> None:
    model = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1))
    parameters = list(model.parameters())
    optimizer = torch.optim.SGD(parameters[:2], lr=0.1, weight_decay=0.0001)
    with pytest.raises(ValueError, match="exactly once"):
        validate_optimizer_parameter_coverage(
            model, optimizer, expected_weight_decay=0.0001
        )

    optimizer = torch.optim.SGD(parameters, lr=0.1, weight_decay=0.0001)
    optimizer.param_groups[0]["params"].append(parameters[0])
    with pytest.raises(ValueError, match="exactly once"):
        validate_optimizer_parameter_coverage(
            model, optimizer, expected_weight_decay=0.0001
        )
