from __future__ import annotations

from pathlib import Path

import pytest
import torch

from resnet_repro.config import FrozenConfig, load_frozen_config
from resnet_repro.models import Plain20, ResNet20


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "cifar10_plain20_resnet20_frozen.yaml"


@pytest.fixture(scope="session")
def frozen_config() -> FrozenConfig:
    return load_frozen_config(CONFIG_PATH)


@pytest.fixture
def plain20(frozen_config: FrozenConfig) -> Plain20:
    torch.manual_seed(frozen_config.data["reproducibility"]["seed"])
    return Plain20(frozen_config)


@pytest.fixture
def resnet20(frozen_config: FrozenConfig) -> ResNet20:
    torch.manual_seed(frozen_config.data["reproducibility"]["seed"])
    return ResNet20(frozen_config)
