from __future__ import annotations

from pathlib import Path

import pytest

from resnet_repro.config import FrozenConfig, load_frozen_config


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def frozen_config() -> FrozenConfig:
    return load_frozen_config(ROOT / "configs" / "cifar10_plain20_resnet20_frozen.yaml")
