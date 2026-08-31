"""CIFAR-10 post-activation ResNet-20 model."""

from __future__ import annotations

from resnet_repro.config import FrozenConfig

from ._cifar20 import Cifar20Base
from .blocks import PostActivationResidualBlock


class ResNet20(Cifar20Base):
    def __init__(self, config: FrozenConfig) -> None:
        super().__init__(config, PostActivationResidualBlock)
        self.model_name = "resnet20"
