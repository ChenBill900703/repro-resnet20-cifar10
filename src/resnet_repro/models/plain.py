"""CIFAR-10 Plain-20 model."""

from __future__ import annotations

from resnet_repro.config import FrozenConfig

from ._cifar20 import Cifar20Base
from .blocks import PlainBlock


class Plain20(Cifar20Base):
    def __init__(self, config: FrozenConfig) -> None:
        super().__init__(config, PlainBlock)
        self.model_name = "plain20"
