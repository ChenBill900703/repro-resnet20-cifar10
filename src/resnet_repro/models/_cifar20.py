"""Internal config-driven backbone shared by Plain-20 and ResNet-20."""

from __future__ import annotations

import torch
from torch import nn

from resnet_repro.config import FrozenConfig

from .blocks import PlainBlock, PostActivationResidualBlock, make_batch_norm, make_conv3x3
from .initialization import initialize_module_tree


BlockType = type[PlainBlock] | type[PostActivationResidualBlock]


class Cifar20Base(nn.Module):
    def __init__(self, config: FrozenConfig, block_type: BlockType) -> None:
        super().__init__()
        config.assert_source_unchanged()
        architecture = config.data["architecture"]
        dataset = config.data["dataset"]
        expected_block_order = (
            "conv",
            "batch_norm",
            "relu",
            "conv",
            "batch_norm",
            "add",
            "relu",
        )
        if (
            architecture["block"]["type"] != "post_activation_basic"
            or architecture["block"]["order"] != expected_block_order
            or not architecture["global_average_pooling"]
        ):
            raise ValueError("unsupported Phase 1 architecture semantics")
        self.model_name = "unassigned"
        self.input_shape = tuple(dataset["input_shape"])
        self.stage_block_counts = tuple(architecture["stage_blocks"])
        self.stage_channels = tuple(architecture["stage_channels"])
        self.stage_spatial_sizes = tuple(architecture["stage_spatial_sizes"])
        self.weighted_layers = architecture["weighted_layers"]

        stem = architecture["stem"]
        self.stem_conv = make_conv3x3(
            config.data,
            self.input_shape[0],
            stem["out_channels"],
            stride=stem["stride"],
        )
        self.stem_bn = make_batch_norm(config.data, stem["out_channels"])
        self.stem_relu = nn.ReLU(inplace=False)

        current_channels = stem["out_channels"]
        stages: list[nn.Sequential] = []
        for stage_index, (block_count, out_channels) in enumerate(
            zip(self.stage_block_counts, self.stage_channels, strict=True)
        ):
            blocks: list[nn.Module] = []
            for block_index in range(block_count):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                blocks.append(
                    block_type(
                        config.data,
                        current_channels,
                        out_channels,
                        stride=stride,
                    )
                )
                current_channels = out_channels
            stages.append(nn.Sequential(*blocks))
        self.stage1, self.stage2, self.stage3 = stages
        self.global_average_pool = nn.AdaptiveAvgPool2d((1, 1))
        classifier = architecture["classifier"]
        self.classifier = nn.Linear(
            classifier["in_features"],
            classifier["out_features"],
            bias=classifier["bias"],
        )
        initialize_module_tree(self, config.data)

    def _check_input(self, value: torch.Tensor) -> None:
        if value.ndim != 4:
            raise ValueError(f"model expects 4D NCHW input, got {value.ndim}D")
        expected = self.input_shape
        if tuple(value.shape[1:]) != expected:
            raise ValueError(
                f"model expects input shape [N,{expected[0]},{expected[1]},{expected[2]}], "
                f"got {list(value.shape)}"
            )

    def forward_features(
        self, value: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        self._check_input(value)
        value = self.stem_relu(self.stem_bn(self.stem_conv(value)))
        stem_output = value
        value = self.stage1(value)
        stage1_output = value
        value = self.stage2(value)
        stage2_output = value
        value = self.stage3(value)
        stage3_output = value
        return value, (stem_output, stage1_output, stage2_output, stage3_output)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value, _ = self.forward_features(value)
        value = self.global_average_pool(value)
        value = torch.flatten(value, 1)
        return self.classifier(value)
