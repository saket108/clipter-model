from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightFPNNeck(nn.Module):
    """Small top-down feature fusion neck for detector memory construction."""

    def __init__(self, in_channels: list[int], out_channels: int):
        super().__init__()
        if len(in_channels) == 0:
            raise ValueError("in_channels must contain at least one stage.")

        self.in_channels = [int(c) for c in in_channels]
        self.out_channels = int(out_channels)

        self.lateral_convs = nn.ModuleList(
            [
                nn.Identity()
                if channels == self.out_channels
                else nn.Conv2d(channels, self.out_channels, kernel_size=1)
                for channels in self.in_channels
            ]
        )
        self.output_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
                    nn.GELU(),
                )
                for _ in self.in_channels
            ]
        )

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(features) != len(self.lateral_convs):
            raise ValueError(
                f"Expected {len(self.lateral_convs)} feature maps, got {len(features)}."
            )

        laterals = [proj(feature) for feature, proj in zip(features, self.lateral_convs)]
        fused = [None] * len(laterals)
        fused[-1] = laterals[-1]

        for idx in range(len(laterals) - 2, -1, -1):
            upsampled = F.interpolate(
                fused[idx + 1],
                size=laterals[idx].shape[-2:],
                mode="nearest",
            )
            fused[idx] = laterals[idx] + upsampled

        return [out_conv(feature) for feature, out_conv in zip(fused, self.output_convs)]
