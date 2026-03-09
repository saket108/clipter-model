from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.models import (
    convnext_tiny,
    ConvNeXt_Tiny_Weights,
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)


class ImageEncoder(nn.Module):
    _STAGE_SPECS = {
        "convnext_tiny": [("res3", 3, 192), ("res4", 5, 384), ("res5", 7, 768)],
        "mobilenet_v3_small": [("res3", 3, 24), ("res4", 8, 48), ("res5", 12, 576)],
    }

    def __init__(
        self,
        backbone_name: str = "mobilenet_v3_small",
        pretrained: bool = False,
        output_dim: int = 288,
        multiscale_levels: int = 0,
    ):
        super(ImageEncoder, self).__init__()

        if backbone_name == "convnext_tiny":
            backbone = self._load_backbone(
                builder=convnext_tiny,
                default_weights=ConvNeXt_Tiny_Weights.DEFAULT,
                pretrained=pretrained,
                backbone_name=backbone_name,
            )
            self.features = backbone.features
            self.backbone_out_dim = 768
        elif backbone_name == "mobilenet_v3_small":
            backbone = self._load_backbone(
                builder=mobilenet_v3_small,
                default_weights=MobileNet_V3_Small_Weights.DEFAULT,
                pretrained=pretrained,
                backbone_name=backbone_name,
            )
            self.features = backbone.features
            self.backbone_out_dim = 576
        else:
            raise ValueError(
                f"Unsupported backbone_name='{backbone_name}'. "
                "Choose one of: ['mobilenet_v3_small', 'convnext_tiny']."
            )

        self.backbone_name = backbone_name
        self.output_dim = output_dim
        self.multiscale_levels = int(max(0, multiscale_levels))
        # keep an optional projection in case downstream wants a different dim
        self.proj = (
            nn.Identity()
            if self.backbone_out_dim == self.output_dim
            else nn.Linear(self.backbone_out_dim, self.output_dim)
        )
        self.multiscale_specs = []
        self.multiscale_projs = None
        self.level_embed = None
        if self.multiscale_levels > 0:
            self.multiscale_specs = self.get_stage_specs(levels=self.multiscale_levels)
            self.multiscale_projs = nn.ModuleList(
                [
                    nn.Identity()
                    if channels == self.output_dim
                    else nn.Conv2d(channels, self.output_dim, kernel_size=1)
                    for _, _, channels in self.multiscale_specs
                ]
            )
            self.level_embed = nn.Parameter(torch.zeros(len(self.multiscale_specs), self.output_dim))

    @staticmethod
    def _load_backbone(builder, default_weights, pretrained: bool, backbone_name: str):
        weights = default_weights if pretrained else None
        try:
            return builder(weights=weights)
        except Exception as e:
            if not pretrained:
                raise
            print(
                f"Warning: failed to load pretrained weights for {backbone_name}: {e}. "
                "Falling back to random initialization."
            )
            return builder(weights=None)

    def get_stage_specs(self, levels: int | None = None):
        available = list(self._STAGE_SPECS[self.backbone_name])
        if levels is None or int(levels) <= 0 or int(levels) >= len(available):
            return available
        return available[-int(levels) :]

    def _run_backbone(self, x: torch.Tensor, requested_specs: list[tuple[str, int, int]] | None = None):
        requested_by_index = {}
        if requested_specs:
            requested_by_index = {int(idx): str(name) for name, idx, _ in requested_specs}

        current = x
        outputs: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        for idx, block in enumerate(self.features):
            current = block(current)
            if idx in requested_by_index:
                outputs[requested_by_index[idx]] = current
        return current, outputs

    def extract_stage_features(self, x: torch.Tensor, levels: int | None = None):
        stage_specs = self.get_stage_specs(levels=levels)
        _, stage_outputs = self._run_backbone(x, requested_specs=stage_specs)
        return stage_outputs

    def forward(self, x, return_patch_tokens: bool = True, use_multiscale_tokens: bool = False):
        """
        x: [B, 3, H, W]
        If return_patch_tokens is True, returns patch tokens [B, N, C].
        Otherwise returns a pooled vector [B, output_dim].
        """
        if return_patch_tokens and use_multiscale_tokens and self.multiscale_levels > 0:
            stage_outputs = self.extract_stage_features(x, levels=self.multiscale_levels)
            collected = list(stage_outputs.values())

            tokens_per_level = []
            for level_idx, (feature_map, projector) in enumerate(zip(collected, self.multiscale_projs)):
                projected = projector(feature_map)
                tokens = projected.flatten(2).transpose(1, 2)
                tokens = tokens + self.level_embed[level_idx].view(1, 1, -1)
                tokens_per_level.append(tokens)
            return torch.cat(tokens_per_level, dim=1)

        features, _ = self._run_backbone(x)  # [B, C, H', W']

        if return_patch_tokens:
            # Flatten spatial dimensions into tokens -> [B, N, C]
            patch_tokens = features.flatten(2).transpose(1, 2)
            patch_tokens = self.proj(patch_tokens)
            return patch_tokens

        # global average pool spatial dimensions -> [B, C]
        pooled = features.mean(dim=(2, 3))
        pooled = self.proj(pooled)
        return pooled


if __name__ == "__main__":
    # quick smoke test (patch tokens + pooled)
    model = ImageEncoder()
    model.eval()
    dummy = torch.randn(2, 3, 224, 224)
    tokens = model(dummy, return_patch_tokens=True)
    pooled = model(dummy, return_patch_tokens=False)
    print('patch tokens shape:', tokens.shape)
    print('pooled shape:', pooled.shape)
