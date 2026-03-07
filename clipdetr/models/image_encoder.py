import torch
import torch.nn as nn
from torchvision.models import (
    convnext_tiny,
    ConvNeXt_Tiny_Weights,
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)


class ImageEncoder(nn.Module):
    _MULTISCALE_SPECS = {
        "convnext_tiny": [(3, 192), (5, 384), (7, 768)],
        "mobilenet_v3_small": [(3, 24), (8, 48), (12, 576)],
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
            weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            backbone = convnext_tiny(weights=weights)
            self.features = backbone.features
            self.backbone_out_dim = 768
        elif backbone_name == "mobilenet_v3_small":
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            backbone = mobilenet_v3_small(weights=weights)
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
        self.multiscale_indices = set()
        self.multiscale_projs = None
        self.level_embed = None
        if self.multiscale_levels > 0:
            available = list(self._MULTISCALE_SPECS[self.backbone_name])
            self.multiscale_specs = available[-self.multiscale_levels :]
            self.multiscale_indices = {idx for idx, _ in self.multiscale_specs}
            self.multiscale_projs = nn.ModuleList(
                [
                    nn.Identity() if channels == self.output_dim else nn.Conv2d(channels, self.output_dim, kernel_size=1)
                    for _, channels in self.multiscale_specs
                ]
            )
            self.level_embed = nn.Parameter(torch.zeros(len(self.multiscale_specs), self.output_dim))

    def forward(self, x, return_patch_tokens: bool = True, use_multiscale_tokens: bool = False):
        """
        x: [B, 3, H, W]
        If return_patch_tokens is True, returns patch tokens [B, N, C].
        Otherwise returns a pooled vector [B, output_dim].
        """
        if return_patch_tokens and use_multiscale_tokens and self.multiscale_levels > 0:
            current = x
            collected = []
            for idx, block in enumerate(self.features):
                current = block(current)
                if idx in self.multiscale_indices:
                    collected.append(current)

            tokens_per_level = []
            for level_idx, (feature_map, projector) in enumerate(zip(collected, self.multiscale_projs)):
                projected = projector(feature_map)
                tokens = projected.flatten(2).transpose(1, 2)
                tokens = tokens + self.level_embed[level_idx].view(1, 1, -1)
                tokens_per_level.append(tokens)
            return torch.cat(tokens_per_level, dim=1)

        features = self.features(x)  # [B, C, H', W']

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
