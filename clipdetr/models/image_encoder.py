import torch
import torch.nn as nn
from torchvision.models import (
    convnext_tiny,
    ConvNeXt_Tiny_Weights,
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)


class ImageEncoder(nn.Module):
    def __init__(
        self,
        backbone_name: str = "mobilenet_v3_small",
        pretrained: bool = False,
        output_dim: int = 288,
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
        # keep an optional projection in case downstream wants a different dim
        self.proj = (
            nn.Identity()
            if self.backbone_out_dim == self.output_dim
            else nn.Linear(self.backbone_out_dim, self.output_dim)
        )

    def forward(self, x, return_patch_tokens: bool = True):
        """
        x: [B, 3, H, W]
        If return_patch_tokens is True, returns patch tokens [B, N, C].
        Otherwise returns a pooled vector [B, output_dim].
        """

        features = self.features(x)  # [B, C, H', W']
        B, C, H, W = features.shape

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
