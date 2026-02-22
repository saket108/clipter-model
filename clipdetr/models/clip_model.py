"""High-level CLIP model that composes image + text encoders and projection heads.
Forward returns normalized image/text embeddings and the learned logit_scale parameter.
"""
import torch
import torch.nn as nn

from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder
from .projection import ProjectionHead


class CLIPModel(nn.Module):
    def __init__(
        self,
        embed_dim: int = 288,
        proj_dim: int = 256,
        max_text_len: int = 32,
        vocab_size: int = 2048,
        image_backbone: str = "mobilenet_v3_small",
        image_pretrained: bool = False,
        text_num_heads: int = 8,
        text_num_layers: int = 2,
        text_ff_dim: int = 576,
        text_dropout: float = 0.1,
    ):
        super().__init__()
        # ImageEncoder returns patch tokens [B, N, embed_dim]
        self.image_encoder = ImageEncoder(
            backbone_name=image_backbone,
            pretrained=image_pretrained,
            output_dim=embed_dim,
        )
        # Phase‑2 TextEncoder returns token-level outputs [B, L, embed_dim]
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            max_length=max_text_len,
            embed_dim=embed_dim,
            num_heads=text_num_heads,
            num_layers=text_num_layers,
            ff_dim=text_ff_dim,
            dropout=text_dropout,
        )

        # Projection heads operate on pooled embeddings
        self.image_projection = ProjectionHead(input_dim=embed_dim, proj_dim=proj_dim)
        self.text_projection = ProjectionHead(input_dim=embed_dim, proj_dim=proj_dim)

        # logit scale parameter (learnable) initialized to log(1 / temperature)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1.0 / 0.07)))

    def forward(self, images: torch.Tensor, token_ids: torch.Tensor):
        """Returns (image_embeddings, text_embeddings, logit_scale)
        - image_embeddings: (B, proj_dim) L2-normalized
        - text_embeddings: (B, proj_dim) L2-normalized
        - logit_scale: scalar parameter (torch.Tensor)
        """
        # image_encoder -> patch tokens [B, N, C]
        img_tokens = self.image_encoder(images)
        if img_tokens.dim() == 3:
            img_feats = img_tokens.mean(dim=1)
        else:
            img_feats = img_tokens

        txt_feats = self.text_encoder(token_ids)  # (B, L, D)
        # pool token-level text features to a single vector per example
        if txt_feats.dim() == 3:
            txt_feats = txt_feats.mean(dim=1)

        img_embeds = self.image_projection(img_feats)
        txt_embeds = self.text_projection(txt_feats)

        return img_embeds, txt_embeds, self.logit_scale

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.image_encoder(images)
        if tokens.dim() == 3:
            pooled = tokens.mean(dim=1)
        else:
            pooled = tokens
        return self.image_projection(pooled)

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        txt_feats = self.text_encoder(token_ids)
        if txt_feats.dim() == 3:
            txt_feats = txt_feats.mean(dim=1)
        return self.text_projection(txt_feats)
