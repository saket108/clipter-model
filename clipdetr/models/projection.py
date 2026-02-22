"""Projection head to map encoder outputs into the CLIP embedding space and L2-normalize."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int = 768, proj_dim: int = 512):
        super().__init__()
        # simple linear projection followed by layer-norm + L2 normalize
        self.projection = nn.Linear(input_dim, proj_dim)
        self.layer_norm = nn.LayerNorm(proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # support token-level ([B, L, D]) or pooled ([B, D]) inputs
        x = self.projection(x)
        x = self.layer_norm(x)
        return F.normalize(x, p=2, dim=-1)
