"""Lightweight DETR-style detector built on top of the project's image encoder."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_neck import LightweightFPNNeck
from .image_encoder import ImageEncoder


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


class LightDETR(nn.Module):
    """A compact detector with Transformer decoder queries.

    Outputs:
      - pred_logits: [B, Q, num_classes + 1]
      - pred_boxes:  [B, Q, 4]  (normalized cx, cy, w, h in [0, 1])
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 288,
        num_queries: int = 50,
        decoder_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 576,
        dropout: float = 0.1,
        image_backbone: str = "mobilenet_v3_small",
        image_pretrained: bool = False,
        use_multiscale_memory: bool = False,
        use_multiscale_neck: bool = False,
        multiscale_levels: int = 3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.use_multiscale_memory = bool(use_multiscale_memory)
        self.use_multiscale_neck = bool(use_multiscale_neck)
        self.multiscale_levels = int(multiscale_levels)
        if self.use_multiscale_memory and self.use_multiscale_neck:
            raise ValueError(
                "use_multiscale_memory and use_multiscale_neck are mutually exclusive."
            )

        self.image_encoder = ImageEncoder(
            backbone_name=image_backbone,
            pretrained=image_pretrained,
            output_dim=hidden_dim,
            multiscale_levels=self.multiscale_levels if self.use_multiscale_memory else 0,
        )
        self.multiscale_neck = None
        self.neck_level_embed = None
        self.neck_pool_sizes: list[int] = []
        if self.use_multiscale_neck:
            stage_specs = self.image_encoder.get_stage_specs(levels=self.multiscale_levels)
            self.multiscale_neck = LightweightFPNNeck(
                in_channels=[channels for _, _, channels in stage_specs],
                out_channels=hidden_dim,
            )
            self.neck_level_embed = nn.Parameter(torch.zeros(len(stage_specs), hidden_dim))
            self.neck_pool_sizes = self._default_neck_pool_sizes(len(stage_specs))
        self.memory_norm = nn.LayerNorm(hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 is no-object class
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

    @staticmethod
    def _default_neck_pool_sizes(levels: int) -> list[int]:
        sizes = [8, 4, 2, 1]
        if levels <= len(sizes):
            return sizes[:levels]
        while len(sizes) < levels:
            sizes.append(1)
        return sizes[:levels]

    def _flatten_neck_features(self, feature_maps: list[torch.Tensor]) -> torch.Tensor:
        tokens_per_level = []
        for level_idx, (feature_map, pool_size) in enumerate(zip(feature_maps, self.neck_pool_sizes)):
            pooled = F.adaptive_avg_pool2d(feature_map, output_size=(pool_size, pool_size))
            tokens = pooled.flatten(2).transpose(1, 2)
            tokens = tokens + self.neck_level_embed[level_idx].view(1, 1, -1)
            tokens_per_level.append(tokens)
        return torch.cat(tokens_per_level, dim=1)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        if self.use_multiscale_neck:
            stage_maps = self.image_encoder.extract_stage_features(
                images,
                levels=self.multiscale_levels,
            )
            fused_maps = self.multiscale_neck(list(stage_maps.values()))
            memory = self._flatten_neck_features(fused_maps)
        else:
            memory = self.image_encoder(
                images,
                return_patch_tokens=True,
                use_multiscale_tokens=self.use_multiscale_memory,
            )  # [B, N, D]
        return self.memory_norm(memory)

    def forward(self, images: torch.Tensor):
        memory = self.encode_images(images)
        bs = memory.size(0)
        query_embed = self.query_embed.weight.unsqueeze(0).expand(bs, -1, -1)
        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(tgt + query_embed, memory)

        pred_logits = self.class_embed(hs)
        pred_boxes = self.bbox_embed(hs).sigmoid()
        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
