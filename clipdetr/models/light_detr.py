"""Lightweight DETR-style detector built on top of the project's image encoder."""
import torch
import torch.nn as nn

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
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries

        self.image_encoder = ImageEncoder(
            backbone_name=image_backbone,
            pretrained=image_pretrained,
            output_dim=hidden_dim,
        )
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

    def forward(self, images: torch.Tensor):
        memory = self.image_encoder(images, return_patch_tokens=True)  # [B, N, D]
        memory = self.memory_norm(memory)

        bs = memory.size(0)
        query_embed = self.query_embed.weight.unsqueeze(0).expand(bs, -1, -1)
        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(tgt + query_embed, memory)

        pred_logits = self.class_embed(hs)
        pred_boxes = self.bbox_embed(hs).sigmoid()
        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
