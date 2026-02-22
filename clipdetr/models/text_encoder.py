import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    """Custom lightweight Transformer text encoder (token-level output).

    Default compact design:
    - max_length = 32
    - embed_dim = 288
    - num_layers = 2
    - num_heads = 8
    - feedforward (ff_dim) = 576
    - learnable positional embeddings
    Returns token-level outputs of shape (B, L, D).
    """

    def __init__(
        self,
        vocab_size,
        max_length: int = 32,
        embed_dim: int = 288,
        num_heads: int = 8,
        num_layers: int = 2,
        ff_dim: int = 576,
        dropout: float = 0.1,
    ):
        super(TextEncoder, self).__init__()

        self.max_length = max_length
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional embedding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_length, embed_dim))

        # Transformer Encoder (batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: (B, L) -> returns token-level outputs (B, L, D)"""
        # Support larger upstream token ids by folding them into the configured compact vocab.
        input_ids = input_ids.remainder(self.vocab_size)

        x = self.token_embedding(input_ids)  # [B, L, D]
        x = x + self.pos_embedding[:, : x.size(1), :]

        x = self.transformer(x)  # [B, L, D]
        x = self.layer_norm(x)

        return x


if __name__ == "__main__":
    vocab_size = 10000
    model = TextEncoder(vocab_size=vocab_size)

    dummy_ids = torch.randint(0, vocab_size, (2, 32))
    tokens = model(dummy_ids)

    print(tokens.shape)
