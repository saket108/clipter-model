import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """CLIP-style symmetric InfoNCE loss (averaged image->text and text->image).

    Usage:
        criterion = ContrastiveLoss(temperature=0.07)
        loss = criterion(image_embeddings, text_embeddings)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        # logits: [B, B]
        logits = image_embeddings @ text_embeddings.T
        logits = logits / self.temperature

        labels = torch.arange(logits.size(0), device=logits.device)

        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)

        return (loss_i + loss_t) / 2.0
