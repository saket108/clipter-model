"""Dataset diagnostics helpers for class distribution checks."""
from __future__ import annotations

from typing import Dict, Optional

import torch


def compute_class_distribution(
    dataset,
    num_classes: int,
    max_samples: Optional[int] = None,
) -> Dict[str, object]:
    n = len(dataset)
    if max_samples is not None:
        n = min(n, max_samples)

    counts = torch.zeros((num_classes,), dtype=torch.long)
    empty_images = 0

    for i in range(n):
        sample = dataset[i]
        class_ids = sample[3]
        if class_ids.numel() == 0:
            empty_images += 1
            continue
        class_ids = class_ids.long()
        class_ids = class_ids[(class_ids >= 0) & (class_ids < num_classes)]
        if class_ids.numel() == 0:
            continue
        counts += torch.bincount(class_ids, minlength=num_classes)

    total_boxes = int(counts.sum().item())
    proportions = (
        (counts.float() / max(1, total_boxes)).tolist()
        if num_classes > 0
        else []
    )

    nonzero = counts[counts > 0]
    imbalance_ratio = (
        float(nonzero.max().item() / max(1, nonzero.min().item()))
        if nonzero.numel() > 0
        else 0.0
    )

    return {
        "num_samples_checked": n,
        "empty_images": empty_images,
        "total_boxes": total_boxes,
        "counts": counts.tolist(),
        "proportions": proportions,
        "imbalance_ratio_max_over_min_nonzero": imbalance_ratio,
    }
