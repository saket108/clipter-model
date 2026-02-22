"""Set-based detection loss with Hungarian matching for DETR-style models."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / union.clamp(min=1e-6)
    return iou, union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / area.clamp(min=1e-6)


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @staticmethod
    def _greedy_assignment(cost: torch.Tensor):
        cost_cpu = cost.detach().clone().cpu()
        nq, ng = cost_cpu.shape
        n = min(nq, ng)
        if n == 0:
            return torch.empty((0,), dtype=torch.int64), torch.empty((0,), dtype=torch.int64)

        src_idx = []
        tgt_idx = []
        for _ in range(n):
            flat_idx = int(torch.argmin(cost_cpu).item())
            i = flat_idx // ng
            j = flat_idx % ng
            src_idx.append(i)
            tgt_idx.append(j)
            cost_cpu[i, :] = float("inf")
            cost_cpu[:, j] = float("inf")

        return torch.tensor(src_idx, dtype=torch.int64), torch.tensor(tgt_idx, dtype=torch.int64)

    @torch.no_grad()
    def forward(self, outputs, targets: List[dict]):
        pred_logits = outputs["pred_logits"]  # [B, Q, C+1]
        pred_boxes = outputs["pred_boxes"]  # [B, Q, 4]
        bs = pred_logits.shape[0]

        out_prob = pred_logits.softmax(-1)
        out_bbox = pred_boxes

        indices = []
        for b in range(bs):
            tgt_ids = targets[b]["labels"]
            tgt_bbox = targets[b]["boxes"]

            if tgt_ids.numel() == 0:
                empty = torch.empty((0,), dtype=torch.int64, device=pred_logits.device)
                indices.append((empty, empty))
                continue

            cost_class = -out_prob[b][:, tgt_ids]
            cost_bbox = torch.cdist(out_bbox[b], tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox[b]),
                box_cxcywh_to_xyxy(tgt_bbox),
            )

            cost = (
                self.cost_class * cost_class
                + self.cost_bbox * cost_bbox
                + self.cost_giou * cost_giou
            )

            if SCIPY_AVAILABLE:
                row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
                src_idx = torch.as_tensor(row_ind, dtype=torch.int64, device=pred_logits.device)
                tgt_idx = torch.as_tensor(col_ind, dtype=torch.int64, device=pred_logits.device)
            else:
                src_idx, tgt_idx = self._greedy_assignment(cost)
                src_idx = src_idx.to(pred_logits.device)
                tgt_idx = tgt_idx.to(pred_logits.device)

            indices.append((src_idx, tgt_idx))

        return indices


class DetectionLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher | None = None,
        cls_loss_coef: float = 1.0,
        bbox_loss_coef: float = 5.0,
        giou_loss_coef: float = 2.0,
        eos_coef: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher or HungarianMatcher()
        self.cls_loss_coef = cls_loss_coef
        self.bbox_loss_coef = bbox_loss_coef
        self.giou_loss_coef = giou_loss_coef

        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, indices):
        src_logits = outputs["pred_logits"]
        bs, num_queries, _ = src_logits.shape
        target_classes = torch.full(
            (bs, num_queries),
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            target_classes[b, src_idx] = targets[b]["labels"][tgt_idx]

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes,
            weight=self.empty_weight,
        )
        return {"loss_ce": loss_ce}

    def loss_boxes(self, outputs, targets, indices):
        src_boxes = []
        tgt_boxes = []
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            src_boxes.append(outputs["pred_boxes"][b, src_idx])
            tgt_boxes.append(targets[b]["boxes"][tgt_idx])

        if len(src_boxes) == 0:
            zero = outputs["pred_boxes"].sum() * 0.0
            return {"loss_bbox": zero, "loss_giou": zero}

        src_boxes = torch.cat(src_boxes, dim=0)
        tgt_boxes = torch.cat(tgt_boxes, dim=0)
        num_boxes = max(1, tgt_boxes.size(0))

        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction="none").sum() / num_boxes
        giou = generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(tgt_boxes),
        )
        loss_giou = (1.0 - torch.diag(giou)).sum() / num_boxes
        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def forward(self, outputs, targets: List[dict]):
        indices = self.matcher(outputs, targets)
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices))
        losses.update(self.loss_boxes(outputs, targets, indices))

        loss_total = (
            self.cls_loss_coef * losses["loss_ce"]
            + self.bbox_loss_coef * losses["loss_bbox"]
            + self.giou_loss_coef * losses["loss_giou"]
        )
        losses["loss_total"] = loss_total
        return losses
