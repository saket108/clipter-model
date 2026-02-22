"""Detection evaluation helpers (AP/mAP) for lightweight DETR training/eval."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.size(0), boxes2.size(0)), dtype=torch.float32)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area1 = (
        (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0)
        * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    )
    area2 = (
        (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0)
        * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    )
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def _ap_101_point(recall: torch.Tensor, precision: torch.Tensor) -> float:
    if recall.numel() == 0 or precision.numel() == 0:
        return 0.0

    ap = 0.0
    for thr in torch.linspace(0, 1, steps=101):
        keep = recall >= thr
        p = torch.max(precision[keep]).item() if keep.any() else 0.0
        ap += p / 101.0
    return float(ap)


def postprocess_outputs(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    conf_thres: float = 0.001,
    top_k: int = 100,
    nms_iou: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Convert raw detector outputs for one image into filtered xyxy predictions."""
    probs = pred_logits.softmax(dim=-1)
    scores, labels = probs[:, :-1].max(dim=-1)  # drop no-object class

    keep = scores >= conf_thres
    boxes = pred_boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    if boxes.numel() == 0:
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "scores": torch.zeros((0,), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.long),
        }

    boxes = box_cxcywh_to_xyxy(boxes).clamp(0.0, 1.0)

    if nms_iou > 0.0:
        try:
            from torchvision.ops import nms

            keep_idx = nms(boxes, scores, nms_iou)
            boxes = boxes[keep_idx]
            scores = scores[keep_idx]
            labels = labels[keep_idx]
        except Exception:
            # Fall back silently if torchvision NMS is unavailable.
            pass

    if top_k > 0 and scores.numel() > top_k:
        top_idx = torch.argsort(scores, descending=True)[:top_k]
        boxes = boxes[top_idx]
        scores = scores[top_idx]
        labels = labels[top_idx]

    return {"boxes": boxes, "scores": scores, "labels": labels}


def evaluate_detections(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    num_classes: int,
    iou_thresholds: Optional[Sequence[float]] = None,
    target_box_format: str = "cxcywh",
) -> Dict[str, object]:
    """Compute COCO-style mean AP over classes and IoU thresholds."""
    if iou_thresholds is None:
        iou_thresholds = [0.5 + 0.05 * i for i in range(10)]  # 0.50 ... 0.95

    if target_box_format not in ("cxcywh", "xyxy"):
        raise ValueError("target_box_format must be one of ['cxcywh', 'xyxy']")

    gt_by_img: List[Dict[str, torch.Tensor]] = []
    for t in targets:
        boxes = t["boxes"].float().cpu()
        labels = t["labels"].long().cpu()
        if target_box_format == "cxcywh":
            boxes = box_cxcywh_to_xyxy(boxes)
        gt_by_img.append({"boxes": boxes.clamp(0.0, 1.0), "labels": labels})

    pred_by_img: List[Dict[str, torch.Tensor]] = []
    for p in predictions:
        pred_by_img.append(
            {
                "boxes": p["boxes"].float().cpu().clamp(0.0, 1.0),
                "scores": p["scores"].float().cpu(),
                "labels": p["labels"].long().cpu(),
            }
        )

    aps_by_thr: Dict[float, List[float]] = {thr: [] for thr in iou_thresholds}
    ap50_per_class: Dict[int, float] = {}
    gt_counts: Dict[int, int] = {cls: 0 for cls in range(num_classes)}

    for cls in range(num_classes):
        gt_for_class: Dict[int, torch.Tensor] = {}
        num_gt = 0
        for img_idx, gt in enumerate(gt_by_img):
            keep = gt["labels"] == cls
            boxes = gt["boxes"][keep]
            gt_for_class[img_idx] = boxes
            n = int(boxes.size(0))
            num_gt += n
        gt_counts[cls] = num_gt

        preds = []
        for img_idx, pr in enumerate(pred_by_img):
            keep = pr["labels"] == cls
            boxes = pr["boxes"][keep]
            scores = pr["scores"][keep]
            for box, score in zip(boxes, scores):
                preds.append((float(score.item()), img_idx, box))
        preds.sort(key=lambda x: x[0], reverse=True)

        if num_gt == 0:
            continue

        for thr in iou_thresholds:
            matched = {
                img_idx: torch.zeros((gt_for_class[img_idx].size(0),), dtype=torch.bool)
                for img_idx in gt_for_class
            }

            tp = torch.zeros((len(preds),), dtype=torch.float32)
            fp = torch.zeros((len(preds),), dtype=torch.float32)

            for i, (_, img_idx, pred_box) in enumerate(preds):
                gt_boxes = gt_for_class[img_idx]
                if gt_boxes.numel() == 0:
                    fp[i] = 1.0
                    continue

                ious = box_iou_matrix(pred_box.unsqueeze(0), gt_boxes).squeeze(0)
                best_iou, best_idx = torch.max(ious, dim=0)

                if best_iou.item() >= thr and not matched[img_idx][best_idx]:
                    tp[i] = 1.0
                    matched[img_idx][best_idx] = True
                else:
                    fp[i] = 1.0

            tp_cum = torch.cumsum(tp, dim=0)
            fp_cum = torch.cumsum(fp, dim=0)
            recall = tp_cum / max(1, num_gt)
            precision = tp_cum / (tp_cum + fp_cum + 1e-9)
            ap = _ap_101_point(recall, precision)
            aps_by_thr[thr].append(ap)

            if abs(thr - 0.5) < 1e-9:
                ap50_per_class[cls] = ap

    ap_by_thr = {
        f"{thr:.2f}": (sum(vals) / len(vals) if len(vals) > 0 else 0.0)
        for thr, vals in aps_by_thr.items()
    }

    valid_ap_lists = [vals for vals in aps_by_thr.values() if len(vals) > 0]
    flat_aps = [ap for vals in valid_ap_lists for ap in vals]
    map_all = float(sum(flat_aps) / len(flat_aps)) if len(flat_aps) > 0 else 0.0
    map50 = ap_by_thr.get("0.50", 0.0)
    map75 = ap_by_thr.get("0.75", 0.0)

    return {
        "map": map_all,
        "map50": map50,
        "map75": map75,
        "ap_by_iou": ap_by_thr,
        "ap50_per_class": ap50_per_class,
        "gt_count_per_class": gt_counts,
    }


@torch.no_grad()
def evaluate_model_map(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    num_classes: int,
    conf_thres: float = 0.001,
    top_k: int = 100,
    nms_iou: float = 0.0,
    iou_thresholds: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    """Run model inference on a loader and return AP/mAP metrics."""
    was_training = model.training
    model.eval()

    all_predictions: List[Dict[str, torch.Tensor]] = []
    all_targets: List[Dict[str, torch.Tensor]] = []

    for images, targets in data_loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)

        batch_logits = outputs["pred_logits"].detach().cpu()
        batch_boxes = outputs["pred_boxes"].detach().cpu()
        for i in range(batch_logits.size(0)):
            all_predictions.append(
                postprocess_outputs(
                    pred_logits=batch_logits[i],
                    pred_boxes=batch_boxes[i],
                    conf_thres=conf_thres,
                    top_k=top_k,
                    nms_iou=nms_iou,
                )
            )

        for t in targets:
            all_targets.append(
                {
                    "boxes": t["boxes"].detach().cpu().float(),
                    "labels": t["labels"].detach().cpu().long(),
                }
            )

    metrics = evaluate_detections(
        predictions=all_predictions,
        targets=all_targets,
        num_classes=num_classes,
        iou_thresholds=iou_thresholds,
        target_box_format="cxcywh",
    )

    if was_training:
        model.train()
    return metrics
