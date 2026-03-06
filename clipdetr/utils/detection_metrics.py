"""Detection evaluation helpers (AP/mAP) for lightweight DETR training/eval."""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence

import torch


_TILE_KEY_RE = re.compile(
    r"^(?P<base>.+)__x(?P<x>\d+)_y(?P<y>\d+)_tw(?P<tw>\d+)_th(?P<th>\d+)_ow(?P<ow>\d+)_oh(?P<oh>\d+)$"
)


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


def _parse_tiled_image_key(image_key: str) -> dict[str, Any] | None:
    stem = os.path.splitext(os.path.basename(str(image_key)))[0]
    m = _TILE_KEY_RE.match(stem)
    if m is None:
        return None
    return {
        "base": m.group("base"),
        "x": int(m.group("x")),
        "y": int(m.group("y")),
        "tw": int(m.group("tw")),
        "th": int(m.group("th")),
        "ow": int(m.group("ow")),
        "oh": int(m.group("oh")),
    }


def _dataset_image_keys(dataset: Any) -> list[str] | None:
    if dataset is None:
        return None

    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        base = dataset.dataset
        out = []
        for idx in dataset.indices:
            if hasattr(base, "get_image_path"):
                out.append(str(base.get_image_path(int(idx))))
            else:
                return None
        return out

    if hasattr(dataset, "get_image_path"):
        return [str(dataset.get_image_path(i)) for i in range(len(dataset))]
    if hasattr(dataset, "files") and getattr(dataset, "files") is not None:
        return [str(x) for x in getattr(dataset, "files")]
    if hasattr(dataset, "records") and getattr(dataset, "records") is not None:
        return [str(r["image_path"]) for r in getattr(dataset, "records")]
    return None


def _nms_xyxy(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long)
    if iou_thresh <= 0.0:
        return torch.argsort(scores, descending=True)
    try:
        from torchvision.ops import nms

        return nms(boxes, scores, float(iou_thresh))
    except Exception:
        order = torch.argsort(scores, descending=True)
        keep: list[int] = []
        while order.numel() > 0:
            i = int(order[0].item())
            keep.append(i)
            if order.numel() == 1:
                break
            rest = order[1:]
            ious = box_iou_matrix(boxes[i].unsqueeze(0), boxes[rest]).squeeze(0)
            order = rest[ious <= float(iou_thresh)]
        return torch.tensor(keep, dtype=torch.long)


def _dedupe_gt_xyxy(boxes: torch.Tensor, labels: torch.Tensor, iou_thresh: float) -> tuple[torch.Tensor, torch.Tensor]:
    if boxes.numel() == 0:
        return boxes, labels
    keep_all: list[int] = []
    for cls in labels.unique(sorted=True):
        cls_idx = torch.where(labels == cls)[0]
        if cls_idx.numel() == 0:
            continue
        cls_boxes = boxes[cls_idx]
        areas = ((cls_boxes[:, 2] - cls_boxes[:, 0]).clamp(min=0.0) * (cls_boxes[:, 3] - cls_boxes[:, 1]).clamp(min=0.0))
        order = cls_idx[torch.argsort(areas, descending=True)]
        kept: list[int] = []
        for idx in order.tolist():
            if not kept:
                kept.append(int(idx))
                continue
            ious = box_iou_matrix(boxes[idx].unsqueeze(0), boxes[torch.tensor(kept, dtype=torch.long)]).squeeze(0)
            if bool(torch.all(ious <= float(iou_thresh))):
                kept.append(int(idx))
        keep_all.extend(kept)
    if not keep_all:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)
    keep_idx = torch.tensor(sorted(set(keep_all)), dtype=torch.long)
    return boxes[keep_idx], labels[keep_idx]


def _norm_xyxy_to_abs(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    out = boxes.float().clone()
    out[:, [0, 2]] *= float(width)
    out[:, [1, 3]] *= float(height)
    return out


def _abs_xyxy_to_norm(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    out = boxes.float().clone()
    out[:, [0, 2]] /= max(1.0, float(width))
    out[:, [1, 3]] /= max(1.0, float(height))
    return out.clamp(0.0, 1.0)


def _targets_to_xyxy(targets: List[Dict[str, torch.Tensor]], target_box_format: str) -> List[Dict[str, torch.Tensor]]:
    out: List[Dict[str, torch.Tensor]] = []
    for t in targets:
        boxes = t["boxes"].float().cpu()
        if target_box_format == "cxcywh":
            boxes = box_cxcywh_to_xyxy(boxes)
        out.append({"boxes": boxes.clamp(0.0, 1.0), "labels": t["labels"].long().cpu()})
    return out


def _stitch_tiled_predictions(
    predictions: List[Dict[str, torch.Tensor]],
    targets_xyxy: List[Dict[str, torch.Tensor]],
    image_keys: list[str],
    nms_iou: float = 0.5,
    gt_dedup_iou: float = 0.9,
) -> tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]], dict[str, int]] | None:
    total = len(image_keys)
    tile_keys = 0
    groups: dict[str, dict[str, Any]] = {}

    for image_key, pred, target in zip(image_keys, predictions, targets_xyxy):
        parsed = _parse_tiled_image_key(image_key)
        if parsed is None:
            continue
        tile_keys += 1
        base = str(parsed["base"])
        group = groups.setdefault(
            base,
            {
                "ow": int(parsed["ow"]),
                "oh": int(parsed["oh"]),
                "gt_boxes": [],
                "gt_labels": [],
                "pred_boxes": [],
                "pred_scores": [],
                "pred_labels": [],
            },
        )

        gt_abs = _norm_xyxy_to_abs(target["boxes"], int(parsed["tw"]), int(parsed["th"]))
        if gt_abs.numel() > 0:
            gt_abs[:, [0, 2]] += float(parsed["x"])
            gt_abs[:, [1, 3]] += float(parsed["y"])
            group["gt_boxes"].append(gt_abs)
            group["gt_labels"].append(target["labels"].long())

        pred_abs = _norm_xyxy_to_abs(pred["boxes"], int(parsed["tw"]), int(parsed["th"]))
        if pred_abs.numel() > 0:
            pred_abs[:, [0, 2]] += float(parsed["x"])
            pred_abs[:, [1, 3]] += float(parsed["y"])
            group["pred_boxes"].append(pred_abs)
            group["pred_scores"].append(pred["scores"].float())
            group["pred_labels"].append(pred["labels"].long())

    if tile_keys == 0:
        return None
    if tile_keys < max(1, int(round(0.6 * total))):
        return None

    stitched_predictions: List[Dict[str, torch.Tensor]] = []
    stitched_targets: List[Dict[str, torch.Tensor]] = []
    for _, group in groups.items():
        ow = int(group["ow"])
        oh = int(group["oh"])

        if group["gt_boxes"]:
            gt_boxes = torch.cat(group["gt_boxes"], dim=0)
            gt_labels = torch.cat(group["gt_labels"], dim=0)
            gt_boxes, gt_labels = _dedupe_gt_xyxy(gt_boxes, gt_labels, float(gt_dedup_iou))
        else:
            gt_boxes = torch.zeros((0, 4), dtype=torch.float32)
            gt_labels = torch.zeros((0,), dtype=torch.long)

        if group["pred_boxes"]:
            pred_boxes = torch.cat(group["pred_boxes"], dim=0)
            pred_scores = torch.cat(group["pred_scores"], dim=0)
            pred_labels = torch.cat(group["pred_labels"], dim=0)
            keep_all: list[torch.Tensor] = []
            for cls in pred_labels.unique(sorted=True):
                cls_idx = torch.where(pred_labels == cls)[0]
                if cls_idx.numel() == 0:
                    continue
                keep = _nms_xyxy(pred_boxes[cls_idx], pred_scores[cls_idx], float(nms_iou))
                keep_all.append(cls_idx[keep])
            if keep_all:
                keep_idx = torch.cat(keep_all, dim=0)
                pred_boxes = pred_boxes[keep_idx]
                pred_scores = pred_scores[keep_idx]
                pred_labels = pred_labels[keep_idx]
            else:
                pred_boxes = torch.zeros((0, 4), dtype=torch.float32)
                pred_scores = torch.zeros((0,), dtype=torch.float32)
                pred_labels = torch.zeros((0,), dtype=torch.long)
        else:
            pred_boxes = torch.zeros((0, 4), dtype=torch.float32)
            pred_scores = torch.zeros((0,), dtype=torch.float32)
            pred_labels = torch.zeros((0,), dtype=torch.long)

        stitched_targets.append(
            {
                "boxes": _abs_xyxy_to_norm(gt_boxes, ow, oh),
                "labels": gt_labels,
            }
        )
        stitched_predictions.append(
            {
                "boxes": _abs_xyxy_to_norm(pred_boxes, ow, oh),
                "scores": pred_scores,
                "labels": pred_labels,
            }
        )

    return stitched_predictions, stitched_targets, {"tile_keys": int(tile_keys), "stitched_images": int(len(groups))}


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
    tile_stitch: bool = False,
    tile_stitch_nms_iou: float = 0.5,
    tile_stitch_gt_dedup_iou: float = 0.9,
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

    eval_predictions = all_predictions
    eval_targets = all_targets
    target_format = "cxcywh"
    stitch_meta = None

    if tile_stitch:
        image_keys = _dataset_image_keys(getattr(data_loader, "dataset", None))
        if image_keys is not None and len(image_keys) == len(all_predictions):
            stitched = _stitch_tiled_predictions(
                predictions=all_predictions,
                targets_xyxy=_targets_to_xyxy(all_targets, "cxcywh"),
                image_keys=image_keys,
                nms_iou=float(tile_stitch_nms_iou),
                gt_dedup_iou=float(tile_stitch_gt_dedup_iou),
            )
            if stitched is not None:
                eval_predictions, eval_targets, stitch_meta = stitched
                target_format = "xyxy"

    metrics = evaluate_detections(
        predictions=eval_predictions,
        targets=eval_targets,
        num_classes=num_classes,
        iou_thresholds=iou_thresholds,
        target_box_format=target_format,
    )
    if stitch_meta is not None:
        metrics["tile_stitch_eval"] = True
        metrics.update(stitch_meta)
    else:
        metrics["tile_stitch_eval"] = False

    if was_training:
        model.train()
    return metrics
