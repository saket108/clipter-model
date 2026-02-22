"""Run inference with the lightweight detector and save visualized predictions."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image, ImageDraw
import torchvision.transforms as T

from config import Config
from models.light_detr import LightDETR


cfg = Config()


def load_yaml(path: Path):
    if not path.exists():
        return None
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return None


def load_class_names(
    classes_path: Optional[str],
    data_root: Path,
    data_yaml: Optional[str],
    num_classes: int,
) -> List[str]:
    names: List[str] = []

    if classes_path:
        cp = Path(classes_path)
        if cp.exists():
            with open(cp, "r", encoding="utf-8") as f:
                names = [line.strip() for line in f.readlines() if line.strip()]

    if not names and data_yaml:
        yp = Path(data_yaml)
        if not yp.is_absolute():
            yp = data_root / yp
        data = load_yaml(yp)
        if isinstance(data, dict):
            raw_names = data.get("names")
            if isinstance(raw_names, list):
                names = [str(x) for x in raw_names]
            elif isinstance(raw_names, dict):
                def to_int(k):
                    try:
                        return int(k)
                    except Exception:
                        return 10**9

                names = [str(v) for _, v in sorted(raw_names.items(), key=lambda kv: to_int(kv[0]))]

    if len(names) < num_classes:
        names.extend([f"class_{i}" for i in range(len(names), num_classes)])
    return names[:num_classes]


def list_images(input_path: Path, max_images: Optional[int]) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [
        p
        for p in sorted(input_path.rglob("*"))
        if p.is_file() and p.suffix.lower() in exts
    ]
    if max_images is not None:
        images = images[:max_images]
    return images


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def infer_num_classes_from_state(state_dict: dict) -> int:
    for key, value in state_dict.items():
        if key.endswith("class_embed.weight") and value.ndim == 2:
            return int(value.shape[0] - 1)
    raise KeyError("Could not infer num_classes from checkpoint state_dict.")


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[dict, int]:
    raw = torch.load(checkpoint_path, map_location=device)
    if isinstance(raw, dict) and "model_state" in raw:
        state = raw["model_state"]
        num_classes = int(raw.get("num_classes", -1))
        if num_classes <= 0:
            num_classes = infer_num_classes_from_state(state)
        return state, num_classes

    if isinstance(raw, dict):
        state = raw
        num_classes = infer_num_classes_from_state(state)
        return state, num_classes

    raise ValueError("Unsupported checkpoint format.")


def postprocess_single(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    conf_thres: float,
    top_k: int,
    nms_iou: float,
):
    probs = pred_logits.softmax(dim=-1)
    scores, labels = probs[:, :-1].max(dim=-1)  # drop no-object class
    keep = scores >= conf_thres

    scores = scores[keep]
    labels = labels[keep]
    boxes = pred_boxes[keep]

    if scores.numel() == 0:
        return boxes, labels, scores

    boxes = cxcywh_to_xyxy(boxes).clamp(0.0, 1.0)

    if nms_iou > 0:
        from torchvision.ops import nms

        keep_nms = nms(boxes, scores, nms_iou)
        boxes = boxes[keep_nms]
        labels = labels[keep_nms]
        scores = scores[keep_nms]

    if top_k > 0 and scores.numel() > top_k:
        top_idx = torch.argsort(scores, descending=True)[:top_k]
        boxes = boxes[top_idx]
        labels = labels[top_idx]
        scores = scores[top_idx]

    return boxes, labels, scores


def draw_predictions(
    image: Image.Image,
    boxes_xyxy: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    class_names: List[str],
) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size

    for box, label, score in zip(boxes_xyxy, labels, scores):
        x1 = float(box[0].item() * w)
        y1 = float(box[1].item() * h)
        x2 = float(box[2].item() * w)
        y2 = float(box[3].item() * h)
        cls_idx = int(label.item())
        cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"class_{cls_idx}"
        text = f"{cls_name} {float(score.item()):.2f}"

        draw.rectangle((x1, y1, x2, y2), outline=(255, 50, 50), width=2)
        text_xy = (max(0.0, x1), max(0.0, y1 - 12.0))
        draw.text(text_xy, text, fill=(255, 50, 50))

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="light_detr.pth")
    p.add_argument("--input", type=str, required=True, help="Path to image file or folder")
    p.add_argument("--output-dir", type=str, default="predictions")
    p.add_argument("--conf-thres", type=float, default=0.3)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--nms-iou", type=float, default=0.0)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--data-root", type=str, default=cfg.data_root)
    p.add_argument("--data-yaml", type=str, default=cfg.data_yaml)
    p.add_argument("--classes-path", type=str, default=cfg.classes_path)
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict, num_classes = load_checkpoint(checkpoint_path, device)
    print(f"Loaded checkpoint: {checkpoint_path} (num_classes={num_classes})")

    model = LightDETR(
        num_classes=num_classes,
        hidden_dim=cfg.embed_dim,
        num_queries=cfg.det_num_queries,
        decoder_layers=cfg.det_decoder_layers,
        num_heads=cfg.det_num_heads,
        ff_dim=cfg.det_ff_dim,
        dropout=cfg.det_dropout,
        image_backbone=cfg.image_backbone,
        image_pretrained=False,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    image_size = args.image_size if args.image_size is not None else cfg.image_size
    preprocess = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    data_root = Path(args.data_root)
    class_names = load_class_names(
        classes_path=args.classes_path,
        data_root=data_root,
        data_yaml=args.data_yaml,
        num_classes=num_classes,
    )

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    image_paths = list_images(input_path, args.max_images)
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running inference on {len(image_paths)} image(s) with device={device}...")
    for i, image_path in enumerate(image_paths, start=1):
        image = Image.open(image_path).convert("RGB")
        image_t = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_t)
            pred_logits = outputs["pred_logits"][0].cpu()
            pred_boxes = outputs["pred_boxes"][0].cpu()

        boxes, labels, scores = postprocess_single(
            pred_logits=pred_logits,
            pred_boxes=pred_boxes,
            conf_thres=args.conf_thres,
            top_k=args.top_k,
            nms_iou=args.nms_iou,
        )

        vis = draw_predictions(image, boxes, labels, scores, class_names)
        out_path = output_dir / image_path.name
        vis.save(out_path)

        print(f"[{i}/{len(image_paths)}] {image_path.name}: {scores.numel()} detections -> {out_path}")

    print(f"Done. Predictions saved in: {output_dir}")


if __name__ == "__main__":
    main()
