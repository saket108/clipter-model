"""Run detector inference and save both visualizations and YOLO-style TXT outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as T

from config import Config
from models.light_detr import LightDETR
from predict_detect import (
    draw_predictions,
    list_images,
    load_checkpoint,
    load_class_names,
    postprocess_single,
)


cfg = Config()


def xyxy_to_cxcywh(boxes_xyxy: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = (x2 - x1).clamp(min=0.0)
    h = (y2 - y1).clamp(min=0.0)
    return torch.stack([cx, cy, w, h], dim=-1)


def save_yolo_txt(
    txt_path: Path,
    labels: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    scores: torch.Tensor,
    save_conf: bool = False,
):
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    if boxes_xyxy.numel() == 0:
        txt_path.write_text("", encoding="utf-8")
        return

    boxes_cxcywh = xyxy_to_cxcywh(boxes_xyxy).clamp(0.0, 1.0)
    lines = []
    for i in range(boxes_cxcywh.size(0)):
        cls_id = int(labels[i].item())
        cx, cy, w, h = [float(v.item()) for v in boxes_cxcywh[i]]
        if save_conf:
            conf = float(scores[i].item())
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.6f}")
        else:
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def resolve_output_paths(
    image_path: Path,
    input_path: Path,
    vis_dir: Path,
    txt_dir: Path,
):
    if input_path.is_dir():
        rel = image_path.relative_to(input_path)
        vis_path = vis_dir / rel
        txt_path = txt_dir / rel.with_suffix(".txt")
    else:
        vis_path = vis_dir / image_path.name
        txt_path = txt_dir / f"{image_path.stem}.txt"
    vis_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    return vis_path, txt_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="light_detr.pth")
    p.add_argument("--input", type=str, required=True, help="Path to image file or folder")
    p.add_argument("--vis-dir", type=str, default="predictions")
    p.add_argument("--txt-dir", type=str, default=None, help="Default: <vis-dir>/labels")
    p.add_argument("--save-conf", action="store_true", help="Append confidence to YOLO txt lines")
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

    vis_dir = Path(args.vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)
    txt_dir = Path(args.txt_dir) if args.txt_dir else (vis_dir / "labels")
    txt_dir.mkdir(parents=True, exist_ok=True)

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

        vis_path, txt_path = resolve_output_paths(
            image_path=image_path,
            input_path=input_path,
            vis_dir=vis_dir,
            txt_dir=txt_dir,
        )

        vis = draw_predictions(image, boxes, labels, scores, class_names)
        vis.save(vis_path)
        save_yolo_txt(txt_path, labels, boxes, scores, save_conf=args.save_conf)

        print(
            f"[{i}/{len(image_paths)}] {image_path.name}: {scores.numel()} detections"
            f" -> img:{vis_path} txt:{txt_path}"
        )

    print(f"Done. Visual predictions: {vis_dir}")
    print(f"Done. YOLO txt predictions: {txt_dir}")


if __name__ == "__main__":
    main()
