"""Standalone detector evaluation script with AP/mAP reporting."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from datasets.yolo_dataset import YOLODataset
from models.light_detr import LightDETR
from train_detect import detection_collate_fn
from utils.detection_metrics import evaluate_model_map


cfg = Config()


def _unique_non_empty(values):
    seen = set()
    out = []
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if not s or s in seen:
            continue
        out.append(s)
        seen.add(s)
    return out


def _split_name_from_yaml_entry(entry: Any) -> str | None:
    if entry is None:
        return None
    raw = str(entry).strip()
    if not raw:
        return None
    p = Path(raw)
    if p.name.lower() == "images":
        p = p.parent
    return p.name if p.name else None


def _load_data_yaml(root: Path, yaml_path_cfg: str | None):
    if not yaml_path_cfg:
        return None
    yaml_path = Path(yaml_path_cfg)
    if not yaml_path.is_absolute():
        yaml_path = root / yaml_path
    if not yaml_path.exists():
        return None

    try:
        import yaml

        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: failed to parse data.yaml at {yaml_path}: {e}")
        return None


def _class_names_from_yaml(data_yaml: dict | None):
    if not isinstance(data_yaml, dict):
        return None
    names = data_yaml.get("names")
    if isinstance(names, list):
        return [str(x) for x in names]
    if isinstance(names, dict):
        def _to_int_key(k):
            try:
                return int(k)
            except Exception:
                return 10**9

        return [str(v) for _, v in sorted(names.items(), key=lambda kv: _to_int_key(kv[0]))]
    return None


def _resolve_split_name(root: Path, candidates):
    for split in _unique_non_empty(candidates):
        split_dir = root / split
        if (split_dir / "images").exists() and (split_dir / "labels").exists():
            return split
    return None


def _resolve_annotation_file(root: Path, candidates):
    for entry in _unique_non_empty(candidates):
        p = Path(entry)
        if not p.is_absolute():
            p = root / p
        if p.exists():
            return str(p)
    return None


def build_eval_dataset():
    data_root = Path(cfg.data_root)
    data_yaml = _load_data_yaml(data_root, cfg.data_yaml)

    yaml_val_split = _split_name_from_yaml_entry(
        (data_yaml or {}).get("val") if isinstance(data_yaml, dict) else None
    )
    yaml_test_split = _split_name_from_yaml_entry(
        (data_yaml or {}).get("test") if isinstance(data_yaml, dict) else None
    )
    eval_split = _resolve_split_name(
        data_root,
        [
            cfg.val_split,
            yaml_val_split,
            "stratified_val_10pct",
            "stratified_val",
            "valid",
            "val",
            yaml_test_split,
            "test",
        ],
    )
    if eval_split is None:
        raise FileNotFoundError("Could not resolve evaluation split from cfg/data.yaml.")

    classes_file = None
    if cfg.classes_path:
        cp = Path(cfg.classes_path)
        if cp.exists():
            classes_file = str(cp)
    classes_from_yaml = _class_names_from_yaml(data_yaml)

    eval_annotations = _resolve_annotation_file(
        data_root,
        [
            cfg.val_annotations,
            f"{eval_split}.json",
            f"annotations/{eval_split}.json",
            f"annotations/instances_{eval_split}.json",
            cfg.test_annotations,
        ],
    )

    ds = YOLODataset(
        root=cfg.data_root,
        split=eval_split,
        classes=classes_from_yaml if classes_file is None else None,
        classes_file=classes_file,
        image_size=cfg.image_size,
        tokenizer=None,
        augment=False,
        annotation_format=cfg.annotation_format,
        annotations_file=eval_annotations,
    )
    return ds, eval_split


def _infer_num_classes_from_state(state_dict: dict) -> int:
    for k, v in state_dict.items():
        if k.endswith("class_embed.weight") and v.ndim == 2:
            return int(v.shape[0] - 1)
    raise KeyError("Could not infer num_classes from checkpoint state_dict.")


def _normalize_model_config(raw_cfg: dict | None) -> Dict[str, object]:
    out: Dict[str, object] = {
        "hidden_dim": int(cfg.embed_dim),
        "num_queries": int(cfg.det_num_queries),
        "decoder_layers": int(cfg.det_decoder_layers),
        "num_heads": int(cfg.det_num_heads),
        "ff_dim": int(cfg.det_ff_dim),
        "dropout": float(cfg.det_dropout),
        "image_backbone": str(cfg.image_backbone),
        "image_size": int(cfg.image_size),
    }
    if not isinstance(raw_cfg, dict):
        return out
    if "hidden_dim" in raw_cfg:
        out["hidden_dim"] = int(raw_cfg["hidden_dim"])
    if "num_queries" in raw_cfg:
        out["num_queries"] = int(raw_cfg["num_queries"])
    if "decoder_layers" in raw_cfg:
        out["decoder_layers"] = int(raw_cfg["decoder_layers"])
    if "num_heads" in raw_cfg:
        out["num_heads"] = int(raw_cfg["num_heads"])
    if "ff_dim" in raw_cfg:
        out["ff_dim"] = int(raw_cfg["ff_dim"])
    if "dropout" in raw_cfg:
        out["dropout"] = float(raw_cfg["dropout"])
    if "image_backbone" in raw_cfg:
        out["image_backbone"] = str(raw_cfg["image_backbone"])
    if "image_size" in raw_cfg:
        out["image_size"] = int(raw_cfg["image_size"])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--conf-thres", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--nms-iou", type=float, default=None)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--data-yaml", type=str, default=None)
    p.add_argument("--classes-path", type=str, default=None)
    p.add_argument("--train-split", type=str, default=None)
    p.add_argument("--val-split", type=str, default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--output-json", type=str, default=None)
    tile_group = p.add_mutually_exclusive_group()
    tile_group.add_argument("--tile-stitch-eval", dest="tile_stitch_eval", action="store_true")
    tile_group.add_argument("--no-tile-stitch-eval", dest="tile_stitch_eval", action="store_false")
    p.set_defaults(tile_stitch_eval=None)
    p.add_argument("--tile-stitch-nms-iou", type=float, default=None)
    p.add_argument("--tile-stitch-gt-dedup-iou", type=float, default=None)
    args = p.parse_args()

    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.data_yaml is not None:
        cfg.data_yaml = args.data_yaml
    if args.classes_path is not None:
        cfg.classes_path = args.classes_path
    if args.train_split is not None:
        cfg.train_split = args.train_split
    if args.val_split is not None:
        cfg.val_split = args.val_split
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.image_size is not None:
        cfg.image_size = args.image_size

    conf_thres = cfg.eval_conf_thres if args.conf_thres is None else args.conf_thres
    top_k = cfg.eval_top_k if args.top_k is None else args.top_k
    nms_iou = cfg.eval_nms_iou if args.nms_iou is None else args.nms_iou
    tile_stitch_eval = cfg.tile_stitch_eval if args.tile_stitch_eval is None else bool(args.tile_stitch_eval)
    tile_stitch_nms_iou = (
        cfg.tile_stitch_nms_iou if args.tile_stitch_nms_iou is None else args.tile_stitch_nms_iou
    )
    tile_stitch_gt_dedup_iou = (
        cfg.tile_stitch_gt_dedup_iou
        if args.tile_stitch_gt_dedup_iou is None
        else args.tile_stitch_gt_dedup_iou
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    raw = torch.load(checkpoint_path, map_location=device)
    if isinstance(raw, dict) and "model_state" in raw:
        state_dict = raw["model_state"]
        num_classes = int(raw.get("num_classes", -1))
        if num_classes <= 0:
            num_classes = _infer_num_classes_from_state(state_dict)
        model_cfg = _normalize_model_config(raw.get("model_config"))
    elif isinstance(raw, dict):
        state_dict = raw
        num_classes = _infer_num_classes_from_state(state_dict)
        model_cfg = _normalize_model_config(None)
    else:
        raise ValueError("Unsupported checkpoint format.")

    if args.image_size is None:
        cfg.image_size = int(model_cfg["image_size"])

    ds, eval_split = build_eval_dataset()
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=detection_collate_fn,
    )

    model = LightDETR(
        num_classes=num_classes,
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_queries=int(model_cfg["num_queries"]),
        decoder_layers=int(model_cfg["decoder_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        ff_dim=int(model_cfg["ff_dim"]),
        dropout=float(model_cfg["dropout"]),
        image_backbone=str(model_cfg["image_backbone"]),
        image_pretrained=False,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)

    metrics = evaluate_model_map(
        model=model,
        data_loader=dl,
        device=device,
        num_classes=num_classes,
        conf_thres=conf_thres,
        top_k=top_k,
        nms_iou=nms_iou,
        tile_stitch=bool(tile_stitch_eval),
        tile_stitch_nms_iou=float(tile_stitch_nms_iou),
        tile_stitch_gt_dedup_iou=float(tile_stitch_gt_dedup_iou),
    )

    result = {
        "checkpoint": str(checkpoint_path.resolve()),
        "eval_split": eval_split,
        "num_samples": len(ds),
        "num_classes": num_classes,
        "metrics": metrics,
    }

    print(
        "Evaluation results: "
        f"mAP@[.50:.95]={metrics['map']:.4f} "
        f"mAP@0.50={metrics['map50']:.4f} "
        f"mAP@0.75={metrics['map75']:.4f}"
    )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        print(f"Saved eval JSON: {out_path}")


if __name__ == "__main__":
    main()
