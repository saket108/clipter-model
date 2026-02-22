"""Standalone detector evaluation script with AP/mAP reporting."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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
    p.add_argument("--output-json", type=str, default=None)
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

    conf_thres = cfg.eval_conf_thres if args.conf_thres is None else args.conf_thres
    top_k = cfg.eval_top_k if args.top_k is None else args.top_k
    nms_iou = cfg.eval_nms_iou if args.nms_iou is None else args.nms_iou

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
    elif isinstance(raw, dict):
        state_dict = raw
        num_classes = _infer_num_classes_from_state(state_dict)
    else:
        raise ValueError("Unsupported checkpoint format.")

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

    metrics = evaluate_model_map(
        model=model,
        data_loader=dl,
        device=device,
        num_classes=num_classes,
        conf_thres=conf_thres,
        top_k=top_k,
        nms_iou=nms_iou,
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
