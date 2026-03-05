"""Grid-search detector postprocess thresholds (conf/NMS/top-k) on eval split."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.light_detr import LightDETR
from train_detect import detection_collate_fn
from utils import eval_detect
from utils.detection_metrics import evaluate_model_map

def _parse_float_list(raw: str):
    out = []
    for p in raw.split(","):
        s = p.strip()
        if not s:
            continue
        out.append(float(s))
    if len(out) == 0:
        raise ValueError("Expected at least one float value.")
    return out


def _parse_int_list(raw: str):
    out = []
    for p in raw.split(","):
        s = p.strip()
        if not s:
            continue
        out.append(int(s))
    if len(out) == 0:
        raise ValueError("Expected at least one integer value.")
    return out


def _load_checkpoint(checkpoint_path: Path, device: torch.device):
    raw = torch.load(checkpoint_path, map_location=device)
    if isinstance(raw, dict) and "model_state" in raw:
        state_dict = raw["model_state"]
        num_classes = int(raw.get("num_classes", -1))
        if num_classes <= 0:
            num_classes = eval_detect._infer_num_classes_from_state(state_dict)
        model_cfg = eval_detect._normalize_model_config(raw.get("model_config"))
    elif isinstance(raw, dict):
        state_dict = raw
        num_classes = eval_detect._infer_num_classes_from_state(state_dict)
        model_cfg = eval_detect._normalize_model_config(None)
    else:
        raise ValueError("Unsupported checkpoint format.")
    return state_dict, num_classes, model_cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--data-yaml", type=str, default=None)
    p.add_argument("--classes-path", type=str, default=None)
    p.add_argument("--train-split", type=str, default=None)
    p.add_argument("--val-split", type=str, default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--conf-grid", type=str, default="0.001,0.01,0.05,0.1,0.2,0.3")
    p.add_argument("--nms-grid", type=str, default="0.0,0.3,0.5")
    p.add_argument("--topk-grid", type=str, default="50,100")
    p.add_argument(
        "--optimize",
        type=str,
        choices=["map", "map50", "map75"],
        default="map",
        help="which metric to maximize when choosing best threshold tuple",
    )
    p.add_argument("--output-json", type=str, default=None)
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Keep eval dataset resolution logic consistent with eval_detect.py.
    if args.data_root is not None:
        eval_detect.cfg.data_root = args.data_root
    if args.data_yaml is not None:
        eval_detect.cfg.data_yaml = args.data_yaml
    if args.classes_path is not None:
        eval_detect.cfg.classes_path = args.classes_path
    if args.train_split is not None:
        eval_detect.cfg.train_split = args.train_split
    if args.val_split is not None:
        eval_detect.cfg.val_split = args.val_split
    if args.num_workers is not None:
        eval_detect.cfg.num_workers = args.num_workers

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    conf_grid = _parse_float_list(args.conf_grid)
    nms_grid = _parse_float_list(args.nms_grid)
    topk_grid = _parse_int_list(args.topk_grid)

    state_dict, num_classes, model_cfg = _load_checkpoint(checkpoint_path, device)
    if args.image_size is not None:
        eval_detect.cfg.image_size = args.image_size
    else:
        eval_detect.cfg.image_size = int(model_cfg["image_size"])

    ds, eval_split = eval_detect.build_eval_dataset()
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=eval_detect.cfg.num_workers,
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

    results = []
    best = None
    best_score = float("-inf")

    combos = list(itertools.product(conf_grid, nms_grid, topk_grid))
    print(f"Running threshold sweep over {len(combos)} combinations on split='{eval_split}'...")
    for idx, (conf_thres, nms_iou, top_k) in enumerate(combos, start=1):
        metrics = evaluate_model_map(
            model=model,
            data_loader=dl,
            device=device,
            num_classes=num_classes,
            conf_thres=float(conf_thres),
            top_k=int(top_k),
            nms_iou=float(nms_iou),
        )

        row = {
            "conf_thres": float(conf_thres),
            "nms_iou": float(nms_iou),
            "top_k": int(top_k),
            "map": float(metrics["map"]),
            "map50": float(metrics["map50"]),
            "map75": float(metrics["map75"]),
        }
        results.append(row)

        score = float(row[args.optimize])
        if score > best_score:
            best_score = score
            best = row

        print(
            f"[{idx}/{len(combos)}] conf={conf_thres:.4f} nms={nms_iou:.3f} topk={top_k} "
            f"-> mAP={row['map']:.4f} mAP50={row['map50']:.4f} mAP75={row['map75']:.4f}"
        )

    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "eval_split": eval_split,
        "num_samples": len(ds),
        "num_classes": num_classes,
        "optimize": args.optimize,
        "best": best,
        "results": results,
    }

    if best is not None:
        print(
            "Best thresholds: "
            f"conf={best['conf_thres']}, nms={best['nms_iou']}, top_k={best['top_k']} "
            f"({args.optimize}={best[args.optimize]:.4f})"
        )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"Saved sweep JSON: {out_path}")


if __name__ == "__main__":
    main()
