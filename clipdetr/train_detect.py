"""Train a lightweight DETR-style detector on YOLO or COCO-JSON annotations."""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from config import Config
from datasets.yolo_dataset import YOLODataset
from losses.detection_loss import DetectionLoss
from models.light_detr import LightDETR
from utils.dataset_stats import compute_class_distribution
from utils.detection_metrics import evaluate_model_map
from utils.experiment_logger import ExperimentLogger, make_run_id


cfg = Config()


def resolve_device(device_arg: str | None) -> torch.device:
    requested = (device_arg or "").strip().lower()

    if not requested or requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Requested --device cuda but CUDA is not available. "
                "Use --device cpu or run on a CUDA-enabled setup."
            )
        return torch.device("cuda")

    raise ValueError(f"Unsupported device option: {device_arg!r}")


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _sanitize_tag(tag: Optional[str]) -> str:
    if tag is None:
        return ""
    raw = tag.strip()
    if not raw:
        return ""
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in raw)


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


def detection_collate_fn(batch):
    images = torch.stack([sample[0] for sample in batch], dim=0)
    targets = []
    for sample in batch:
        boxes = sample[2].float()
        labels = sample[3].long()

        if boxes.numel() > 0:
            keep = (boxes[:, 2] > 0.0) & (boxes[:, 3] > 0.0)
            boxes = boxes[keep]
            labels = labels[keep]

        targets.append({"boxes": boxes, "labels": labels})
    return images, targets


def _extract_state_dict(raw_ckpt):
    if isinstance(raw_ckpt, dict) and "model_state" in raw_ckpt:
        return raw_ckpt["model_state"]
    if isinstance(raw_ckpt, dict):
        return raw_ckpt
    raise ValueError("Unsupported checkpoint format.")


def load_image_encoder_from_clip_checkpoint(
    detector_model: LightDETR,
    checkpoint_path: str,
    device: torch.device,
):
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"CLIP init checkpoint not found: {ckpt_path}")

    raw = torch.load(ckpt_path, map_location=device)
    state = _extract_state_dict(raw)

    image_state = {}
    for k, v in state.items():
        key = None
        if k.startswith("image_encoder."):
            key = k[len("image_encoder.") :]
        elif k.startswith("module.image_encoder."):
            key = k[len("module.image_encoder.") :]
        if key is not None:
            image_state[key] = v

    if len(image_state) == 0:
        raise RuntimeError(
            f"No image_encoder weights found in checkpoint: {ckpt_path}"
        )

    load_result = detector_model.image_encoder.load_state_dict(image_state, strict=False)
    loaded_count = len(image_state) - len(load_result.unexpected_keys)
    print(
        f"Initialized detector image encoder from CLIP checkpoint: {ckpt_path} "
        f"(loaded_keys={loaded_count}, missing={len(load_result.missing_keys)}, "
        f"unexpected={len(load_result.unexpected_keys)})"
    )


def set_backbone_trainable(model: LightDETR, trainable: bool):
    for p in model.image_encoder.parameters():
        p.requires_grad = trainable


def infer_num_classes(ds: Dataset, max_samples: int = 2000) -> int:
    n = min(len(ds), max_samples)
    max_cls = -1
    for i in range(n):
        _, _, _, class_ids = ds[i]
        if class_ids.numel() > 0:
            max_cls = max(max_cls, int(class_ids.max().item()))
    return max(1, max_cls + 1)


def _nonzero_class_ids(class_stats: dict) -> set[int]:
    counts = class_stats.get("counts", [])
    return {i for i, c in enumerate(counts) if int(c) > 0}


def build_datasets():
    data_root = Path(cfg.data_root)
    data_yaml = _load_data_yaml(data_root, cfg.data_yaml)

    yaml_train_split = _split_name_from_yaml_entry(
        (data_yaml or {}).get("train") if isinstance(data_yaml, dict) else None
    )
    yaml_val_split = _split_name_from_yaml_entry(
        (data_yaml or {}).get("val") if isinstance(data_yaml, dict) else None
    )
    yaml_test_split = _split_name_from_yaml_entry(
        (data_yaml or {}).get("test") if isinstance(data_yaml, dict) else None
    )

    train_split = _resolve_split_name(
        data_root,
        [cfg.train_split, yaml_train_split, "stratified_train", "train"],
    )
    val_split = _resolve_split_name(
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

    if train_split is None:
        raise FileNotFoundError(
            "Could not resolve training split. Tried configured, data.yaml, and common names "
            "['stratified_train', 'train']."
        )

    classes_file = None
    if cfg.classes_path:
        cp = Path(cfg.classes_path)
        if cp.exists():
            classes_file = str(cp)
    classes_from_yaml = _class_names_from_yaml(data_yaml)

    train_annotations = _resolve_annotation_file(
        data_root,
        [
            cfg.train_annotations,
            f"{train_split}.json" if train_split else None,
            f"annotations/{train_split}.json" if train_split else None,
            f"annotations/instances_{train_split}.json" if train_split else None,
        ],
    )
    val_annotations = _resolve_annotation_file(
        data_root,
        [
            cfg.val_annotations,
            f"{val_split}.json" if val_split else None,
            f"annotations/{val_split}.json" if val_split else None,
            f"annotations/instances_{val_split}.json" if val_split else None,
            cfg.test_annotations,
        ],
    )

    train_ds = YOLODataset(
        root=cfg.data_root,
        split=train_split,
        classes=classes_from_yaml if classes_file is None else None,
        classes_file=classes_file,
        image_size=cfg.image_size,
        tokenizer=None,
        augment=cfg.train_augment,
        annotation_format=cfg.annotation_format,
        annotations_file=train_annotations,
    )

    val_ds = None
    if val_split is not None:
        val_ds = YOLODataset(
            root=cfg.data_root,
            split=val_split,
            classes=classes_from_yaml if classes_file is None else None,
            classes_file=classes_file,
            image_size=cfg.image_size,
            tokenizer=None,
            augment=False,
            annotation_format=cfg.annotation_format,
            annotations_file=val_annotations,
        )

    print(
        f"Using detection dataset: train='{train_split}'"
        + (f", eval='{val_split}'" if val_split else ", eval='none'")
    )
    print(
        f"Annotation mode: train={train_ds.annotation_format}"
        + (f" ({train_ds.annotations_file})" if train_ds.annotations_file is not None else "")
    )
    return train_ds, val_ds, train_split, val_split


@torch.no_grad()
def evaluate_val_loss(
    model: torch.nn.Module,
    criterion: DetectionLoss,
    val_dl: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    val_loss_sum = 0.0
    val_n = 0

    for images, targets in val_dl:
        images = images.to(device, non_blocking=True)
        targets = [
            {
                "boxes": t["boxes"].to(device, non_blocking=True),
                "labels": t["labels"].to(device, non_blocking=True),
            }
            for t in targets
        ]

        outputs = model(images)
        losses = criterion(outputs, targets)
        batch_loss = losses["loss_total"].item()
        val_loss_sum += batch_loss * images.size(0)
        val_n += images.size(0)

    return val_loss_sum / max(1, val_n)


def train(
    fast: bool = False,
    subset: int | None = None,
    clip_init: str | None = None,
    experiment_tag: str | None = None,
    summary_out: str | None = None,
    device_override: str | None = None,
):
    set_seed(cfg.seed)
    requested_device = device_override if device_override is not None else cfg.device
    device = resolve_device(requested_device)
    print(f"Using device: {device}")

    epochs = 2 if fast else cfg.epochs
    batch_size = min(cfg.batch_size, 8) if fast else cfg.batch_size

    train_ds, val_ds, train_split, val_split = build_datasets()

    if fast and subset is None:
        subset = min(256, len(train_ds))
    if subset is not None:
        subset = min(subset, len(train_ds))
        train_ds = Subset(train_ds, list(range(subset)))

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=detection_collate_fn,
    )
    val_dl = (
        DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=detection_collate_fn,
        )
        if val_ds is not None
        else None
    )

    if hasattr(train_ds, "dataset"):
        base_ds = train_ds.dataset
    else:
        base_ds = train_ds

    num_classes = getattr(base_ds, "num_classes", 0)
    if num_classes <= 0:
        num_classes = infer_num_classes(base_ds)
    print(f"Detected num_classes={num_classes}")

    run_id = make_run_id(
        prefix="detect_fast" if fast else "detect",
        tag=_sanitize_tag(experiment_tag) if experiment_tag else None,
    )
    logger = ExperimentLogger(root_dir=cfg.experiments_root, run_id=run_id)
    print(f"Experiment run_id={logger.run_id}")

    config_payload = {
        "config": asdict(cfg),
        "fast": fast,
        "subset": subset,
        "clip_init_arg": clip_init,
        "train_split": train_split,
        "val_split": val_split,
        "requested_device": requested_device,
        "device": str(device),
        "resolved_epochs": epochs,
        "resolved_batch_size": batch_size,
        "num_classes": num_classes,
    }
    logger.log_config(config_payload)

    class_stats = compute_class_distribution(
        train_ds,
        num_classes=num_classes,
        max_samples=cfg.class_stats_max_samples,
    )
    class_stats_path = logger.run_dir / "class_distribution.json"
    with open(class_stats_path, "w", encoding="utf-8") as f:
        json.dump(class_stats, f, indent=2, sort_keys=True)

    counts = class_stats["counts"]
    missing_classes = [i for i, c in enumerate(counts) if int(c) == 0]
    print(
        "Train class stats: "
        f"samples_checked={class_stats['num_samples_checked']}, "
        f"total_boxes={class_stats['total_boxes']}, "
        f"empty_images={class_stats['empty_images']}, "
        f"imbalance_ratio={class_stats['imbalance_ratio_max_over_min_nonzero']:.2f}"
    )
    if len(missing_classes) > 0:
        print(f"Warning: classes with zero boxes in train set: {missing_classes}")

    val_class_stats_path = None
    if val_ds is not None:
        consistency_train_stats = class_stats
        if cfg.strict_class_check and train_ds is not base_ds:
            consistency_train_stats = compute_class_distribution(
                base_ds,
                num_classes=num_classes,
                max_samples=cfg.class_stats_max_samples,
            )

        val_class_stats = compute_class_distribution(
            val_ds,
            num_classes=num_classes,
            max_samples=cfg.class_stats_max_samples,
        )
        val_class_stats_path = logger.run_dir / "val_class_distribution.json"
        with open(val_class_stats_path, "w", encoding="utf-8") as f:
            json.dump(val_class_stats, f, indent=2, sort_keys=True)

        print(
            "Val class stats: "
            f"samples_checked={val_class_stats['num_samples_checked']}, "
            f"total_boxes={val_class_stats['total_boxes']}, "
            f"empty_images={val_class_stats['empty_images']}, "
            f"imbalance_ratio={val_class_stats['imbalance_ratio_max_over_min_nonzero']:.2f}"
        )

        train_present = _nonzero_class_ids(consistency_train_stats)
        val_present = _nonzero_class_ids(val_class_stats)
        val_only = sorted(val_present - train_present)
        if len(val_only) > 0:
            msg = (
                "Validation has classes with boxes that are missing in training: "
                f"{val_only}. This usually indicates split/class-map mismatch."
            )
            if cfg.strict_class_check:
                raise RuntimeError(msg)
            print(f"Warning: {msg}")

    model = LightDETR(
        num_classes=num_classes,
        hidden_dim=cfg.embed_dim,
        num_queries=cfg.det_num_queries,
        decoder_layers=cfg.det_decoder_layers,
        num_heads=cfg.det_num_heads,
        ff_dim=cfg.det_ff_dim,
        dropout=cfg.det_dropout,
        image_backbone=cfg.image_backbone,
        image_pretrained=cfg.image_pretrained,
    ).to(device)

    clip_init_path = clip_init if clip_init is not None else cfg.clip_init_checkpoint
    if clip_init_path:
        load_image_encoder_from_clip_checkpoint(
            detector_model=model,
            checkpoint_path=clip_init_path,
            device=device,
        )

    freeze_epochs = max(0, int(cfg.freeze_backbone_epochs))
    if clip_init_path and freeze_epochs > 0:
        set_backbone_trainable(model, trainable=False)
        print(f"Backbone frozen for first {freeze_epochs} epoch(s).")

    print(f"Detector params: total={count_params(model) / 1e6:.3f}M")

    criterion = DetectionLoss(
        num_classes=num_classes,
        cls_loss_coef=cfg.det_cls_loss_coef,
        bbox_loss_coef=cfg.det_bbox_loss_coef,
        giou_loss_coef=cfg.det_giou_loss_coef,
        eos_coef=cfg.det_eos_coef,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_fast" if fast else ""
    tag_safe = _sanitize_tag(experiment_tag)
    if tag_safe:
        suffix = f"{suffix}_{tag_safe}" if suffix else f"_{tag_safe}"

    best_val_loss = float("inf")
    best_map = -1.0
    best_loss_epoch = None
    best_map_epoch = None
    best_loss_path = None
    best_map_path = None

    try:
        for epoch in range(epochs):
            if clip_init_path and freeze_epochs > 0 and epoch == freeze_epochs:
                set_backbone_trainable(model, trainable=True)
                print("Backbone unfrozen; full detector is now trainable.")

            model.train()
            running_loss = 0.0
            running_items = 0
            loop = tqdm(train_dl, desc=f"Detect Epoch [{epoch + 1}/{epochs}]")

            for images, targets in loop:
                images = images.to(device, non_blocking=True)
                targets = [
                    {
                        "boxes": t["boxes"].to(device, non_blocking=True),
                        "labels": t["labels"].to(device, non_blocking=True),
                    }
                    for t in targets
                ]

                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                losses = criterion(outputs, targets)
                loss = losses["loss_total"]
                loss.backward()
                optimizer.step()

                bs = images.size(0)
                running_loss += loss.item() * bs
                running_items += bs
                loop.set_postfix(
                    loss=f"{loss.item():.4f}",
                    ce=f"{losses['loss_ce'].item():.4f}",
                    bbox=f"{losses['loss_bbox'].item():.4f}",
                    giou=f"{losses['loss_giou'].item():.4f}",
                )

            avg_train = running_loss / max(1, running_items)
            current_lr = float(optimizer.param_groups[0]["lr"])
            print(f"Epoch {epoch + 1}/{epochs} — train_loss: {avg_train:.4f} — lr: {current_lr:.2e}")

            val_avg = None
            map_metrics = None
            map_val = None
            map50 = None
            map75 = None

            if val_dl is not None:
                val_avg = evaluate_val_loss(model, criterion, val_dl, device)
                print(f"Validation — loss: {val_avg:.4f}")

                map_metrics = evaluate_model_map(
                    model=model,
                    data_loader=val_dl,
                    device=device,
                    num_classes=num_classes,
                    conf_thres=cfg.eval_conf_thres,
                    top_k=cfg.eval_top_k,
                    nms_iou=cfg.eval_nms_iou,
                )
                map_val = float(map_metrics["map"])
                map50 = float(map_metrics["map50"])
                map75 = float(map_metrics["map75"])
                print(
                    "Validation — "
                    f"mAP@[.50:.95]: {map_val:.4f} | "
                    f"mAP@0.50: {map50:.4f} | "
                    f"mAP@0.75: {map75:.4f}"
                )

                if val_avg < best_val_loss:
                    best_val_loss = val_avg
                    best_loss_epoch = epoch + 1
                    best_loss_path = ckpt_dir / f"light_detr_best_loss{suffix}.pth"
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "num_classes": num_classes,
                            "epoch": epoch + 1,
                            "val_loss": val_avg,
                        },
                        best_loss_path,
                    )
                    print("Saved new best detector checkpoint by loss")

                if map_val > best_map:
                    best_map = map_val
                    best_map_epoch = epoch + 1
                    best_map_path = ckpt_dir / f"light_detr_best_map{suffix}.pth"
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "num_classes": num_classes,
                            "epoch": epoch + 1,
                            "map": map_val,
                            "map50": map50,
                            "map75": map75,
                            "metrics": map_metrics,
                        },
                        best_map_path,
                    )
                    print("Saved new best detector checkpoint by mAP")

            epoch_ckpt_path = ckpt_dir / f"light_detr_epoch{epoch + 1}{suffix}.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "num_classes": num_classes,
                    "train_loss": avg_train,
                    "val_loss": val_avg,
                    "map": map_val,
                    "map50": map50,
                    "map75": map75,
                },
                epoch_ckpt_path,
            )

            logger.log_epoch(
                {
                    "epoch": epoch + 1,
                    "lr": current_lr,
                    "train_loss": avg_train,
                    "val_loss": val_avg,
                    "map": map_val,
                    "map50": map50,
                    "map75": map75,
                    "best_val_loss_so_far": best_val_loss if best_val_loss < float("inf") else None,
                    "best_map_so_far": best_map if best_map >= 0 else None,
                }
            )

            scheduler.step()

        final_name = f"light_detr{suffix}.pth"
        torch.save({"model_state": model.state_dict(), "num_classes": num_classes}, final_name)
        print(f"Detection training complete — saved to {final_name}")

        summary = {
            "run_id": logger.run_id,
            "final_checkpoint": str(Path(final_name).resolve()),
            "best_loss_checkpoint": str(best_loss_path.resolve()) if best_loss_path else None,
            "best_map_checkpoint": str(best_map_path.resolve()) if best_map_path else None,
            "best_val_loss": best_val_loss if best_val_loss < float("inf") else None,
            "best_val_loss_epoch": best_loss_epoch,
            "best_map": best_map if best_map >= 0 else None,
            "best_map_epoch": best_map_epoch,
            "num_classes": num_classes,
            "fast": fast,
            "subset": subset,
            "clip_init_used": clip_init_path,
            "train_split": train_split,
            "val_split": val_split,
            "class_distribution_path": str(class_stats_path.resolve()),
            "val_class_distribution_path": (
                str(val_class_stats_path.resolve()) if val_class_stats_path is not None else None
            ),
            "strict_class_check": bool(cfg.strict_class_check),
        }
        summary_path = logger.write_summary(summary, summary_out=summary_out)
        print(f"Run summary saved to: {summary_path}")
        return summary
    finally:
        logger.close()


def apply_cli_overrides(args):
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
    if args.image_size is not None:
        cfg.image_size = args.image_size
    if args.image_backbone is not None:
        cfg.image_backbone = args.image_backbone
    if args.embed_dim is not None:
        cfg.embed_dim = args.embed_dim
    if args.det_dropout is not None:
        cfg.det_dropout = args.det_dropout
    if args.image_pretrained:
        cfg.image_pretrained = True
    elif args.no_image_pretrained:
        cfg.image_pretrained = False

    if args.seed is not None:
        cfg.seed = args.seed
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers

    if args.num_queries is not None:
        cfg.det_num_queries = args.num_queries
    if args.decoder_layers is not None:
        cfg.det_decoder_layers = args.decoder_layers
    if args.num_heads is not None:
        cfg.det_num_heads = args.num_heads
    if args.ff_dim is not None:
        cfg.det_ff_dim = args.ff_dim
    if args.freeze_backbone_epochs is not None:
        cfg.freeze_backbone_epochs = args.freeze_backbone_epochs

    if args.eval_conf_thres is not None:
        cfg.eval_conf_thres = args.eval_conf_thres
    if args.eval_top_k is not None:
        cfg.eval_top_k = args.eval_top_k
    if args.eval_nms_iou is not None:
        cfg.eval_nms_iou = args.eval_nms_iou

    if args.experiments_root is not None:
        cfg.experiments_root = args.experiments_root
    if args.class_stats_max_samples is not None:
        cfg.class_stats_max_samples = args.class_stats_max_samples
    if args.device is not None:
        cfg.device = args.device
    if args.strict_class_check:
        cfg.strict_class_check = True
    if args.no_strict_class_check:
        cfg.strict_class_check = False

    if args.no_train_augment:
        cfg.train_augment = False
    elif args.train_augment:
        cfg.train_augment = True


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fast", action="store_true", help="run a small quick training")
    p.add_argument("--subset", type=int, default=None, help="limit train samples")
    p.add_argument("--clip-init", type=str, default=None, help="optional CLIP checkpoint to initialize image encoder")
    p.add_argument("--tag", type=str, default=None, help="optional experiment tag for run/checkpoint naming")
    p.add_argument("--summary-out", type=str, default=None, help="optional JSON path for writing run summary")

    p.add_argument("--data-root", type=str, default=None, help="override cfg.data_root")
    p.add_argument("--data-yaml", type=str, default=None, help="override cfg.data_yaml")
    p.add_argument("--classes-path", type=str, default=None, help="override cfg.classes_path")
    p.add_argument("--train-split", type=str, default=None, help="override cfg.train_split")
    p.add_argument("--val-split", type=str, default=None, help="override cfg.val_split")
    p.add_argument("--image-size", type=int, default=None, help="override cfg.image_size")
    p.add_argument(
        "--image-backbone",
        type=str,
        choices=["mobilenet_v3_small", "convnext_tiny"],
        default=None,
        help="override cfg.image_backbone",
    )
    p.add_argument("--embed-dim", type=int, default=None, help="override cfg.embed_dim")
    p.add_argument("--image-pretrained", action="store_true", help="enable pretrained image backbone")
    p.add_argument("--no-image-pretrained", action="store_true", help="disable pretrained image backbone")

    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=None)

    p.add_argument("--num-queries", type=int, default=None, help="override det_num_queries")
    p.add_argument("--decoder-layers", type=int, default=None, help="override det_decoder_layers")
    p.add_argument("--num-heads", type=int, default=None, help="override det_num_heads")
    p.add_argument("--ff-dim", type=int, default=None, help="override det_ff_dim")
    p.add_argument("--det-dropout", type=float, default=None, help="override det_dropout")
    p.add_argument("--freeze-backbone-epochs", type=int, default=None)

    p.add_argument("--eval-conf-thres", type=float, default=None)
    p.add_argument("--eval-top-k", type=int, default=None)
    p.add_argument("--eval-nms-iou", type=float, default=None)

    p.add_argument("--experiments-root", type=str, default=None)
    p.add_argument("--class-stats-max-samples", type=int, default=None)
    p.add_argument("--strict-class-check", action="store_true", help="fail if validation contains classes missing in train")
    p.add_argument("--no-strict-class-check", action="store_true", help="disable strict train/val class consistency check")
    p.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="force training device (default: auto-detect)",
    )
    p.add_argument("--train-augment", action="store_true", help="force-enable train augmentation")
    p.add_argument("--no-train-augment", action="store_true", help="disable train augmentation")

    args = p.parse_args()
    apply_cli_overrides(args)
    train(
        fast=args.fast,
        subset=args.subset,
        clip_init=args.clip_init,
        experiment_tag=args.tag,
        summary_out=args.summary_out,
        device_override=args.device,
    )
