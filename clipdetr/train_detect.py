"""Train a lightweight DETR-style detector on YOLO or COCO-JSON annotations."""
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from config import Config
from datasets.yolo_dataset import YOLODataset
from losses.detection_loss import DetectionLoss
from models.light_detr import LightDETR


cfg = Config()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


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
        [cfg.val_split, yaml_val_split, "stratified_val_10pct", "stratified_val", "valid", "val", yaml_test_split, "test"],
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
    return train_ds, val_ds


def train(
    fast: bool = False,
    subset: int | None = None,
    clip_init: str | None = None,
):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 2 if fast else cfg.epochs
    batch_size = min(cfg.batch_size, 8) if fast else cfg.batch_size

    train_ds, val_ds = build_datasets()

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
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
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

        scheduler.step()
        avg_train = running_loss / max(1, running_items)
        print(f"Epoch {epoch + 1}/{epochs} — train_loss: {avg_train:.4f}")

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "num_classes": num_classes,
            },
            ckpt_dir / f"light_detr_epoch{epoch + 1}{'_fast' if fast else ''}.pth",
        )

        if val_dl is not None:
            model.eval()
            val_loss_sum = 0.0
            val_n = 0
            with torch.no_grad():
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

            val_avg = val_loss_sum / max(1, val_n)
            print(f"Validation — loss: {val_avg:.4f}")

            if val_avg < best_val_loss:
                best_val_loss = val_avg
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "num_classes": num_classes,
                    },
                    ckpt_dir / f"light_detr_best{'_fast' if fast else ''}.pth",
                )
                print("Saved new best detector checkpoint")

    final_name = "light_detr_fast.pth" if fast else "light_detr.pth"
    torch.save({"model_state": model.state_dict(), "num_classes": num_classes}, final_name)
    print(f"Detection training complete — saved to {final_name}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--fast", action="store_true", help="run a small quick training")
    p.add_argument("--subset", type=int, default=None, help="limit train samples")
    p.add_argument("--clip-init", type=str, default=None, help="optional CLIP checkpoint to initialize image encoder")
    args = p.parse_args()
    train(fast=args.fast, subset=args.subset, clip_init=args.clip_init)
