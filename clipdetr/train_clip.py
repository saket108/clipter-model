"""Minimal training script for the CLIP-style model in `clipdetr`.
This is a runnable example using torchvision.datasets.FakeData so you can test the training loop.
Replace `DummyCaptionDataset` with your real dataset that returns (PIL image, token_ids_tensor).
"""
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, default_collate
import torchvision.transforms as T
import torchvision.datasets as tvd
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from config import Config
from models.clip_model import CLIPModel
from losses.contrastive_loss import ContrastiveLoss


cfg = Config()


class DummyCaptionDataset(Dataset):
    """Fake dataset: returns (image, token_ids) — uses `SimpleTokenizer` if available.

    - If `SimpleTokenizer` is installed/available this will encode short synthetic captions.
    - Otherwise it falls back to the previous tensor-based dummy tokens.
    """
    def __init__(self, num_samples=1024, image_size=cfg.image_size, max_len=cfg.max_text_len):
        self.ds = tvd.FakeData(size=num_samples, image_size=(3, image_size, image_size), transform=T.ToTensor())
        self.max_len = max_len

        # try to import the project's tokenizer (optional dependency)
        try:
            from tokenizer import SimpleTokenizer  # project-local

            self.tokenizer = SimpleTokenizer(max_length=self.max_len)
        except Exception:
            self.tokenizer = None

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]

        if self.tokenizer is not None:
            # synthetic caption text (deterministic short string)
            caption = f"image number {label}"
            token_ids = self.tokenizer.encode([caption])[0]
            return img, token_ids.long()

        # fallback: numeric pseudo-caption (original behavior)
        rng = (label % 100) + 1
        token_ids = torch.arange(1, min(self.max_len, 8) + 1) + (rng % 5)
        token_ids = torch.cat([token_ids, torch.zeros(self.max_len - token_ids.size(0), dtype=torch.long)])
        return img, token_ids.long()


def set_seed(s):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def count_params(module: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def clip_collate_fn(batch):
    """Collate (image, token_ids, optional variable-sized extras) safely."""
    first = batch[0]
    if isinstance(first, (list, tuple)):
        images = torch.stack([sample[0] for sample in batch], dim=0)
        token_ids = torch.stack([sample[1] for sample in batch], dim=0)
        if len(first) == 2:
            return images, token_ids
        extras = []
        for idx in range(2, len(first)):
            extras.append([sample[idx] for sample in batch])
        return (images, token_ids, *extras)

    if isinstance(first, dict):
        out = {}
        out["image"] = torch.stack([sample["image"] for sample in batch], dim=0)
        out["tokens"] = torch.stack([sample["tokens"] for sample in batch], dim=0)
        for key in first.keys():
            if key in ("image", "tokens"):
                continue
            out[key] = [sample[key] for sample in batch]
        return out

    return default_collate(batch)


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


def train(fast: bool = False, subset: int | None = None):
    """Train the CLIP backbone.

    fast: if True, run a short CPU-friendly training (fewer epochs, smaller batch, limited samples).
    subset: optional int to limit number of training samples (useful for quick local runs).
    """
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # local overrides for quick/debug runs
    epochs = 2 if fast else cfg.epochs
    batch_size = min(cfg.batch_size, 16) if fast else cfg.batch_size

    # Dataset (YOLODataset if configured)
    val_split_for_eval = None
    if cfg.data_root:
        try:
            from datasets.yolo_dataset import YOLODataset
            from tokenizer import SimpleTokenizer

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
            val_split_for_eval = _resolve_split_name(
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
                else:
                    print(f"classes_path not found at {cfg.classes_path}; trying class names from data.yaml")
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
                    f"{val_split_for_eval}.json" if val_split_for_eval else None,
                    f"annotations/{val_split_for_eval}.json" if val_split_for_eval else None,
                    f"annotations/instances_{val_split_for_eval}.json" if val_split_for_eval else None,
                    cfg.test_annotations,
                ],
            )

            tokenizer = SimpleTokenizer(max_length=cfg.max_text_len)
            dataset = YOLODataset(
                root=cfg.data_root,
                split=train_split,
                classes=classes_from_yaml if classes_file is None else None,
                classes_file=classes_file,
                image_size=cfg.image_size,
                tokenizer=tokenizer,
                annotation_format=cfg.annotation_format,
                annotations_file=train_annotations,
            )

            val_dataset = None
            if val_split_for_eval:
                val_dataset = YOLODataset(
                    root=cfg.data_root,
                    split=val_split_for_eval,
                    classes=classes_from_yaml if classes_file is None else None,
                    classes_file=classes_file,
                    image_size=cfg.image_size,
                    tokenizer=tokenizer,
                    annotation_format=cfg.annotation_format,
                    annotations_file=val_annotations,
                )

            print(
                f"Using YOLODataset splits: train='{train_split}'"
                + (f", eval='{val_split_for_eval}'" if val_split_for_eval else ", eval='none'")
            )
            print(
                f"Annotation mode: train={dataset.annotation_format}"
                + (
                    f" ({dataset.annotations_file})"
                    if dataset.annotations_file is not None
                    else ""
                )
            )
        except Exception as e:
            print("Failed to load YOLODataset — falling back to DummyCaptionDataset:", e)
            dataset = DummyCaptionDataset(num_samples=2048)
            val_dataset = None
    else:
        dataset = DummyCaptionDataset(num_samples=2048)
        val_dataset = None

    # apply subset limit for quick runs or explicit `subset` arg
    if fast and subset is None:
        subset = min(512, len(dataset))
    if subset is not None:
        from torch.utils.data import Subset
        subset = min(subset, len(dataset))
        dataset = Subset(dataset, list(range(subset)))

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=clip_collate_fn,
    )
    val_dl = (
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=clip_collate_fn,
        )
        if val_dataset is not None
        else None
    )

    # Model
    model = CLIPModel(
        embed_dim=cfg.embed_dim,
        proj_dim=cfg.proj_dim,
        max_text_len=cfg.max_text_len,
        vocab_size=cfg.vocab_size,
        image_backbone=cfg.image_backbone,
        image_pretrained=cfg.image_pretrained,
        text_num_heads=cfg.text_num_heads,
        text_num_layers=cfg.text_num_layers,
        text_ff_dim=cfg.text_ff_dim,
        text_dropout=cfg.text_dropout,
    )
    model = model.to(device)

    total_params = count_params(model)
    print(
        "Model size:"
        f" total={total_params / 1e6:.3f}M"
        f" image={count_params(model.image_encoder) / 1e6:.3f}M"
        f" text={count_params(model.text_encoder) / 1e6:.3f}M"
        f" image_proj={count_params(model.image_projection) / 1e6:.3f}M"
        f" text_proj={count_params(model.text_projection) / 1e6:.3f}M"
    )

    # Loss / optimizer / scheduler / AMP
    criterion = ContrastiveLoss(temperature=cfg.temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    scaler = GradScaler()

    # checkpoints
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        loop = tqdm(dl, desc=f"Epoch [{epoch+1}/{epochs}]")

        for batch in loop:
            # batch can be (img, tokens) or (img, tokens, boxes, cls)
            if isinstance(batch, (list, tuple)):
                imgs = batch[0]
                token_ids = batch[1]
            elif isinstance(batch, dict):
                imgs = batch['image']
                token_ids = batch['tokens']
            else:
                imgs, token_ids = batch

            imgs = imgs.to(device, non_blocking=True)
            token_ids = token_ids.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast(enabled=torch.cuda.is_available()):
                img_embeds, txt_embeds, _ = model(imgs, token_ids)
                loss = criterion(img_embeds, txt_embeds)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * imgs.size(0)
            loop.set_postfix(loss=loss.item())

        scheduler.step()

        avg_loss = total_loss / (len(dataset) if not hasattr(dataset, 'indices') else len(dataset))
        print(f"Epoch {epoch+1}/{epochs} — avg_loss: {avg_loss:.4f} — lr: {scheduler.get_last_lr()[0]:.2e}")

        # save checkpoint
        suffix = "_fast" if fast else ""
        torch.save({
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scaler_state': scaler.state_dict(),
        }, ckpt_dir / f"clip_backbone_epoch{epoch+1}{suffix}.pth")

        # run validation if available
        if val_dl is not None:
            model.eval()
            val_loss_total = 0.0
            val_n = 0
            with torch.no_grad():
                for vbatch in val_dl:
                    imgs = vbatch[0].to(device, non_blocking=True)
                    token_ids = vbatch[1].to(device, non_blocking=True)
                    with autocast(enabled=torch.cuda.is_available()):
                        img_embeds, txt_embeds, _ = model(imgs, token_ids)
                        vloss = criterion(img_embeds, txt_embeds)
                    val_loss_total += vloss.item() * imgs.size(0)
                    val_n += imgs.size(0)

            val_avg = val_loss_total / val_n
            print(f"Validation — avg_loss: {val_avg:.4f}  (split: {val_split_for_eval})")

            # save best checkpoint by validation loss
            if 'best_val_loss' not in locals():
                best_val_loss = float('inf')
            if val_avg < best_val_loss:
                best_val_loss = val_avg
                torch.save(model.state_dict(), ckpt_dir / f"clip_backbone_best_by_val{suffix}.pth")
                print("Saved new best model (by val loss)")

            model.train()

    # final save
    final_name = "clip_backbone_fast.pth" if fast else "clip_backbone.pth"
    torch.save(model.state_dict(), final_name)
    print(f"Training complete — model saved to {final_name}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--fast', action='store_true', help='run a small quick training (CPU-friendly)')
    p.add_argument('--subset', type=int, default=None, help='limit number of training samples (useful for quick local runs)')
    args = p.parse_args()
    train(fast=args.fast, subset=args.subset)
