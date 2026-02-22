"""Dataset loader for YOLO TXT labels and optional COCO-style JSON annotations.

Supported split structure:
  <root>/<split>/images/*.jpg
  <root>/<split>/labels/*.txt       # YOLO format: class x_center y_center w h

Supported JSON format:
  COCO dict with keys: images, annotations, categories

Returns:
  (image_tensor, token_ids, boxes_rel, class_ids)
Where boxes_rel are normalized cx,cy,w,h in [0, 1].
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class YOLODataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        classes: Optional[List[str]] = None,
        classes_file: Optional[str] = None,
        image_size: int = 224,
        tokenizer=None,
        transform: Optional[T.Compose] = None,
        augment: bool = False,
        annotation_format: str = "auto",
        annotations_file: Optional[str] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.augment = augment

        if classes_file is not None:
            with open(classes_file, "r", encoding="utf-8") as f:
                classes = [l.strip() for l in f.readlines() if l.strip()]
        if classes is None:
            classes = []

        self.classes = classes
        self.annotation_format = annotation_format
        self.annotations_file = self._resolve_annotations_file(annotations_file)

        if self.annotation_format not in ("auto", "yolo", "coco_json"):
            raise ValueError(
                "annotation_format must be one of ['auto', 'yolo', 'coco_json']"
            )

        if self.annotation_format == "auto":
            self.annotation_format = "coco_json" if self.annotations_file is not None else "yolo"

        self.images_dir = self.root / self.split / "images"
        self.labels_dir = self.root / self.split / "labels"

        if self.annotation_format == "yolo":
            assert self.images_dir.exists(), f"Images dir not found: {self.images_dir}"
            assert self.labels_dir.exists(), f"Labels dir not found: {self.labels_dir}"
            self.files = sorted(
                [
                    p
                    for p in self.images_dir.glob("**/*")
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png")
                ]
            )
            self.records = None
            self.num_classes = len(self.classes)
            self.class_map = {
                i: self.classes[i] if i < len(self.classes) else f"class_{i}"
                for i in range(max(1, len(self.classes)))
            }
        else:
            if self.annotations_file is None:
                raise FileNotFoundError(
                    "COCO JSON mode selected but no annotations_file could be resolved."
                )
            self.records, categories_by_id = self._load_coco_records(self.annotations_file)
            self.files = None

            all_cat_ids = sorted(categories_by_id.keys())
            self.cat_id_to_contig = {cat_id: i for i, cat_id in enumerate(all_cat_ids)}

            if len(self.classes) == 0:
                self.classes = [categories_by_id[cid] for cid in all_cat_ids]

            self.class_map = {}
            for i, cid in enumerate(all_cat_ids):
                if i < len(self.classes):
                    self.class_map[i] = self.classes[i]
                else:
                    self.class_map[i] = categories_by_id.get(cid, f"class_{cid}")

            self.num_classes = len(self.class_map)

        if transform is None:
            tfms = [T.Resize((image_size, image_size))]
            if self.augment:
                # Photometric augmentation only; boxes remain unchanged.
                tfms.extend(
                    [
                        T.ColorJitter(
                            brightness=0.2,
                            contrast=0.2,
                            saturation=0.2,
                            hue=0.02,
                        ),
                        T.RandomGrayscale(p=0.05),
                    ]
                )
            tfms.extend(
                [
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            self.transform = T.Compose(tfms)
        else:
            self.transform = transform

    def _resolve_annotations_file(self, annotations_file: Optional[str]) -> Optional[Path]:
        candidates: List[Path] = []
        if annotations_file:
            p = Path(annotations_file)
            candidates.append(p if p.is_absolute() else self.root / p)

        candidates.extend(
            [
                self.root / f"{self.split}.json",
                self.root / "annotations" / f"{self.split}.json",
                self.root / "annotations" / f"instances_{self.split}.json",
            ]
        )

        for c in candidates:
            if c.exists():
                return c
        return None

    def _resolve_image_path(self, file_name: str) -> Optional[Path]:
        p = Path(file_name)
        candidates = []
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.extend(
                [
                    self.root / p,
                    self.images_dir / p,
                    self.images_dir / p.name,
                    self.root / "images" / p,
                    self.root / "images" / p.name,
                ]
            )

        for c in candidates:
            if c.exists():
                return c
        return None

    def _load_coco_records(self, annotations_path: Path):
        with open(annotations_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Unsupported JSON schema in {annotations_path}")

        images = data.get("images", [])
        annotations = data.get("annotations", [])
        categories = data.get("categories", [])

        categories_by_id: Dict[int, str] = {}
        for cat in categories:
            if "id" not in cat:
                continue
            cat_id = int(cat["id"])
            categories_by_id[cat_id] = str(cat.get("name", f"class_{cat_id}"))

        anns_by_img = defaultdict(list)
        for ann in annotations:
            img_id = ann.get("image_id")
            bbox = ann.get("bbox")
            cat_id = ann.get("category_id")
            if img_id is None or bbox is None or cat_id is None:
                continue
            if len(bbox) < 4:
                continue
            anns_by_img[int(img_id)].append((bbox, int(cat_id)))
            if int(cat_id) not in categories_by_id:
                categories_by_id[int(cat_id)] = f"class_{int(cat_id)}"

        records = []
        missing_images = 0

        for img in images:
            img_id = img.get("id")
            file_name = img.get("file_name")
            if img_id is None or file_name is None:
                continue

            img_path = self._resolve_image_path(str(file_name))
            if img_path is None:
                missing_images += 1
                continue

            boxes_abs = []
            class_ids_raw = []
            for bbox, cat_id in anns_by_img.get(int(img_id), []):
                x, y, w, h = [float(v) for v in bbox[:4]]
                if w <= 0 or h <= 0:
                    continue
                boxes_abs.append([x, y, w, h])
                class_ids_raw.append(int(cat_id))

            if len(boxes_abs) == 0:
                boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            else:
                boxes_t = torch.tensor(boxes_abs, dtype=torch.float32)

            records.append(
                {
                    "image_path": img_path,
                    "boxes_abs": boxes_t,
                    "class_ids_raw": class_ids_raw,
                }
            )

        if missing_images > 0:
            print(
                f"Warning: {missing_images} images from {annotations_path} were not found on disk and were skipped."
            )

        if len(records) == 0:
            raise RuntimeError(f"No usable records found in JSON annotations: {annotations_path}")

        return records, categories_by_id

    def __len__(self):
        if self.annotation_format == "yolo":
            return len(self.files)
        return len(self.records)

    def _read_yolo_labels(self, label_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        boxes = []
        class_ids = []
        if not label_path.exists():
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)

        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            x_c, y_c, w, h = [float(x) for x in parts[1:5]]
            boxes.append([x_c, y_c, w, h])
            class_ids.append(cls)

        if len(boxes) == 0:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(class_ids, dtype=torch.long)

    def _read_coco_labels(self, record, image_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        img_w, img_h = image_size
        boxes_abs = record["boxes_abs"]
        raw_ids = record["class_ids_raw"]

        if boxes_abs.numel() == 0:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)

        x = boxes_abs[:, 0]
        y = boxes_abs[:, 1]
        w = boxes_abs[:, 2]
        h = boxes_abs[:, 3]

        cx = (x + 0.5 * w) / float(img_w)
        cy = (y + 0.5 * h) / float(img_h)
        ww = w / float(img_w)
        hh = h / float(img_h)
        boxes_rel = torch.stack([cx, cy, ww, hh], dim=-1).clamp(0.0, 1.0)

        class_ids = torch.tensor(
            [self.cat_id_to_contig.get(int(cid), 0) for cid in raw_ids],
            dtype=torch.long,
        )
        return boxes_rel, class_ids

    def _synthesize_caption(self, class_ids: torch.Tensor) -> str:
        if class_ids.numel() == 0:
            return "a photo"
        cls0 = int(class_ids[0].item())
        name = self.class_map.get(cls0, f"class_{cls0}")
        return f"a photo of a {name}"

    def __getitem__(self, idx: int):
        if self.annotation_format == "yolo":
            img_path = self.files[idx]
            label_path = self.labels_dir / (img_path.stem + ".txt")
            img = Image.open(img_path).convert("RGB")
            boxes_rel, class_ids = self._read_yolo_labels(label_path)
        else:
            record = self.records[idx]
            img_path = record["image_path"]
            img = Image.open(img_path).convert("RGB")
            boxes_rel, class_ids = self._read_coco_labels(record, img.size)

        img_t = self.transform(img)
        caption = self._synthesize_caption(class_ids)

        if self.tokenizer is not None:
            token_ids = self.tokenizer.encode([caption])[0]
        else:
            token_ids = torch.zeros((32,), dtype=torch.long)

        return img_t, token_ids.long(), boxes_rel, class_ids
