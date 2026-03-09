"""Dataset loader for YOLO TXT labels and JSON annotations.

Supported split structure:
  <root>/<split>/images/*.jpg
  <root>/<split>/labels/*.txt       # YOLO format: class x_center y_center w h

Supported JSON formats:
  - COCO dict with keys: images, annotations, categories
  - Structured per-image JSON with keys:
      {
        "images": [
          {
            "file_name": "...",
            "annotations": [
              {
                "category_name": "...",
                "bounding_box_normalized": {
                  "x_center": ...,
                  "y_center": ...,
                  "width": ...,
                  "height": ...
                },
                "risk_assessment": {"severity_level": "..."},
                "zone_estimation": "...",
                "description": "..."
              }
            ]
          }
        ]
      }

Returns:
  (image_tensor, token_ids, boxes_rel, class_ids)
Where boxes_rel are normalized cx,cy,w,h in [0, 1].
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
        caption_style: str = "auto",
        max_caption_annotations: int = 4,
    ):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.augment = augment
        self.caption_style = str(caption_style or "auto").strip().lower()
        self.max_caption_annotations = max(1, int(max_caption_annotations))
        self.images_dir = self.root / self.split / "images"
        self.labels_dir = self.root / self.split / "labels"

        if classes_file is not None:
            with open(classes_file, "r", encoding="utf-8") as f:
                classes = [l.strip() for l in f.readlines() if l.strip()]
        if classes is None:
            classes = []

        self.classes = classes
        self.annotation_format = str(annotation_format or "auto").strip().lower()
        self.annotations_file = self._resolve_annotations_file(annotations_file)

        if self.annotation_format not in ("auto", "yolo", "coco_json", "structured_json"):
            raise ValueError(
                "annotation_format must be one of "
                "['auto', 'yolo', 'coco_json', 'structured_json']"
            )
        if self.caption_style not in (
            "auto",
            "synthetic",
            "structured",
            "first_sentence",
            "raw_description",
        ):
            raise ValueError(
                "caption_style must be one of "
                "['auto', 'synthetic', 'structured', 'first_sentence', 'raw_description']"
            )

        if self.annotation_format == "auto":
            self.annotation_format = self._resolve_auto_annotation_format()

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
        elif self.annotation_format == "coco_json":
            if self.annotations_file is None:
                raise FileNotFoundError(
                    "COCO JSON mode selected but no annotations_file could be resolved."
                )
            self.records, self.class_map, self.num_classes = self._load_coco_records(
                self.annotations_file
            )
            self.files = None
        else:
            if self.annotations_file is None:
                raise FileNotFoundError(
                    "Structured JSON mode selected but no annotations_file could be resolved."
                )
            self.records, self.class_map, self.num_classes = self._load_structured_records(
                self.annotations_file
            )
            self.files = None

        self._use_default_transform = transform is None
        if transform is None:
            self.transform = T.Compose(
                [
                    T.Resize((image_size, image_size)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            self.photometric_aug = (
                T.Compose(
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
                if self.augment
                else None
            )
        else:
            self.transform = transform
            self.photometric_aug = None

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

    def _resolve_auto_annotation_format(self) -> str:
        has_yolo_dirs = self.images_dir.exists() and self.labels_dir.exists()
        detected_json_schema = self._detect_annotation_schema(self.annotations_file)

        if self.tokenizer is not None and detected_json_schema is not None:
            return detected_json_schema
        if has_yolo_dirs:
            return "yolo"
        if detected_json_schema is not None:
            return detected_json_schema
        return "yolo"

    @staticmethod
    def _detect_annotation_schema(annotations_file: Optional[Path]) -> Optional[str]:
        if annotations_file is None or not annotations_file.exists():
            return None
        try:
            with open(annotations_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None

        if not isinstance(data, dict):
            return None

        if isinstance(data.get("images"), list) and isinstance(data.get("annotations"), list):
            return "coco_json"

        images = data.get("images")
        if isinstance(images, list) and any(
            isinstance(item, dict) and isinstance(item.get("annotations"), list)
            for item in images[: min(len(images), 8)]
        ):
            return "structured_json"

        return None

    def _resolve_image_path(self, file_name: str, split_name: Optional[str] = None) -> Optional[Path]:
        p = Path(file_name)
        candidates: List[Path] = []

        if p.is_absolute():
            candidates.append(p)
        else:
            split_token = str(split_name).strip() if split_name is not None else ""
            if split_token:
                split_images = self.root / split_token / "images"
                candidates.extend([split_images / p, split_images / p.name])
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

    def _build_category_mapping(
        self,
        discovered_categories: Dict[int, str],
    ) -> Tuple[Dict[int, int], Dict[int, str], int]:
        if self.classes:
            name_to_idx = {name: idx for idx, name in enumerate(self.classes)}
            discovered_names = {name for name in discovered_categories.values() if name}
            if discovered_names.issubset(name_to_idx):
                raw_to_contig = {
                    raw_id: name_to_idx[name]
                    for raw_id, name in discovered_categories.items()
                }
                class_map = {idx: name for idx, name in enumerate(self.classes)}
                return raw_to_contig, class_map, len(self.classes)
            print(
                "Warning: provided class names do not fully cover JSON categories; "
                "falling back to categories discovered from annotations."
            )

        ordered_raw_ids = sorted(discovered_categories.keys())
        raw_to_contig = {raw_id: idx for idx, raw_id in enumerate(ordered_raw_ids)}
        class_map = {
            idx: discovered_categories[raw_id]
            for idx, raw_id in enumerate(ordered_raw_ids)
        }
        return raw_to_contig, class_map, len(class_map)

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

        raw_to_contig, class_map, num_classes = self._build_category_mapping(categories_by_id)
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
            class_ids = []
            for bbox, cat_id in anns_by_img.get(int(img_id), []):
                x, y, w, h = [float(v) for v in bbox[:4]]
                if w <= 0 or h <= 0:
                    continue
                boxes_abs.append([x, y, w, h])
                class_ids.append(raw_to_contig.get(int(cat_id), 0))

            boxes_t = (
                torch.tensor(boxes_abs, dtype=torch.float32)
                if boxes_abs
                else torch.zeros((0, 4), dtype=torch.float32)
            )
            class_ids_t = (
                torch.tensor(class_ids, dtype=torch.long)
                if class_ids
                else torch.zeros((0,), dtype=torch.long)
            )

            records.append(
                {
                    "image_path": img_path,
                    "boxes_abs": boxes_t,
                    "class_ids": class_ids_t,
                    "text_entries": [],
                }
            )

        if missing_images > 0:
            print(
                f"Warning: {missing_images} images from {annotations_path} "
                "were not found on disk and were skipped."
            )

        if len(records) == 0:
            raise RuntimeError(f"No usable records found in JSON annotations: {annotations_path}")

        return records, class_map, num_classes

    @staticmethod
    def _extract_structured_box(annotation: Dict[str, Any]) -> Optional[List[float]]:
        box = annotation.get("bounding_box_normalized")
        if isinstance(box, dict):
            x_center = box.get("x_center", box.get("cx"))
            y_center = box.get("y_center", box.get("cy"))
            width = box.get("width", box.get("w"))
            height = box.get("height", box.get("h"))
            values = [x_center, y_center, width, height]
        else:
            box = annotation.get("bbox_normalized", annotation.get("bbox"))
            if not isinstance(box, (list, tuple)) or len(box) < 4:
                return None
            values = list(box[:4])

        try:
            x_center, y_center, width, height = [float(v) for v in values]
        except Exception:
            return None

        if width <= 0.0 or height <= 0.0:
            return None

        return [
            max(0.0, min(1.0, x_center)),
            max(0.0, min(1.0, y_center)),
            max(0.0, min(1.0, width)),
            max(0.0, min(1.0, height)),
        ]

    def _load_structured_records(self, annotations_path: Path):
        with open(annotations_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Unsupported JSON schema in {annotations_path}")

        images = data.get("images", [])
        if not isinstance(images, list):
            raise ValueError(f"Structured JSON missing 'images' list: {annotations_path}")

        generated_ids: Dict[str, int] = {}
        next_generated_id = 0
        discovered_categories: Dict[int, str] = {}

        def resolve_category(annotation: Dict[str, Any]) -> Tuple[int, str]:
            nonlocal next_generated_id
            raw_name = str(annotation.get("category_name", "")).strip()
            raw_id = annotation.get("category_id")

            if raw_id is not None:
                raw_cat_id = int(raw_id)
            else:
                key = raw_name or f"class_{next_generated_id}"
                if key not in generated_ids:
                    generated_ids[key] = next_generated_id
                    next_generated_id += 1
                raw_cat_id = generated_ids[key]

            raw_cat_name = raw_name or discovered_categories.get(raw_cat_id, f"class_{raw_cat_id}")
            discovered_categories.setdefault(raw_cat_id, raw_cat_name)
            return raw_cat_id, raw_cat_name

        for image in images:
            annotations = image.get("annotations", [])
            if not isinstance(annotations, list):
                continue
            for annotation in annotations:
                if not isinstance(annotation, dict):
                    continue
                resolve_category(annotation)

        raw_to_contig, class_map, num_classes = self._build_category_mapping(discovered_categories)
        records = []
        missing_images = 0

        for image in images:
            if not isinstance(image, dict):
                continue
            file_name = image.get("file_name")
            if not file_name:
                continue

            image_split = image.get("split", self.split)
            image_path = self._resolve_image_path(str(file_name), split_name=str(image_split))
            if image_path is None:
                missing_images += 1
                continue

            boxes_rel: List[List[float]] = []
            class_ids: List[int] = []
            text_entries: List[Dict[str, str]] = []

            annotations = image.get("annotations", [])
            if not isinstance(annotations, list):
                annotations = []

            for annotation in annotations:
                if not isinstance(annotation, dict):
                    continue

                raw_cat_id, raw_cat_name = resolve_category(annotation)
                box_rel = self._extract_structured_box(annotation)
                if box_rel is None:
                    continue

                boxes_rel.append(box_rel)
                class_ids.append(raw_to_contig.get(raw_cat_id, 0))
                text_entries.append(
                    {
                        "category_name": raw_cat_name,
                        "severity": str(
                            (annotation.get("risk_assessment") or {}).get("severity_level", "")
                        ).strip(),
                        "zone": str(annotation.get("zone_estimation", "")).strip(),
                        "description": str(annotation.get("description", "")).strip(),
                    }
                )

            records.append(
                {
                    "image_path": image_path,
                    "boxes_rel": (
                        torch.tensor(boxes_rel, dtype=torch.float32)
                        if boxes_rel
                        else torch.zeros((0, 4), dtype=torch.float32)
                    ),
                    "class_ids": (
                        torch.tensor(class_ids, dtype=torch.long)
                        if class_ids
                        else torch.zeros((0,), dtype=torch.long)
                    ),
                    "text_entries": text_entries,
                }
            )

        if missing_images > 0:
            print(
                f"Warning: {missing_images} images from {annotations_path} "
                "were not found on disk and were skipped."
            )

        if len(records) == 0:
            raise RuntimeError(f"No usable records found in JSON annotations: {annotations_path}")

        return records, class_map, num_classes

    def __len__(self):
        if self.annotation_format == "yolo":
            return len(self.files)
        return len(self.records)

    def get_image_path(self, idx: int) -> Path:
        if self.annotation_format == "yolo":
            return self.files[idx]
        return self.records[idx]["image_path"]

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
        class_ids = record["class_ids"]

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
        return boxes_rel, class_ids

    @staticmethod
    def _format_damage_name(name: str) -> str:
        return re.sub(r"\s+", " ", str(name).replace("-", " ").strip()).lower()

    @staticmethod
    def _normalize_caption_text(text: str) -> str:
        compact = re.sub(r"\s+", " ", str(text).strip())
        compact = compact.rstrip(" .")
        return compact.lower()

    def _structured_caption_from_entry(self, entry: Dict[str, str]) -> str:
        damage = self._format_damage_name(entry.get("category_name", "damage"))
        severity = self._normalize_caption_text(entry.get("severity", "")) if entry.get("severity") else ""
        zone = self._normalize_caption_text(entry.get("zone", "")) if entry.get("zone") else ""

        parts = [part for part in (severity, damage) if part]
        damage_phrase = " ".join(parts).strip() or damage or "damage"
        if zone:
            return f"{damage_phrase} detected in the {zone} structural region"
        return f"{damage_phrase} detected on aircraft structure"

    def _entry_to_caption(self, entry: Dict[str, str]) -> str:
        description = str(entry.get("description", "")).strip()

        if self.caption_style == "synthetic":
            return ""
        if self.caption_style == "raw_description":
            if description:
                return self._normalize_caption_text(description)
            return self._structured_caption_from_entry(entry)
        if self.caption_style in ("auto", "first_sentence"):
            if description:
                first_sentence = description.split(".", 1)[0]
                first_sentence = self._normalize_caption_text(first_sentence)
                if first_sentence:
                    return first_sentence
            return self._structured_caption_from_entry(entry)
        if self.caption_style == "structured":
            return self._structured_caption_from_entry(entry)
        return self._structured_caption_from_entry(entry)

    def _synthesize_caption(self, class_ids: torch.Tensor) -> str:
        if class_ids.numel() == 0:
            return "a photo of aircraft surface"

        names = []
        seen = set()
        for cls_id in class_ids.tolist():
            name = self._format_damage_name(self.class_map.get(int(cls_id), f"class_{cls_id}"))
            if name and name not in seen:
                seen.add(name)
                names.append(name)

        if len(names) == 0:
            return "a photo of aircraft surface"
        if len(names) == 1:
            return f"a photo of a {names[0]}"
        if len(names) == 2:
            return f"a photo of {names[0]} and {names[1]}"
        return f"a photo of {', '.join(names[:-1])}, and {names[-1]}"

    def _compose_caption(self, record: Optional[Dict[str, Any]], class_ids: torch.Tensor) -> str:
        if record is None or self.caption_style == "synthetic":
            return self._synthesize_caption(class_ids)

        text_entries = record.get("text_entries", [])
        if not text_entries:
            return self._synthesize_caption(class_ids)

        captions: List[str] = []
        seen = set()
        for entry in text_entries:
            caption = self._entry_to_caption(entry)
            if not caption or caption in seen:
                continue
            seen.add(caption)
            captions.append(caption)

        if not captions:
            return self._synthesize_caption(class_ids)

        return "; ".join(captions[: self.max_caption_annotations])

    @staticmethod
    def _boxes_cxcywh_to_xyxy_abs(boxes_rel: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
        if boxes_rel.numel() == 0:
            return torch.zeros((0, 4), dtype=torch.float32)
        boxes = boxes_rel.float()
        cx = boxes[:, 0] * float(img_w)
        cy = boxes[:, 1] * float(img_h)
        bw = boxes[:, 2] * float(img_w)
        bh = boxes[:, 3] * float(img_h)
        x1 = cx - 0.5 * bw
        y1 = cy - 0.5 * bh
        x2 = cx + 0.5 * bw
        y2 = cy + 0.5 * bh
        return torch.stack([x1, y1, x2, y2], dim=-1)

    @staticmethod
    def _boxes_xyxy_abs_to_cxcywh_rel(boxes_abs: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
        if boxes_abs.numel() == 0:
            return torch.zeros((0, 4), dtype=torch.float32)
        boxes = boxes_abs.float()
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        w = (x2 - x1).clamp(min=0.0)
        h = (y2 - y1).clamp(min=0.0)
        cx = x1 + 0.5 * w
        cy = y1 + 0.5 * h
        return torch.stack(
            [
                cx / float(img_w),
                cy / float(img_h),
                w / float(img_w),
                h / float(img_h),
            ],
            dim=-1,
        ).clamp(0.0, 1.0)

    def _random_horizontal_flip(self, img: Image.Image, boxes_rel: torch.Tensor, p: float = 0.5):
        if torch.rand(1).item() >= p:
            return img, boxes_rel
        out_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if boxes_rel.numel() == 0:
            return out_img, boxes_rel
        out_boxes = boxes_rel.clone()
        out_boxes[:, 0] = 1.0 - out_boxes[:, 0]
        return out_img, out_boxes

    def _random_resized_crop(
        self,
        img: Image.Image,
        boxes_rel: torch.Tensor,
        class_ids: torch.Tensor,
        p: float = 0.5,
        min_scale: float = 0.7,
    ):
        if boxes_rel.numel() == 0 or torch.rand(1).item() >= p:
            return img, boxes_rel, class_ids

        img_w, img_h = img.size
        scale = float(torch.empty(1).uniform_(min_scale, 1.0).item())
        crop_w = max(2, int(round(img_w * scale)))
        crop_h = max(2, int(round(img_h * scale)))
        if crop_w >= img_w or crop_h >= img_h:
            return img, boxes_rel, class_ids

        left = int(torch.randint(0, img_w - crop_w + 1, (1,)).item())
        top = int(torch.randint(0, img_h - crop_h + 1, (1,)).item())

        boxes_abs = self._boxes_cxcywh_to_xyxy_abs(boxes_rel, img_w, img_h)
        boxes_abs[:, [0, 2]] = boxes_abs[:, [0, 2]] - float(left)
        boxes_abs[:, [1, 3]] = boxes_abs[:, [1, 3]] - float(top)
        boxes_abs[:, [0, 2]] = boxes_abs[:, [0, 2]].clamp(0.0, float(crop_w))
        boxes_abs[:, [1, 3]] = boxes_abs[:, [1, 3]].clamp(0.0, float(crop_h))

        widths = boxes_abs[:, 2] - boxes_abs[:, 0]
        heights = boxes_abs[:, 3] - boxes_abs[:, 1]
        keep = (widths > 1.0) & (heights > 1.0)
        if not bool(keep.any()):
            return img, boxes_rel, class_ids

        cropped_img = img.crop((left, top, left + crop_w, top + crop_h))
        cropped_boxes_rel = self._boxes_xyxy_abs_to_cxcywh_rel(
            boxes_abs[keep],
            img_w=crop_w,
            img_h=crop_h,
        )
        cropped_class_ids = class_ids[keep]
        return cropped_img, cropped_boxes_rel, cropped_class_ids

    def _apply_train_augment(
        self, img: Image.Image, boxes_rel: torch.Tensor, class_ids: torch.Tensor
    ):
        img_aug, boxes_aug, class_ids_aug = self._random_resized_crop(
            img,
            boxes_rel,
            class_ids,
            p=0.5,
            min_scale=0.7,
        )
        img_aug, boxes_aug = self._random_horizontal_flip(img_aug, boxes_aug, p=0.5)
        if self.photometric_aug is not None:
            img_aug = self.photometric_aug(img_aug)
        return img_aug, boxes_aug, class_ids_aug

    def __getitem__(self, idx: int):
        record = None
        if self.annotation_format == "yolo":
            img_path = self.files[idx]
            label_path = self.labels_dir / (img_path.stem + ".txt")
            img = Image.open(img_path).convert("RGB")
            boxes_rel, class_ids = self._read_yolo_labels(label_path)
        elif self.annotation_format == "coco_json":
            record = self.records[idx]
            img_path = record["image_path"]
            img = Image.open(img_path).convert("RGB")
            boxes_rel, class_ids = self._read_coco_labels(record, img.size)
        else:
            record = self.records[idx]
            img_path = record["image_path"]
            img = Image.open(img_path).convert("RGB")
            boxes_rel = record["boxes_rel"]
            class_ids = record["class_ids"]

        if self.augment and self._use_default_transform:
            img, boxes_rel, class_ids = self._apply_train_augment(img, boxes_rel, class_ids)

        img_t = self.transform(img)
        caption = self._compose_caption(record, class_ids)

        if self.tokenizer is not None:
            token_ids = self.tokenizer.encode([caption])[0]
        else:
            token_ids = torch.zeros((32,), dtype=torch.long)

        return img_t, token_ids.long(), boxes_rel, class_ids
