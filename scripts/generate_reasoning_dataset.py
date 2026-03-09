#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from dataset_script_utils import resolve_dataset_root, resolve_input_files, resolve_output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert aircraft-damage JSON annotations into box-level multimodal reasoning samples."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Dataset root or parent dataset directory. Defaults to the Desktop datasets path.",
    )
    parser.add_argument(
        "--input-json",
        nargs="+",
        type=Path,
        help="JSON files to scan relative to the dataset root.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("multimodal_reasoning_dataset.json"),
        help="Output path relative to the dataset root unless absolute.",
    )
    parser.add_argument(
        "--caption-style",
        choices=("structured", "first-sentence", "raw-description"),
        default="structured",
        help="How to build the caption field.",
    )
    return parser.parse_args()


def normalize_text(value: object) -> str:
    text = str(value).strip().lower()
    return re.sub(r"\s+", " ", text)


def first_sentence(text: str) -> str:
    cleaned = normalize_text(text)
    if not cleaned:
        return cleaned
    return cleaned.split(". ", 1)[0].rstrip(".")


def structured_caption(annotation: dict[str, object]) -> str:
    severity = normalize_text(
        annotation.get("risk_assessment", {}).get("severity_level", "unknown")
    )
    damage = normalize_text(annotation.get("category_name", "unknown-damage"))
    zone = normalize_text(annotation.get("zone_estimation", "unknown"))
    return f"{severity} {damage} detected in {zone} aircraft structure"


def build_caption(annotation: dict[str, object], style: str) -> str:
    description = normalize_text(annotation.get("description", ""))
    if style == "raw-description":
        return description
    if style == "first-sentence" and description:
        return first_sentence(description)
    return structured_caption(annotation)


def infer_split(image_record: dict[str, object], json_path: Path) -> str:
    split = image_record.get("split")
    if split:
        return normalize_text(split)
    return normalize_text(json_path.stem)


def build_samples(json_files: list[Path], caption_style: str) -> tuple[list[dict[str, object]], int]:
    samples: list[dict[str, object]] = []
    skipped = 0

    for json_path in json_files:
        with json_path.open(encoding="utf-8") as handle:
            data = json.load(handle)

        for image in data.get("images", []):
            split = infer_split(image, json_path)
            file_name = image.get("file_name")
            image_id = image.get("image_id")
            if not file_name:
                skipped += 1
                continue

            image_path = f"{split}/images/{file_name}"
            for annotation in image.get("annotations", []):
                box = annotation.get("bounding_box_normalized", {})
                required = ("x_center", "y_center", "width", "height")
                if not all(key in box for key in required):
                    skipped += 1
                    continue

                samples.append(
                    {
                        "image": image_path,
                        "bbox": [
                            box["x_center"],
                            box["y_center"],
                            box["width"],
                            box["height"],
                        ],
                        "class": normalize_text(annotation.get("category_name", "unknown-damage")),
                        "caption": build_caption(annotation, caption_style),
                        "severity": normalize_text(
                            annotation.get("risk_assessment", {}).get("severity_level", "unknown")
                        ),
                        "zone": normalize_text(annotation.get("zone_estimation", "unknown")),
                        "split": split,
                        "image_id": image_id,
                        "annotation_id": annotation.get("annotation_id"),
                    }
                )

    return samples, skipped


def main() -> int:
    args = parse_args()
    dataset_root = resolve_dataset_root(args.dataset_root)
    json_files = resolve_input_files(dataset_root, args.input_json)
    output_path = resolve_output_path(dataset_root, args.output_file)

    samples, skipped = build_samples(json_files, args.caption_style)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(samples, handle, indent=2)

    print(
        f"Generated {len(samples)} reasoning samples from {len(json_files)} file(s) "
        f"using '{args.caption_style}' captions -> {output_path}"
    )
    if skipped:
        print(f"Skipped {skipped} incomplete annotations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
