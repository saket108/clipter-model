#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from dataset_script_utils import resolve_dataset_root, resolve_input_files, resolve_output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert aircraft-damage JSON annotations into CLIP-style image-text pairs."
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
        default=Path("multimodal_dataset.json"),
        help="Output path relative to the dataset root unless absolute.",
    )
    parser.add_argument(
        "--caption-style",
        choices=("first-sentence", "raw-description", "structured"),
        default="first-sentence",
        help="How to build captions from each annotation.",
    )
    return parser.parse_args()


def normalize_text(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_caption_text(text: str) -> str:
    text = normalize_text(text)
    return text.lower() if text else text


def first_sentence(text: str) -> str:
    cleaned = normalize_text(text)
    if not cleaned:
        return cleaned
    return normalize_caption_text(cleaned.split(". ", 1)[0].rstrip("."))


def structured_caption(annotation: dict[str, object]) -> str:
    damage = normalize_text(annotation.get("category_name", "unknown damage")).lower()
    severity = normalize_text(
        annotation.get("risk_assessment", {}).get("severity_level", "unknown")
    ).lower()
    zone = normalize_text(annotation.get("zone_estimation", "unknown")).lower()
    return f"{severity} {damage} detected in the {zone} structural region of the aircraft surface"


def build_caption(annotation: dict[str, object], style: str) -> str:
    description = normalize_text(annotation.get("description", ""))
    if style == "raw-description":
        return normalize_caption_text(description)
    if style == "structured":
        return structured_caption(annotation)
    if description:
        return first_sentence(description)
    return structured_caption(annotation)


def infer_split(image_record: dict[str, object], json_path: Path) -> str:
    split = image_record.get("split")
    if split:
        return normalize_text(split).lower()
    return json_path.stem.lower()


def build_pairs(json_files: list[Path], caption_style: str) -> tuple[list[dict[str, str]], int]:
    pairs: list[dict[str, str]] = []
    skipped = 0

    for json_path in json_files:
        with json_path.open(encoding="utf-8") as handle:
            data = json.load(handle)

        for image in data.get("images", []):
            split = infer_split(image, json_path)
            file_name = normalize_text(image.get("file_name", ""))
            if not file_name:
                skipped += 1
                continue

            image_path = f"{split}/images/{file_name}"
            for annotation in image.get("annotations", []):
                caption = build_caption(annotation, caption_style)
                if not caption:
                    skipped += 1
                    continue
                pairs.append({"image": image_path, "caption": normalize_caption_text(caption)})

    return pairs, skipped


def main() -> int:
    args = parse_args()
    dataset_root = resolve_dataset_root(args.dataset_root)
    json_files = resolve_input_files(dataset_root, args.input_json)
    output_path = resolve_output_path(dataset_root, args.output_file)

    pairs, skipped = build_pairs(json_files, args.caption_style)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(pairs, handle, indent=2)

    print(
        f"Generated {len(pairs)} image-text pairs from {len(json_files)} file(s) "
        f"using '{args.caption_style}' captions -> {output_path}"
    )
    if skipped:
        print(f"Skipped {skipped} incomplete records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
