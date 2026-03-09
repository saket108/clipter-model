#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset_script_utils import resolve_dataset_root, resolve_input_files, resolve_output_path


DEFAULT_TEMPLATES = [
    "a {severity} {damage} on aircraft {zone} surface",
    "aircraft {damage} damage in the {zone} structural region",
    "{severity} severity {damage} on aircraft fuselage panel",
    "structural {damage} detected on aircraft {zone} area",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CLIP-style text prompts from aircraft-damage JSON annotations."
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
        default=Path("prompts") / "class_prompts.txt",
        help="Output path relative to the dataset root unless absolute.",
    )
    return parser.parse_args()


def normalize_text(value: object) -> str:
    return str(value).strip().lower()


def collect_prompts(json_files: list[Path]) -> tuple[set[str], int]:
    prompts: set[str] = set()
    annotation_count = 0

    for json_path in json_files:
        with json_path.open(encoding="utf-8") as handle:
            data = json.load(handle)

        for image in data.get("images", []):
            for annotation in image.get("annotations", []):
                damage = normalize_text(annotation.get("category_name", "unknown-damage"))
                risk = annotation.get("risk_assessment", {})
                severity = normalize_text(risk.get("severity_level", "unknown"))
                zone = normalize_text(annotation.get("zone_estimation", "unknown"))
                annotation_count += 1

                for template in DEFAULT_TEMPLATES:
                    prompts.add(
                        template.format(
                            damage=damage,
                            severity=severity,
                            zone=zone,
                        )
                    )

    return prompts, annotation_count


def main() -> int:
    args = parse_args()
    dataset_root = resolve_dataset_root(args.dataset_root)
    json_files = resolve_input_files(dataset_root, args.input_json)
    output_path = resolve_output_path(dataset_root, args.output_file)

    prompts, annotation_count = collect_prompts(json_files)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for prompt in sorted(prompts):
            handle.write(prompt + "\n")

    print(
        f"Generated {len(prompts)} prompts from {annotation_count} annotations "
        f"across {len(json_files)} file(s) -> {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
