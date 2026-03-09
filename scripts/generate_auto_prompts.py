#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from dataset_script_utils import resolve_dataset_root, resolve_input_files, resolve_output_path


TEMPLATES = [
    "{severity} {damage} on aircraft {zone} structure",
    "aircraft {damage} damage in {zone} region",
    "{damage} defect on aircraft metal surface",
    "structural {damage} detected on aircraft panel",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate automatic prompt expansions from aircraft-damage JSON annotations."
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
        default=Path("prompts") / "auto_prompts.txt",
        help="Output path relative to the dataset root unless absolute.",
    )
    return parser.parse_args()


def normalize_text(value: object) -> str:
    text = str(value).strip().lower()
    return re.sub(r"\s+", " ", text)


def collect_prompts(json_files: list[Path]) -> tuple[set[str], int]:
    prompts: set[str] = set()
    annotation_count = 0

    for json_path in json_files:
        with json_path.open(encoding="utf-8") as handle:
            data = json.load(handle)

        for image in data.get("images", []):
            for annotation in image.get("annotations", []):
                damage = normalize_text(annotation.get("category_name", "unknown-damage"))
                zone = normalize_text(annotation.get("zone_estimation", "unknown"))
                severity = normalize_text(
                    annotation.get("risk_assessment", {}).get("severity_level", "unknown")
                )
                description = normalize_text(annotation.get("description", ""))

                for template in TEMPLATES:
                    prompts.add(
                        template.format(
                            damage=damage,
                            zone=zone,
                            severity=severity,
                        )
                    )

                if description:
                    prompts.add(description)

                annotation_count += 1

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
