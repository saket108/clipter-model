#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from dataset_script_utils import resolve_dataset_root, resolve_input_files, resolve_output_path


DEFAULT_CLASS_ALIASES = {
    "crack": "crack",
    "dent": "dent",
    "scratch": "scratch",
    "missing-head": "missing head fastener",
    "paint-off": "paint peeling",
}

BASE_TEMPLATES = [
    "a {damage} on aircraft metal surface",
    "aircraft structural {damage}",
    "a {damage} defect on aircraft fuselage",
    "visible {damage} on aircraft aluminum skin",
    "{damage} damage on aircraft structure",
]

CONTEXT_TEMPLATES = [
    "{severity} severity {damage} on aircraft {zone} panel",
    "{damage} in the aircraft {zone} structural region",
    "{severity} {damage} near the aircraft {zone} structure",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate prompt ensembles for aircraft-damage multimodal training."
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
        "--output-text",
        type=Path,
        default=Path("prompts") / "class_prompt_ensemble.txt",
        help="Flat text output path relative to the dataset root unless absolute.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("prompts") / "class_prompt_ensemble.json",
        help="Grouped JSON output path relative to the dataset root unless absolute.",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Generate only the generic class ensemble and skip severity/zone context prompts.",
    )
    return parser.parse_args()


def normalize_text(value: object) -> str:
    return str(value).strip().lower()


def load_annotation_contexts(json_files: list[Path]) -> dict[str, set[tuple[str, str]]]:
    contexts: dict[str, set[tuple[str, str]]] = defaultdict(set)

    for json_path in json_files:
        with json_path.open(encoding="utf-8") as handle:
            data = json.load(handle)

        for image in data.get("images", []):
            for annotation in image.get("annotations", []):
                class_name = normalize_text(annotation.get("category_name", "unknown"))
                severity = normalize_text(
                    annotation.get("risk_assessment", {}).get("severity_level", "unknown")
                )
                zone = normalize_text(annotation.get("zone_estimation", "unknown"))
                contexts[class_name].add((severity, zone))

    return contexts


def ordered_class_names(contexts: dict[str, set[tuple[str, str]]]) -> list[str]:
    preferred = [name for name in DEFAULT_CLASS_ALIASES if name in contexts]
    extras = sorted(name for name in contexts if name not in DEFAULT_CLASS_ALIASES)
    return preferred + extras


def class_phrase(class_name: str) -> str:
    return DEFAULT_CLASS_ALIASES.get(class_name, class_name.replace("-", " "))


def build_prompt_groups(
    contexts: dict[str, set[tuple[str, str]]],
    include_context: bool,
) -> dict[str, list[str]]:
    grouped_prompts: dict[str, list[str]] = {}

    for class_name in ordered_class_names(contexts):
        damage = class_phrase(class_name)
        prompts = {template.format(damage=damage) for template in BASE_TEMPLATES}

        if include_context:
            for severity, zone in sorted(contexts[class_name]):
                for template in CONTEXT_TEMPLATES:
                    prompts.add(
                        template.format(
                            damage=damage,
                            severity=severity,
                            zone=zone,
                        )
                    )

        grouped_prompts[class_name] = sorted(prompts)

    return grouped_prompts


def main() -> int:
    args = parse_args()
    dataset_root = resolve_dataset_root(args.dataset_root)
    json_files = resolve_input_files(dataset_root, args.input_json)
    output_text = resolve_output_path(dataset_root, args.output_text)
    output_json = resolve_output_path(dataset_root, args.output_json)

    contexts = load_annotation_contexts(json_files)
    grouped_prompts = build_prompt_groups(contexts, include_context=not args.base_only)

    output_text.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with output_text.open("w", encoding="utf-8") as handle:
        for class_name, prompts in grouped_prompts.items():
            handle.write(f"# {class_name}\n")
            for prompt in prompts:
                handle.write(prompt + "\n")
            handle.write("\n")

    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(grouped_prompts, handle, indent=2)

    class_count = len(grouped_prompts)
    prompt_count = sum(len(prompts) for prompts in grouped_prompts.values())
    mode = "base-only" if args.base_only else "base+context"
    print(
        f"Generated {prompt_count} prompts across {class_count} classes "
        f"using {mode} ensembling -> {output_text}, {output_json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
