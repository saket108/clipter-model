#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


DEFAULT_JSON_NAMES = ("train.json", "valid.json", "test.json")
DEFAULT_DATASET_HINTS = [
    Path(r"C:\Users\tsake\OneDrive\Desktop\datasets\Aero_dataset"),
    Path(r"C:\Users\tsake\OneDrive\Desktop\datasets"),
]
FINAL_DATASET_DIR_NAME = "Aero_dataset"


def _is_dataset_root(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    has_json = any((path / name).exists() for name in DEFAULT_JSON_NAMES)
    has_train = (path / "train").is_dir()
    has_valid = (path / "valid").is_dir()
    has_test = (path / "test").is_dir()
    return has_json or (has_train and has_valid and has_test)


def _resolve_dataset_child(parent: Path) -> Path | None:
    if not parent.exists() or not parent.is_dir():
        return None

    candidate = parent / FINAL_DATASET_DIR_NAME
    if _is_dataset_root(candidate):
        return candidate

    return None


def resolve_dataset_root(dataset_root: Path | None = None) -> Path:
    candidates: list[Path] = []
    if dataset_root is not None:
        candidates.append(Path(dataset_root).expanduser())
    candidates.extend(DEFAULT_DATASET_HINTS)
    candidates.append(Path.cwd())

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)

        if _is_dataset_root(candidate):
            return candidate

        child = _resolve_dataset_child(candidate)
        if child is not None:
            return child.resolve()

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not resolve a dataset root. "
        f"Tried: {searched}. Pass --dataset-root explicitly."
    )


def resolve_input_files(
    dataset_root: Path,
    explicit_files: list[Path] | None,
) -> list[Path]:
    if explicit_files:
        resolved = [
            path if path.is_absolute() else dataset_root / path
            for path in explicit_files
        ]
    else:
        resolved = [
            dataset_root / name
            for name in DEFAULT_JSON_NAMES
            if (dataset_root / name).exists()
        ]

    if not resolved:
        raise FileNotFoundError(
            f"No input JSON files found under dataset root: {dataset_root}"
        )

    missing = [str(path) for path in resolved if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Input JSON files not found: {', '.join(missing)}")

    return resolved


def resolve_output_path(dataset_root: Path, output_path: Path) -> Path:
    output = Path(output_path)
    if not output.is_absolute():
        output = dataset_root / output
    return output
