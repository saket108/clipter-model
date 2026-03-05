"""One-command launcher for clipdetr training from a YOLO-style data.yaml.

Example:
  python scripts/train_from_data_yaml.py --data /path/to/data.yaml --epochs 80 --batch-size 8
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


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


def _resolve_split_name(root: Path, candidates: list[str | None]) -> str | None:
    for split in candidates:
        if split is None:
            continue
        s = split.strip()
        if not s:
            continue
        split_dir = root / s
        if (split_dir / "images").exists() and (split_dir / "labels").exists():
            return s
    return None


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except Exception as e:  # pragma: no cover - import error path
        raise RuntimeError(
            "PyYAML is required. Install with: pip install pyyaml"
        ) from e

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML dict in {path}, got {type(data).__name__}")
    return data


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train clipdetr from data.yaml with auto split/path resolution."
    )
    p.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument(
        "--image-backbone",
        choices=["mobilenet_v3_small", "convnext_tiny"],
        default=None,
    )
    p.add_argument("--embed-dim", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--subset", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="Print resolved command and exit.")
    strict_group = p.add_mutually_exclusive_group()
    strict_group.add_argument("--strict-class-check", dest="strict_class_check", action="store_true")
    strict_group.add_argument("--no-strict-class-check", dest="strict_class_check", action="store_false")
    p.set_defaults(strict_class_check=None)
    return p


def main() -> int:
    parser = _build_parser()
    args, passthrough = parser.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    data_yaml = Path(args.data).expanduser().resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")
    if not data_yaml.is_file():
        raise ValueError(f"--data must point to a file, got: {data_yaml}")

    data_root = data_yaml.parent
    data_cfg = _load_yaml(data_yaml)

    yaml_train_split = _split_name_from_yaml_entry(data_cfg.get("train"))
    yaml_val_split = _split_name_from_yaml_entry(data_cfg.get("val"))
    yaml_test_split = _split_name_from_yaml_entry(data_cfg.get("test"))

    train_split = _resolve_split_name(
        data_root,
        [yaml_train_split, "stratified_train", "train"],
    )
    val_split = _resolve_split_name(
        data_root,
        [
            yaml_val_split,
            "stratified_val_10pct",
            "stratified_val",
            "valid",
            "val",
            yaml_test_split,
            "test",
        ],
    )

    if train_split is None:
        raise FileNotFoundError(
            "Could not resolve train split under dataset root "
            f"{data_root}. Expected '<split>/images' and '<split>/labels'."
        )

    repo_root = Path(__file__).resolve().parents[1]
    train_script = repo_root / "clipdetr" / "train_detect.py"

    cmd: list[str] = [
        sys.executable,
        str(train_script),
        "--data-root",
        str(data_root),
        "--data-yaml",
        data_yaml.name,
        "--train-split",
        train_split,
        "--device",
        args.device,
    ]

    if val_split is not None:
        cmd.extend(["--val-split", val_split])
    if args.epochs is not None:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.image_size is not None:
        cmd.extend(["--image-size", str(args.image_size)])
    if args.image_backbone is not None:
        cmd.extend(["--image-backbone", args.image_backbone])
    if args.embed_dim is not None:
        cmd.extend(["--embed-dim", str(args.embed_dim)])
    if args.num_workers is not None:
        cmd.extend(["--num-workers", str(args.num_workers)])
    if args.tag:
        cmd.extend(["--tag", args.tag])
    if args.fast:
        cmd.append("--fast")
    if args.subset is not None:
        cmd.extend(["--subset", str(args.subset)])
    if args.strict_class_check is True:
        cmd.append("--strict-class-check")
    elif args.strict_class_check is False:
        cmd.append("--no-strict-class-check")

    cmd.extend(passthrough)

    print(f"Resolved data root : {data_root}")
    print(f"Resolved train split: {train_split}")
    print(f"Resolved val split  : {val_split if val_split is not None else 'none'}")
    print("Launching command:")
    print("  " + " ".join(shlex.quote(x) for x in cmd))

    if args.dry_run:
        return 0

    completed = subprocess.run(cmd, cwd=repo_root, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
