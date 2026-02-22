"""Audit detection dataset integrity (class coverage and split consistency)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_detect import (  # noqa: E402
    _class_names_from_yaml,
    _load_data_yaml,
    _nonzero_class_ids,
    build_datasets,
    cfg,
    infer_num_classes,
)
from utils.dataset_stats import compute_class_distribution  # noqa: E402


def _apply_args(args):
    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.data_yaml is not None:
        cfg.data_yaml = args.data_yaml
    if args.classes_path is not None:
        cfg.classes_path = args.classes_path
    if args.train_split is not None:
        cfg.train_split = args.train_split
    if args.val_split is not None:
        cfg.val_split = args.val_split
    if args.annotation_format is not None:
        cfg.annotation_format = args.annotation_format
    if args.class_stats_max_samples is not None:
        cfg.class_stats_max_samples = args.class_stats_max_samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--data-yaml", type=str, default=None)
    p.add_argument("--classes-path", type=str, default=None)
    p.add_argument("--train-split", type=str, default=None)
    p.add_argument("--val-split", type=str, default=None)
    p.add_argument("--annotation-format", type=str, default=None, choices=["auto", "yolo", "coco_json"])
    p.add_argument("--class-stats-max-samples", type=int, default=None)
    p.add_argument("--output-json", type=str, default=None)
    p.add_argument("--strict", action="store_true", help="exit non-zero on split/class consistency failures")
    args = p.parse_args()

    _apply_args(args)

    train_ds, val_ds, train_split, val_split = build_datasets()
    base_train_ds = train_ds.dataset if hasattr(train_ds, "dataset") else train_ds

    num_classes = getattr(base_train_ds, "num_classes", 0)
    if num_classes <= 0:
        num_classes = infer_num_classes(base_train_ds)

    train_stats = compute_class_distribution(
        train_ds,
        num_classes=num_classes,
        max_samples=cfg.class_stats_max_samples,
    )
    train_present = _nonzero_class_ids(train_stats)
    train_missing = sorted(set(range(num_classes)) - train_present)

    val_stats = None
    val_present = set()
    val_missing = []
    val_only = []
    train_only = []

    if val_ds is not None:
        val_stats = compute_class_distribution(
            val_ds,
            num_classes=num_classes,
            max_samples=cfg.class_stats_max_samples,
        )
        val_present = _nonzero_class_ids(val_stats)
        val_missing = sorted(set(range(num_classes)) - val_present)
        val_only = sorted(val_present - train_present)
        train_only = sorted(train_present - val_present)

    data_yaml = _load_data_yaml(Path(cfg.data_root), cfg.data_yaml)
    yaml_names = _class_names_from_yaml(data_yaml)
    yaml_name_count = len(yaml_names) if yaml_names is not None else None

    warnings = []
    if yaml_name_count is not None and yaml_name_count != num_classes:
        warnings.append(
            f"data.yaml names count ({yaml_name_count}) != detected num_classes ({num_classes})"
        )
    if len(val_only) > 0:
        warnings.append(
            f"validation contains classes absent in train: {val_only}"
        )

    report = {
        "data_root": str(Path(cfg.data_root).resolve()),
        "train_split": train_split,
        "val_split": val_split,
        "annotation_format_train": getattr(train_ds, "annotation_format", None),
        "num_classes_detected": num_classes,
        "yaml_names_count": yaml_name_count,
        "train": {
            **train_stats,
            "present_class_ids": sorted(train_present),
            "missing_class_ids": train_missing,
        },
        "val": (
            {
                **val_stats,
                "present_class_ids": sorted(val_present),
                "missing_class_ids": val_missing,
                "classes_only_in_val": val_only,
                "classes_only_in_train": train_only,
            }
            if val_stats is not None
            else None
        ),
        "warnings": warnings,
    }

    print(
        f"Audit summary: num_classes={num_classes}, train_split='{train_split}', "
        f"val_split='{val_split}'"
    )
    print(
        "Train stats: "
        f"samples_checked={train_stats['num_samples_checked']}, "
        f"total_boxes={train_stats['total_boxes']}, "
        f"empty_images={train_stats['empty_images']}"
    )
    if val_stats is not None:
        print(
            "Val stats: "
            f"samples_checked={val_stats['num_samples_checked']}, "
            f"total_boxes={val_stats['total_boxes']}, "
            f"empty_images={val_stats['empty_images']}"
        )

    if len(warnings) > 0:
        print("Warnings:")
        for w in warnings:
            print(f"- {w}")

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
        print(f"Audit report saved to: {out}")

    if args.strict and len(warnings) > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
