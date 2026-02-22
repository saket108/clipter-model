"""Aggregate detector experiment summaries into flat and grouped CSV reports."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


RUN_ID_TAG_RE = re.compile(r"^detect(?:_fast)?_\d{8}_\d{6}_(.+)$")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _extract_group_tag(run_id: str) -> str:
    m = RUN_ID_TAG_RE.match(run_id)
    if m:
        return m.group(1)
    return run_id


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _collect_rows(experiments_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(experiments_root.glob("*/summary.json")):
        summary = _load_json(summary_path)
        if not isinstance(summary, dict):
            continue

        run_dir = summary_path.parent
        config_payload = _load_json(run_dir / "config.json") or {}
        cfg = config_payload.get("config", {}) if isinstance(config_payload, dict) else {}

        run_id = str(summary.get("run_id", run_dir.name))
        row = {
            "run_id": run_id,
            "group_tag": _extract_group_tag(run_id),
            "timestamp_utc": summary.get("timestamp_utc"),
            "best_map": _to_float(summary.get("best_map")),
            "best_val_loss": _to_float(summary.get("best_val_loss")),
            "best_map_epoch": summary.get("best_map_epoch"),
            "best_val_loss_epoch": summary.get("best_val_loss_epoch"),
            "num_classes": summary.get("num_classes"),
            "fast": summary.get("fast"),
            "subset": summary.get("subset"),
            "train_split": summary.get("train_split"),
            "val_split": summary.get("val_split"),
            "clip_init_used": summary.get("clip_init_used"),
            "requested_device": config_payload.get("requested_device"),
            "resolved_device": config_payload.get("device"),
            "resolved_epochs": config_payload.get("resolved_epochs"),
            "resolved_batch_size": config_payload.get("resolved_batch_size"),
            "seed": cfg.get("seed"),
            "lr": cfg.get("lr"),
            "weight_decay": cfg.get("weight_decay"),
            "image_backbone": cfg.get("image_backbone"),
            "image_size": cfg.get("image_size"),
            "embed_dim": cfg.get("embed_dim"),
            "det_num_queries": cfg.get("det_num_queries"),
            "det_decoder_layers": cfg.get("det_decoder_layers"),
            "det_ff_dim": cfg.get("det_ff_dim"),
            "det_dropout": cfg.get("det_dropout"),
            "freeze_backbone_epochs": cfg.get("freeze_backbone_epochs"),
        }
        rows.append(row)

    rows.sort(key=lambda r: (str(r.get("timestamp_utc", "")), str(r.get("run_id", ""))))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(mean(values))


def _safe_std(values: list[float]) -> float | None:
    if not values:
        return None
    return float(pstdev(values))


def _group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group_tag"])].append(row)

    out: list[dict[str, Any]] = []
    for group_tag, items in grouped.items():
        maps = [x for x in (_to_float(r.get("best_map")) for r in items) if x is not None]
        losses = [x for x in (_to_float(r.get("best_val_loss")) for r in items) if x is not None]

        best_single = None
        if maps:
            best_single = max(maps)

        out.append(
            {
                "group_tag": group_tag,
                "num_runs": len(items),
                "best_map_mean": _safe_mean(maps),
                "best_map_std": _safe_std(maps),
                "best_val_loss_mean": _safe_mean(losses),
                "best_val_loss_std": _safe_std(losses),
                "best_single_map": best_single,
            }
        )

    out.sort(key=lambda r: str(r["group_tag"]))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiments-root", type=str, default="experiments")
    p.add_argument("--output-csv", type=str, default="reports/detect_runs_flat.csv")
    p.add_argument("--output-grouped-csv", type=str, default="reports/detect_runs_grouped.csv")
    args = p.parse_args()

    experiments_root = Path(args.experiments_root)
    if not experiments_root.exists():
        raise FileNotFoundError(f"Experiments root not found: {experiments_root}")

    rows = _collect_rows(experiments_root)
    if len(rows) == 0:
        raise RuntimeError(f"No summary.json files found under: {experiments_root}")

    flat_fields = [
        "run_id",
        "group_tag",
        "timestamp_utc",
        "best_map",
        "best_val_loss",
        "best_map_epoch",
        "best_val_loss_epoch",
        "num_classes",
        "fast",
        "subset",
        "train_split",
        "val_split",
        "clip_init_used",
        "requested_device",
        "resolved_device",
        "resolved_epochs",
        "resolved_batch_size",
        "seed",
        "lr",
        "weight_decay",
        "image_backbone",
        "image_size",
        "embed_dim",
        "det_num_queries",
        "det_decoder_layers",
        "det_ff_dim",
        "det_dropout",
        "freeze_backbone_epochs",
    ]
    _write_csv(Path(args.output_csv), rows, flat_fields)

    grouped_rows = _group_rows(rows)
    grouped_fields = [
        "group_tag",
        "num_runs",
        "best_map_mean",
        "best_map_std",
        "best_val_loss_mean",
        "best_val_loss_std",
        "best_single_map",
    ]
    _write_csv(Path(args.output_grouped_csv), grouped_rows, grouped_fields)

    print(f"Wrote flat report: {args.output_csv}")
    print(f"Wrote grouped report: {args.output_grouped_csv}")
    print(f"Runs aggregated: {len(rows)}")


if __name__ == "__main__":
    main()
