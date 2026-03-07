#!/usr/bin/env python3
"""One-command CLIPTER pipeline.

Modes:
- train: train only
- eval : eval only
- run  : train + eval (+ optional tune/benchmark)
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clipdetr.utils.build_tiled_yolo_dataset import build_tiled_dataset


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_cmd(cmd: list[str], cwd: Path, dry_run: bool) -> int:
    print(">>", " ".join(shlex.quote(str(x)) for x in cmd))
    if dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    return int(proc.returncode)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _resolve_checkpoint(summary: dict[str, Any] | None, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    if not isinstance(summary, dict):
        return None
    for k in ("best_map_checkpoint", "best_loss_checkpoint", "final_checkpoint"):
        v = summary.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def _parse_csv(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _resolve_tile_stitch_eval(explicit: bool | None, tile_size: int) -> bool:
    if explicit is not None:
        return bool(explicit)
    return int(tile_size) > 0


def _planned_tile_root(output_dir: Path, tile_size: int, tile_overlap: float, tile_min_cover: float) -> Path:
    overlap_tag = str(int(round(float(tile_overlap) * 100.0)))
    cover_tag = str(int(round(float(tile_min_cover) * 100.0)))
    return output_dir / "prepared" / f"tiles_s{int(tile_size)}_o{overlap_tag}_c{cover_tag}"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CLIPTER end-to-end pipeline.")
    p.add_argument("--mode", choices=["train", "eval", "run"], default="run")
    p.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    p.add_argument("--output-dir", default="experiments/pipeline_run")

    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--image-size", type=int, default=320)
    p.add_argument(
        "--image-backbone",
        choices=["mobilenet_v3_small", "convnext_tiny"],
        default="convnext_tiny",
    )
    p.add_argument("--embed-dim", type=int, default=384)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--tag", default="pipeline_run")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--subset", type=int, default=None)
    p.add_argument("--use-multiscale-memory", action="store_true", help="enable multi-scale backbone token fusion in LightDETR")
    p.add_argument("--use-multiscale-neck", action="store_true", help="enable structured multiscale feature fusion before decoder memory construction")
    p.add_argument("--multiscale-levels", type=int, default=3, help="number of backbone stages to fuse when multi-scale memory is enabled")

    strict_group = p.add_mutually_exclusive_group()
    strict_group.add_argument("--strict-class-check", dest="strict_class_check", action="store_true")
    strict_group.add_argument("--no-strict-class-check", dest="strict_class_check", action="store_false")
    p.set_defaults(strict_class_check=True)

    p.add_argument("--checkpoint", default=None, help="Optional checkpoint for eval-only mode.")
    p.add_argument("--eval-conf-thres", type=float, default=None)
    p.add_argument("--eval-top-k", type=int, default=None)
    p.add_argument("--eval-nms-iou", type=float, default=None)
    p.add_argument("--tile-size", type=int, default=0, help="If > 0, build a tiled dataset under output-dir and train/eval on tiles.")
    p.add_argument("--tile-overlap", type=float, default=0.2, help="Tile overlap ratio in [0, 1).")
    p.add_argument("--tile-min-cover", type=float, default=0.35, help="Minimum GT area ratio that must remain inside a tile.")
    p.add_argument("--tile-splits", default="", help="Comma-separated split names to tile. Empty means infer from data.yaml.")
    p.add_argument("--include-empty-tiles", action="store_true", help="Keep empty tiles instead of dropping them.")
    tile_group = p.add_mutually_exclusive_group()
    tile_group.add_argument("--tile-stitch-eval", dest="tile_stitch_eval", action="store_true")
    tile_group.add_argument("--no-tile-stitch-eval", dest="tile_stitch_eval", action="store_false")
    p.set_defaults(tile_stitch_eval=None)
    p.add_argument("--tile-stitch-nms-iou", type=float, default=0.5)
    p.add_argument("--tile-stitch-gt-dedup-iou", type=float, default=0.9)

    p.add_argument("--tune-thresholds", action="store_true")
    p.add_argument("--conf-grid", default="0.001,0.01,0.05,0.1,0.2,0.3")
    p.add_argument("--nms-grid", default="0.0,0.3,0.5")
    p.add_argument("--topk-grid", default="50,100")
    p.add_argument("--optimize", choices=["map", "map50", "map75"], default="map")

    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--dataset-name", default=None)
    p.add_argument("--split", default="valid")
    p.add_argument("--yolo-metrics", default="")
    p.add_argument("--detr-metrics", default="")
    p.add_argument("--yolo-cmd", default="")
    p.add_argument("--detr-cmd", default="")

    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    if args.use_multiscale_memory and args.use_multiscale_neck:
        raise ValueError("Use only one of --use-multiscale-memory or --use-multiscale-neck.")
    root = _repo_root()
    input_data_yaml = Path(args.data).expanduser().resolve()
    if not input_data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {input_data_yaml}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_out = output_dir / "train_summary.json"
    eval_out = output_dir / "eval_clipter.json"
    tune_out = output_dir / "best_thresholds.json"
    bench_out = output_dir / "baseline_benchmarks.json"
    bench_csv = output_dir / "baseline_benchmarks.csv"
    manifest_path = output_dir / "pipeline_manifest.json"
    tile_report_path = output_dir / "prepared" / "tile_report.json"
    tile_root = _planned_tile_root(output_dir, args.tile_size, args.tile_overlap, args.tile_min_cover)
    tile_splits = _parse_csv(args.tile_splits)
    effective_tile_stitch_eval = _resolve_tile_stitch_eval(args.tile_stitch_eval, args.tile_size)

    manifest: dict[str, Any] = {
        "created_at_utc": _now_iso(),
        "mode": args.mode,
        "data_yaml": str(input_data_yaml),
        "output_dir": str(output_dir.resolve()),
        "args": vars(args),
        "commands": {},
        "status": {},
        "artifacts": {
            "train_summary": str(summary_out),
            "eval_metrics": str(eval_out),
            "threshold_tuning": str(tune_out),
            "baseline_json": str(bench_out),
            "baseline_csv": str(bench_csv),
            "tile_report": str(tile_report_path),
        },
        "tiling": {
            "enabled": bool(args.tile_size > 0),
            "tile_size": int(args.tile_size),
            "tile_overlap": float(args.tile_overlap),
            "tile_min_cover": float(args.tile_min_cover),
            "tile_splits": tile_splits,
            "include_empty_tiles": bool(args.include_empty_tiles),
            "effective_tile_stitch_eval": bool(effective_tile_stitch_eval),
            "tile_stitch_nms_iou": float(args.tile_stitch_nms_iou),
            "tile_stitch_gt_dedup_iou": float(args.tile_stitch_gt_dedup_iou),
        },
    }

    active_data_yaml = input_data_yaml
    if args.tile_size > 0:
        tile_cmd = [
            sys.executable,
            "clipdetr/utils/build_tiled_yolo_dataset.py",
            "--root",
            str(input_data_yaml.parent),
            "--data-yaml",
            input_data_yaml.name,
            "--output-root",
            str(tile_root),
            "--tile-size",
            str(args.tile_size),
            "--overlap",
            str(args.tile_overlap),
            "--min-cover",
            str(args.tile_min_cover),
            "--report-json",
            str(tile_report_path),
        ]
        if tile_splits:
            tile_cmd.extend(["--tile-splits", args.tile_splits])
        if args.include_empty_tiles:
            tile_cmd.append("--include-empty-tiles")
        manifest["commands"]["prepare_tiles"] = tile_cmd

        if args.dry_run:
            manifest["status"]["prepare_tiles_rc"] = 0
        else:
            tile_report = build_tiled_dataset(
                root=input_data_yaml.parent,
                out_dir=tile_root,
                data_yaml=input_data_yaml,
                tile_size=int(args.tile_size),
                overlap=float(args.tile_overlap),
                min_cover=float(args.tile_min_cover),
                tile_splits=tile_splits or None,
                include_empty_tiles=bool(args.include_empty_tiles),
            )
            _write_json(tile_report_path, tile_report)
            manifest["tiling"]["report"] = tile_report
            manifest["status"]["prepare_tiles_rc"] = 0
        active_data_yaml = tile_root / "data.yaml"

    manifest["artifacts"]["active_data_yaml"] = str(active_data_yaml)
    _write_json(manifest_path, manifest)

    data_root = active_data_yaml.parent
    data_yaml_name = active_data_yaml.name

    train_cmd = [
        sys.executable,
        "scripts/train_from_data_yaml.py",
        "--data",
        str(active_data_yaml),
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--image-size",
        str(args.image_size),
        "--image-backbone",
        args.image_backbone,
        "--embed-dim",
        str(args.embed_dim),
        "--num-workers",
        str(args.num_workers),
        "--tag",
        args.tag,
        "--seed",
        str(args.seed),
        "--summary-out",
        str(summary_out),
        "--train-augment",
        "--use-ema",
        "--ema-decay",
        "0.999",
        "--warmup-epochs",
        "5",
        "--warmup-start-factor",
        "0.1",
        "--grad-clip-norm",
        "1.0",
        "--auto-clip-init",
        "--freeze-backbone-epochs",
        "3",
    ]
    if args.fast:
        train_cmd.append("--fast")
    if args.subset is not None:
        train_cmd.extend(["--subset", str(args.subset)])
    if args.strict_class_check:
        train_cmd.append("--strict-class-check")
    else:
        train_cmd.append("--no-strict-class-check")
    if args.use_multiscale_memory:
        train_cmd.extend(["--use-multiscale-memory", "--multiscale-levels", str(args.multiscale_levels)])
    if args.use_multiscale_neck:
        train_cmd.extend(["--use-multiscale-neck", "--multiscale-levels", str(args.multiscale_levels)])
    if effective_tile_stitch_eval:
        train_cmd.extend(
            [
                "--tile-stitch-eval",
                "--tile-stitch-nms-iou",
                str(args.tile_stitch_nms_iou),
                "--tile-stitch-gt-dedup-iou",
                str(args.tile_stitch_gt_dedup_iou),
            ]
        )
    elif args.tile_stitch_eval is False:
        train_cmd.append("--no-tile-stitch-eval")

    manifest["commands"]["train"] = train_cmd

    if args.mode in {"train", "run"}:
        rc = _run_cmd(train_cmd, cwd=root, dry_run=args.dry_run)
        manifest["status"]["train_rc"] = rc
        _write_json(manifest_path, manifest)
        if rc != 0:
            return rc

    summary_payload = None if args.dry_run else _load_json(summary_out)
    ckpt = _resolve_checkpoint(summary_payload, args.checkpoint)
    if args.dry_run and args.mode in {"eval", "run"} and ckpt is None:
        ckpt = str(output_dir / "checkpoints" / "resolved_after_train.pth")
    if args.mode in {"eval", "run"} and ckpt is None:
        raise RuntimeError(
            "Could not resolve checkpoint for evaluation. "
            "Use --checkpoint, or run train mode first."
        )

    eval_cmd = [
        sys.executable,
        "clipdetr/utils/eval_detect.py",
        "--checkpoint",
        str(ckpt),
        "--batch-size",
        str(args.eval_batch_size),
        "--data-root",
        str(data_root),
        "--data-yaml",
        data_yaml_name,
        "--output-json",
        str(eval_out),
    ]
    if args.eval_conf_thres is not None:
        eval_cmd.extend(["--conf-thres", str(args.eval_conf_thres)])
    if args.eval_top_k is not None:
        eval_cmd.extend(["--top-k", str(args.eval_top_k)])
    if args.eval_nms_iou is not None:
        eval_cmd.extend(["--nms-iou", str(args.eval_nms_iou)])
    if effective_tile_stitch_eval:
        eval_cmd.extend(
            [
                "--tile-stitch-eval",
                "--tile-stitch-nms-iou",
                str(args.tile_stitch_nms_iou),
                "--tile-stitch-gt-dedup-iou",
                str(args.tile_stitch_gt_dedup_iou),
            ]
        )
    elif args.tile_stitch_eval is False:
        eval_cmd.append("--no-tile-stitch-eval")

    manifest["commands"]["eval"] = eval_cmd

    if args.mode in {"eval", "run"}:
        rc = _run_cmd(eval_cmd, cwd=root, dry_run=args.dry_run)
        manifest["status"]["eval_rc"] = rc
        _write_json(manifest_path, manifest)
        if rc != 0:
            return rc

    if args.tune_thresholds and args.mode in {"eval", "run"}:
        tune_cmd = [
            sys.executable,
            "clipdetr/utils/tune_detect_thresholds.py",
            "--checkpoint",
            str(ckpt),
            "--batch-size",
            str(args.eval_batch_size),
            "--data-root",
            str(data_root),
            "--data-yaml",
            data_yaml_name,
            "--conf-grid",
            args.conf_grid,
            "--nms-grid",
            args.nms_grid,
            "--topk-grid",
            args.topk_grid,
            "--optimize",
            args.optimize,
            "--output-json",
            str(tune_out),
        ]
        if effective_tile_stitch_eval:
            tune_cmd.extend(
                [
                    "--tile-stitch-eval",
                    "--tile-stitch-nms-iou",
                    str(args.tile_stitch_nms_iou),
                    "--tile-stitch-gt-dedup-iou",
                    str(args.tile_stitch_gt_dedup_iou),
                ]
            )
        elif args.tile_stitch_eval is False:
            tune_cmd.append("--no-tile-stitch-eval")
        manifest["commands"]["tune_thresholds"] = tune_cmd
        rc = _run_cmd(tune_cmd, cwd=root, dry_run=args.dry_run)
        manifest["status"]["tune_rc"] = rc
        _write_json(manifest_path, manifest)
        if rc != 0:
            return rc

    if args.benchmark and args.mode in {"eval", "run"}:
        dataset_name = args.dataset_name or data_root.name
        bench_cmd = [
            sys.executable,
            "scripts/run_baseline_benchmarks.py",
            "--dataset",
            dataset_name,
            "--split",
            args.split,
            "--clipter-metrics",
            str(eval_out),
            "--summary-csv",
            str(bench_csv),
            "--output-json",
            str(bench_out),
        ]
        if args.yolo_metrics:
            bench_cmd.extend(["--yolo-metrics", args.yolo_metrics])
        if args.detr_metrics:
            bench_cmd.extend(["--detr-metrics", args.detr_metrics])
        if args.yolo_cmd:
            bench_cmd.extend(["--yolo-cmd", args.yolo_cmd])
        if args.detr_cmd:
            bench_cmd.extend(["--detr-cmd", args.detr_cmd])

        manifest["commands"]["benchmark"] = bench_cmd
        rc = _run_cmd(bench_cmd, cwd=root, dry_run=args.dry_run)
        manifest["status"]["benchmark_rc"] = rc
        _write_json(manifest_path, manifest)
        if rc != 0:
            return rc

    manifest["status"]["ok"] = True
    _write_json(manifest_path, manifest)
    print(f"Pipeline manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
