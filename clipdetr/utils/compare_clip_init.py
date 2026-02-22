"""Run a formal scratch vs CLIP-init detector comparison."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "train_detect.py"


def _load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _run_train(cmd, cwd: Path):
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(cwd))
    return proc.returncode, time.time() - t0


def _add_common_args(cmd, args):
    if args.fast:
        cmd.append("--fast")
    if args.subset is not None:
        cmd.extend(["--subset", str(args.subset)])
    if args.epochs is not None:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.lr is not None:
        cmd.extend(["--lr", str(args.lr)])
    if args.num_queries is not None:
        cmd.extend(["--num-queries", str(args.num_queries)])
    if args.decoder_layers is not None:
        cmd.extend(["--decoder-layers", str(args.decoder_layers)])
    if args.experiments_root is not None:
        cmd.extend(["--experiments-root", args.experiments_root])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clip-init", type=str, required=True, help="checkpoint path for --clip-init run")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--subset", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num-queries", type=int, default=None)
    p.add_argument("--decoder-layers", type=int, default=None)
    p.add_argument("--experiments-root", type=str, default="experiments")
    p.add_argument("--base-tag", type=str, default="clipcmp")
    p.add_argument("--extra", type=str, default="", help="extra args passed to both runs")
    args = p.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.experiments_root) / "comparisons" / f"{args.base_tag}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    extra_args = shlex.split(args.extra) if args.extra.strip() else []

    scratch_summary = out_dir / "scratch_summary.json"
    clip_summary = out_dir / "clip_init_summary.json"

    scratch_cmd = [sys.executable, str(TRAIN_SCRIPT)]
    _add_common_args(scratch_cmd, args)
    scratch_cmd.extend(
        [
            "--tag",
            f"{args.base_tag}_scratch",
            "--summary-out",
            str(scratch_summary),
        ]
    )
    scratch_cmd.extend(extra_args)

    clip_cmd = [sys.executable, str(TRAIN_SCRIPT)]
    _add_common_args(clip_cmd, args)
    clip_cmd.extend(
        [
            "--clip-init",
            args.clip_init,
            "--tag",
            f"{args.base_tag}_clipinit",
            "--summary-out",
            str(clip_summary),
        ]
    )
    clip_cmd.extend(extra_args)

    print("Running scratch baseline...")
    print(" ".join(scratch_cmd))
    scratch_rc, scratch_time = _run_train(scratch_cmd, cwd=PROJECT_ROOT.parent)

    print("Running clip-init experiment...")
    print(" ".join(clip_cmd))
    clip_rc, clip_time = _run_train(clip_cmd, cwd=PROJECT_ROOT.parent)

    scratch = _load_json(scratch_summary)
    clip = _load_json(clip_summary)

    scratch_map = scratch.get("best_map") if scratch else None
    clip_map = clip.get("best_map") if clip else None
    scratch_loss = scratch.get("best_val_loss") if scratch else None
    clip_loss = clip.get("best_val_loss") if clip else None

    delta_map = None
    if scratch_map is not None and clip_map is not None:
        delta_map = float(clip_map) - float(scratch_map)

    delta_loss = None
    if scratch_loss is not None and clip_loss is not None:
        delta_loss = float(clip_loss) - float(scratch_loss)

    winner = None
    if delta_map is not None:
        if delta_map > 0:
            winner = "clip_init"
        elif delta_map < 0:
            winner = "scratch"
        else:
            winner = "tie"

    report = {
        "scratch_return_code": scratch_rc,
        "clip_init_return_code": clip_rc,
        "scratch_duration_s": scratch_time,
        "clip_init_duration_s": clip_time,
        "scratch_summary_path": str(scratch_summary),
        "clip_init_summary_path": str(clip_summary),
        "scratch_best_map": scratch_map,
        "clip_init_best_map": clip_map,
        "delta_best_map_clip_minus_scratch": delta_map,
        "scratch_best_val_loss": scratch_loss,
        "clip_init_best_val_loss": clip_loss,
        "delta_best_val_loss_clip_minus_scratch": delta_loss,
        "winner_by_best_map": winner,
    }

    report_path = out_dir / "comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(f"Comparison report: {report_path}")
    print(
        "Comparison summary: "
        f"scratch_map={scratch_map} | clip_init_map={clip_map} | "
        f"delta={delta_map}"
    )


if __name__ == "__main__":
    main()
