"""Grid sweep runner for detector hyperparameters."""
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "train_detect.py"


def _parse_list(raw: str, cast_fn):
    vals = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(cast_fn(token))
    if len(vals) == 0:
        raise ValueError(f"Invalid empty list from: {raw}")
    return vals


def _tag_value(v) -> str:
    s = str(v)
    return s.replace(".", "p").replace("-", "m")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lrs", type=str, default="1e-4,5e-5")
    p.add_argument("--batch-sizes", type=str, default="8,16")
    p.add_argument("--num-queries", type=str, default="50,100")
    p.add_argument("--decoder-layers", type=str, default="2,3")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--subset", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clip-init", type=str, default=None)
    p.add_argument("--experiments-root", type=str, default="experiments")
    p.add_argument("--base-tag", type=str, default="sweep")
    p.add_argument("--stop-on-error", action="store_true")
    p.add_argument("--max-runs", type=int, default=None)
    p.add_argument("--extra", type=str, default="", help="extra args passed to each train run")
    args = p.parse_args()

    lrs = _parse_list(args.lrs, float)
    batch_sizes = _parse_list(args.batch_sizes, int)
    num_queries = _parse_list(args.num_queries, int)
    decoder_layers = _parse_list(args.decoder_layers, int)

    combos = list(product(lrs, batch_sizes, num_queries, decoder_layers))
    if args.max_runs is not None:
        combos = combos[: args.max_runs]

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(args.experiments_root) / "sweeps" / f"{args.base_tag}_{ts}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sweep runs: {len(combos)}")
    print(f"Sweep output dir: {sweep_dir}")

    extra_args = shlex.split(args.extra) if args.extra.strip() else []
    results = []

    for idx, (lr, bs, nq, nl) in enumerate(combos, start=1):
        tag = (
            f"{args.base_tag}_lr{_tag_value(lr)}"
            f"_bs{_tag_value(bs)}_q{_tag_value(nq)}_l{_tag_value(nl)}"
        )
        summary_out = sweep_dir / f"run_{idx:03d}_{tag}.json"

        cmd = [sys.executable, str(TRAIN_SCRIPT)]
        if args.fast:
            cmd.append("--fast")
        if args.subset is not None:
            cmd.extend(["--subset", str(args.subset)])
        if args.epochs is not None:
            cmd.extend(["--epochs", str(args.epochs)])
        if args.seed is not None:
            cmd.extend(["--seed", str(args.seed)])
        if args.clip_init:
            cmd.extend(["--clip-init", args.clip_init])

        cmd.extend(
            [
                "--lr",
                str(lr),
                "--batch-size",
                str(bs),
                "--num-queries",
                str(nq),
                "--decoder-layers",
                str(nl),
                "--experiments-root",
                args.experiments_root,
                "--tag",
                tag,
                "--summary-out",
                str(summary_out),
            ]
        )
        cmd.extend(extra_args)

        print(f"[{idx}/{len(combos)}] Running: {' '.join(cmd)}")
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT.parent))
        duration_s = time.time() - t0

        record = {
            "idx": idx,
            "lr": lr,
            "batch_size": bs,
            "num_queries": nq,
            "decoder_layers": nl,
            "tag": tag,
            "return_code": proc.returncode,
            "duration_s": duration_s,
            "summary_path": str(summary_out),
        }

        if summary_out.exists():
            try:
                with open(summary_out, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                record["best_map"] = summary.get("best_map")
                record["best_val_loss"] = summary.get("best_val_loss")
            except Exception as e:
                record["summary_load_error"] = str(e)

        results.append(record)

        if proc.returncode != 0 and args.stop_on_error:
            print("Stopping sweep due to run failure.")
            break

    json_path = sweep_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    csv_path = sweep_dir / "results.csv"
    if len(results) > 0:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    ok = [r for r in results if r.get("return_code", 1) == 0]
    print(f"Completed runs: {len(results)} | success: {len(ok)}")
    print(f"Sweep JSON: {json_path}")
    print(f"Sweep CSV: {csv_path}")


if __name__ == "__main__":
    main()
