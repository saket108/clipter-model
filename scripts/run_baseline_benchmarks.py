#!/usr/bin/env python3
"""Run unified baseline benchmarks and write one comparable report.

Supports CLIPTER / YOLO / DETR style runs:
1) pass only metrics paths (already-evaluated runs), or
2) pass command + metrics path (this script runs command then parses metrics).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CANON_COLUMNS = [
    "date_utc",
    "git_commit",
    "model",
    "dataset",
    "split",
    "map50",
    "map5095",
    "map75",
    "latency_ms_per_image",
    "runtime_sec",
    "exit_code",
    "notes",
]

MAP50_KEYS = [
    "map50",
    "metrics/map50",
    "mAP@0.5",
    "metrics/mAP50(B)",
    "PascalBoxes_Precision/mAP@0.5IOU",
]
MAP5095_KEYS = [
    "map",
    "metrics/map",
    "mAP@0.5:0.95",
    "metrics/mAP50-95(B)",
    "PascalBoxes_Precision/mAP@0.5:0.95IOU",
]
MAP75_KEYS = [
    "map75",
    "metrics/map75",
    "mAP@0.75",
    "metrics/mAP75(B)",
]
LATENCY_KEYS = [
    "latency_ms_per_image",
    "metrics/latency_ms_per_image",
    "latency",
    "inference_latency_ms",
]


@dataclass
class BaselineSpec:
    model: str
    cmd: str
    metrics: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_commit(root: Path) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True)
        return out.strip()
    except Exception:
        return ""


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key))
        else:
            out[key] = v
    return out


def _first_from_keys(payload: dict[str, Any], keys: list[str]) -> float | None:
    lowered = {str(k).lower(): v for k, v in payload.items()}
    for key in keys:
        if key in payload:
            f = _safe_float(payload[key])
            if f is not None:
                return f
        v = lowered.get(key.lower())
        f = _safe_float(v)
        if f is not None:
            return f
    return None


def _parse_log_like(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    kv = re.findall(r"'([^']+)':\s*([^,\n}]+)", text)
    out: dict[str, Any] = {}
    for key, raw in kv:
        value = raw.strip()
        if value.startswith("np.float64(") and value.endswith(")"):
            value = value[len("np.float64(") : -1]
        out[key] = _safe_float(value)
    return out


def _parse_metrics(path: Path) -> tuple[float | None, float | None, float | None, float | None, str]:
    if not path.exists():
        return None, None, None, None, f"missing metrics file: {path}"

    try:
        ext = path.suffix.lower()
        payload: dict[str, Any]
        if ext == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return None, None, None, None, "json payload is not object"
            payload = _flatten_dict(raw)
        elif ext == ".csv":
            with open(path, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                return None, None, None, None, "csv has no rows"
            payload = rows[-1]
        else:
            payload = _parse_log_like(path)
    except Exception as exc:
        return None, None, None, None, f"parse error: {exc}"

    map50 = _first_from_keys(payload, MAP50_KEYS)
    map5095 = _first_from_keys(payload, MAP5095_KEYS)
    map75 = _first_from_keys(payload, MAP75_KEYS)
    latency = _first_from_keys(payload, LATENCY_KEYS)
    return map50, map5095, map75, latency, "ok"


def _run_cmd(cmd: str, cwd: Path, dry_run: bool) -> int:
    print(">>", cmd)
    if dry_run or not cmd.strip():
        return 0
    proc = subprocess.run(cmd, cwd=str(cwd), shell=True, check=False)
    return int(proc.returncode)


def _append_rows(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CANON_COLUMNS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CANON_COLUMNS})


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified baseline benchmark runner for CLIPTER.")
    p.add_argument("--dataset", default="unknown_dataset")
    p.add_argument("--split", default="valid")
    p.add_argument("--summary-csv", default="reports/baseline_benchmarks.csv")
    p.add_argument("--output-json", default="reports/baseline_benchmarks.json")
    p.add_argument("--dry-run", action="store_true")

    p.add_argument("--clipter-cmd", default="")
    p.add_argument("--clipter-metrics", default="")
    p.add_argument("--yolo-cmd", default="")
    p.add_argument("--yolo-metrics", default="")
    p.add_argument("--detr-cmd", default="")
    p.add_argument("--detr-metrics", default="")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    root = _repo_root()
    timestamp = _now_iso()
    commit = _git_commit(root)

    specs = [
        BaselineSpec("CLIPTER", args.clipter_cmd.strip(), args.clipter_metrics.strip()),
        BaselineSpec("YOLO", args.yolo_cmd.strip(), args.yolo_metrics.strip()),
        BaselineSpec("DETR", args.detr_cmd.strip(), args.detr_metrics.strip()),
    ]

    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for spec in specs:
        if not spec.cmd and not spec.metrics:
            continue

        started = time.time()
        rc = _run_cmd(spec.cmd, cwd=root, dry_run=args.dry_run)
        runtime_sec = time.time() - started

        map50, map5095, map75, latency, status = (None, None, None, None, "no metrics path")
        if spec.metrics:
            map50, map5095, map75, latency, status = _parse_metrics(Path(spec.metrics))

        row = {
            "date_utc": timestamp,
            "git_commit": commit,
            "model": spec.model,
            "dataset": args.dataset,
            "split": args.split,
            "map50": map50,
            "map5095": map5095,
            "map75": map75,
            "latency_ms_per_image": latency,
            "runtime_sec": float(runtime_sec),
            "exit_code": int(rc),
            "notes": f"metrics_status={status};metrics_path={spec.metrics};cmd={shlex.quote(spec.cmd) if spec.cmd else ''}",
        }
        rows.append(row)
        details.append({"model": spec.model, "runtime_sec": runtime_sec, "exit_code": rc, "metrics_status": status})

    if not args.dry_run and rows:
        summary_path = Path(args.summary_csv)
        json_path = Path(args.output_json)
        _append_rows(summary_path, rows)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(
                {
                    "created_at_utc": timestamp,
                    "git_commit": commit,
                    "dataset": args.dataset,
                    "split": args.split,
                    "rows": rows,
                    "details": details,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print(f"Rows prepared: {len(rows)}")
    print(f"Summary CSV: {args.summary_csv}")
    print(f"Run JSON   : {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
