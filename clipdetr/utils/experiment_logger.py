"""Simple experiment logging helpers for detector training runs."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


def make_run_id(prefix: str = "detect", tag: Optional[str] = None) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{prefix}_{ts}"
    if tag:
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in tag.strip())
        if safe:
            run_id = f"{run_id}_{safe}"
    return run_id


class ExperimentLogger:
    def __init__(self, root_dir: str = "experiments", run_id: Optional[str] = None):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)

        self.run_id = run_id if run_id is not None else make_run_id()
        self.run_dir = self.root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.config_path = self.run_dir / "config.json"
        self.epoch_csv_path = self.run_dir / "epoch_metrics.csv"
        self.summary_path = self.run_dir / "summary.json"
        self.runs_jsonl_path = self.root / "detect_runs.jsonl"

        self._epoch_fp = None
        self._epoch_writer = None

    @staticmethod
    def _with_timestamp(data: Dict[str, object]) -> Dict[str, object]:
        out = dict(data)
        out["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        return out

    def log_config(self, config: Dict[str, object]) -> None:
        payload = self._with_timestamp(config)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def log_epoch(self, row: Dict[str, object]) -> None:
        if self._epoch_writer is None:
            self._epoch_fp = open(self.epoch_csv_path, "w", encoding="utf-8", newline="")
            fieldnames = list(row.keys())
            self._epoch_writer = csv.DictWriter(self._epoch_fp, fieldnames=fieldnames)
            self._epoch_writer.writeheader()

        self._epoch_writer.writerow(row)
        self._epoch_fp.flush()

    def write_summary(self, summary: Dict[str, object], summary_out: Optional[str] = None) -> Path:
        payload = self._with_timestamp({"run_id": self.run_id, **summary})
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

        with open(self.runs_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")

        if summary_out:
            summary_out_path = Path(summary_out)
            summary_out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            return summary_out_path
        return self.summary_path

    def close(self) -> None:
        if self._epoch_fp is not None:
            self._epoch_fp.close()
            self._epoch_fp = None
            self._epoch_writer = None

