"""Shared results schema so every member's runs land in comparable files.

results/<track>/summary.csv         one row per (run_id, method, params, gold_pos)  -> committed
results/<track>/<run_id>.jsonl      one row per example (no contexts)               -> committed
data/outputs/<run_id>.jsonl         compressed contexts for error analysis          -> not committed
"""

import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path

from effrag.config import RESULTS_DIR

SUMMARY_FIELDS = [
    "run_id", "date", "git", "track", "dataset", "method", "params", "gold_pos", "n", "seed",
    "llm_model", "ctx_recall", "accuracy", "em", "f1", "orig_tokens", "comp_tokens", "ratio",
    "compress_ms", "llm_ms", "cost_usd_per_1k_q",
]  # fmt: skip


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "nogit"


def new_run_id(method: str) -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}-{method}"


def append_summary(track: str, row: dict) -> Path:
    path = RESULTS_DIR / track / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in SUMMARY_FIELDS})
    return path


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
