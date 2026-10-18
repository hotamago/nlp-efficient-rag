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


def record_run(rows: list[dict], *, run_id: str, method: str, params: dict, gold_pos: int,
               llm_model: str, seed: int, track: str = "compression") -> dict:  # fmt: skip
    """Write per-question rows and append the summary line for one (config, gold position).

    Rows need: accuracy, em, f1, orig_tokens, comp_tokens, compress_s, llm_s (ctx_recall optional).
    Used by scripts that do not go through run_compression.py (soft compression, PISCO, ...).
    """
    import numpy as np

    write_jsonl(RESULTS_DIR / track / f"{run_id}_gold{gold_pos}.jsonl", rows)

    def mean(key: str) -> float:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    comp = mean("comp_tokens")
    summary = {
        "run_id": run_id, "date": f"{datetime.now():%Y-%m-%d %H:%M}", "git": git_commit(),
        "track": track, "dataset": "litm_nq20", "method": method,
        "params": json.dumps(params, sort_keys=True), "gold_pos": gold_pos, "n": len(rows),
        "seed": seed, "llm_model": llm_model, "ctx_recall": round(mean("ctx_recall"), 4),
        "accuracy": round(mean("accuracy"), 4), "em": round(mean("em"), 4),
        "f1": round(mean("f1"), 4), "orig_tokens": round(mean("orig_tokens"), 1),
        "comp_tokens": round(comp, 1), "ratio": round(mean("orig_tokens") / max(comp, 1), 2),
        "compress_ms": round(1000 * mean("compress_s"), 1),
        "llm_ms": round(1000 * mean("llm_s"), 1), "cost_usd_per_1k_q": float("nan"),
    }  # fmt: skip
    append_summary(track, summary)
    print(f"{method} {params} gold@{gold_pos}: acc={summary['accuracy']} ratio={summary['ratio']}")
    return summary
