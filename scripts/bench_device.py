"""Compressor latency on CPU vs GPU (small sample, no LLM).

Answers "can this run without a GPU?": every compressor is timed on the same questions
after one warm-up call. Select the device with EFFRAG_DEVICE=cpu|cuda.

    EFFRAG_DEVICE=cpu uv run python scripts/bench_device.py --n 20
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from effrag.compression import get_compressor
from effrag.config import RESULTS_DIR, SEED, device
from effrag.data import load_litm_nq

OP = "hotchpotch/open-provence-reranker-v1-gte-modernbert-base"
CONFIGS = [
    ("bm25_topk", {"k": 5}),
    ("dense_topk", {"k": 5}),
    ("rerank_topk", {"k": 5}),
    ("provence", {"threshold": 0.1, "model": OP}),
    ("provence", {"threshold": 0.1}),
    ("llmlingua2", {"rate": 0.5}),
    ("longllmlingua", {"target_token": 730}),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--methods", nargs="*", help="subset of method names")
    args = ap.parse_args()

    dev = device()
    examples = load_litm_nq(9, n=args.n + 1, seed=SEED)
    out = RESULTS_DIR / "latency" / f"{dev}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    for method, params in CONFIGS:
        if args.methods and method not in args.methods:
            continue
        comp = get_compressor(method, **params)
        comp.compress(examples[0].question, examples[0].docs)  # warm-up (lazy loading, CUDA init)
        times = []
        for ex in examples[1:]:
            t0 = time.perf_counter()
            comp.compress(ex.question, ex.docs)
            times.append(time.perf_counter() - t0)
        row = {
            "device": dev, "threads": torch.get_num_threads(), "method": method,
            "params": json.dumps(params, sort_keys=True), "n": len(times),
            "mean_ms": 1000 * statistics.mean(times), "median_ms": 1000 * statistics.median(times),
        }  # fmt: skip
        print(json.dumps(row))
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        del comp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
