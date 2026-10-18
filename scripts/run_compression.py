"""Track A: context compression on Lost-in-the-Middle NQ (20 docs).

Phase 1 compresses every (question, gold position); phase 2 queries the LLM concurrently.

Examples:
  uv run python scripts/run_compression.py --method full --dry-run --n 20
  uv run python scripts/run_compression.py --method llmlingua2 --params '{"rate": 0.25}'
  uv run python scripts/run_compression.py --method provence --params '{"threshold": 0.1}' \
      --positions 0 9 19
"""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
from tqdm import tqdm

from effrag.compression import get_compressor
from effrag.config import DATA_DIR, RESULTS_DIR, SEED, LLMSettings
from effrag.data import load_litm_nq
from effrag.metrics import best_subspan_em, exact_match, f1
from effrag.results import append_summary, git_commit, new_run_id, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--params", default="{}", help="JSON kwargs for the compressor")
    ap.add_argument("--positions", type=int, nargs="+", default=[0, 4, 9, 14, 19])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--workers", type=int, default=32, help="concurrent LLM requests")
    ap.add_argument("--dry-run", action="store_true", help="compress only, skip the LLM")
    args = ap.parse_args()

    user_params = json.loads(args.params)
    settings = LLMSettings()
    run_id = new_run_id(args.method)
    track = "compression_dryrun" if args.dry_run else "compression"

    # Phase 1: compression (cached on disk, so LLM runs can reuse a previous --dry-run).
    # The compressor model is only loaded if some position is not cached yet.
    user_key = json.dumps(user_params, sort_keys=True)
    cache_dir = (
        DATA_DIR / "cache" / "compressed"
        / f"{args.method}_{hashlib.md5(user_key.encode()).hexdigest()[:10]}"
    )  # fmt: skip
    meta_path = cache_dir / "meta.json"
    comp = None
    rows_by_pos: dict[int, list[dict]] = {}
    for pos in args.positions:
        cache = cache_dir / f"gold{pos}_n{args.n}_seed{args.seed}.jsonl"
        if cache.exists():
            rows = [json.loads(line) | {"run_id": run_id} for line in cache.open()]
            rows_by_pos[pos] = rows
            print(f"gold@{pos}: loaded {len(rows)} compressed contexts from {cache}")
            continue
        if comp is None:
            comp = get_compressor(args.method, **user_params)
            cache_dir.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps({"name": comp.name, "params": comp.params()}))
        rows = []
        for ex in tqdm(load_litm_nq(pos, n=args.n, seed=args.seed), desc=f"compress gold@{pos}"):
            res = comp.compress(ex.question, ex.docs)
            rows.append({
                "run_id": run_id, "qid": ex.qid, "gold_pos": pos, "question": ex.question,
                "answers": ex.answers, "orig_tokens": res.orig_tokens,
                "comp_tokens": res.comp_tokens, "compress_s": res.latency_s,
                "ctx_recall": best_subspan_em(res.context, ex.answers), "context": res.context,
            })  # fmt: skip
        rows_by_pos[pos] = rows
        write_jsonl(cache, rows)

    # Phase 2: generation
    if not args.dry_run:
        from effrag.generation import LLM, build_prompt

        llm = LLM(settings)
        flat = [r for rows in rows_by_pos.values() for r in rows]

        def run(r: dict) -> None:
            out = llm.generate(build_prompt(r["question"], r["context"]))
            pred = out["text"]
            r |= {
                "prediction": pred, "llm_s": out["latency_s"],
                "prompt_tokens": out["prompt_tokens"],
                "accuracy": best_subspan_em(pred, r["answers"]),
                "em": exact_match(pred, r["answers"]), "f1": f1(pred, r["answers"]),
            }  # fmt: skip

        with ThreadPoolExecutor(args.workers) as pool:
            list(tqdm(pool.map(run, flat), total=len(flat), desc="generate"))

    meta = json.loads(meta_path.read_text())
    params_key = json.dumps(meta["params"], sort_keys=True)
    for pos, rows in rows_by_pos.items():
        write_jsonl(
            RESULTS_DIR / track / f"{run_id}_gold{pos}.jsonl",
            [{k: v for k, v in r.items() if k != "context"} for r in rows],
        )

        def mean(key: str, rows: list[dict] = rows) -> float:
            vals = [r[key] for r in rows if r.get(key) is not None]
            return float(np.mean(vals)) if vals else float("nan")

        comp_tok = mean("comp_tokens")
        summary = {
            "run_id": run_id, "date": f"{datetime.now():%Y-%m-%d %H:%M}", "git": git_commit(),
            "track": track, "dataset": "litm_nq20", "method": meta["name"],
            "params": params_key, "gold_pos": pos,
            "n": len(rows), "seed": args.seed,
            "llm_model": "" if args.dry_run else settings.model,
            "ctx_recall": round(mean("ctx_recall"), 4), "accuracy": round(mean("accuracy"), 4),
            "em": round(mean("em"), 4), "f1": round(mean("f1"), 4),
            "orig_tokens": round(mean("orig_tokens"), 1), "comp_tokens": round(comp_tok, 1),
            "ratio": round(mean("orig_tokens") / max(comp_tok, 1), 2),
            "compress_ms": round(1000 * mean("compress_s"), 1),
            "llm_ms": round(1000 * mean("llm_s"), 1),
            "cost_usd_per_1k_q": round(comp_tok * settings.price_in / 1000, 4),
        }  # fmt: skip
        path = append_summary(track, summary)
        print(
            f"gold@{pos}: acc={summary['accuracy']} ctx_recall={summary['ctx_recall']} "
            f"ratio={summary['ratio']} -> {path}"
        )


if __name__ == "__main__":
    main()
