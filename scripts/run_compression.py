"""Track A: context compression on Lost-in-the-Middle NQ (20 docs).

Examples:
  uv run python scripts/run_compression.py --method full --dry-run --n 20
  uv run python scripts/run_compression.py --method llmlingua2 --params '{"rate": 0.25}'
  uv run python scripts/run_compression.py --method provence --params '{"threshold": 0.1}' \
      --positions 0 9 19
"""

import argparse
import json
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
    ap.add_argument("--dry-run", action="store_true", help="compress only, skip the LLM")
    args = ap.parse_args()

    params = json.loads(args.params)
    comp = get_compressor(args.method, **params)
    settings = LLMSettings()
    llm = None
    if not args.dry_run:
        from effrag.generation import LLM

        llm = LLM(settings)
    run_id = new_run_id(args.method)
    track = "compression_dryrun" if args.dry_run else "compression"

    for pos in args.positions:
        examples = load_litm_nq(pos, n=args.n, seed=args.seed)
        rows, contexts = [], []
        for ex in tqdm(examples, desc=f"{args.method} gold@{pos}"):
            res = comp.compress(ex.question, ex.docs)
            row = {
                "run_id": run_id, "qid": ex.qid, "gold_pos": pos, "question": ex.question,
                "answers": ex.answers, "orig_tokens": res.orig_tokens,
                "comp_tokens": res.comp_tokens, "compress_s": res.latency_s,
                "ctx_recall": best_subspan_em(res.context, ex.answers),
            }  # fmt: skip
            if llm is not None:
                from effrag.generation import build_prompt

                out = llm.generate(build_prompt(ex.question, res.context))
                pred = out["text"]
                row |= {
                    "prediction": pred, "llm_s": out["latency_s"],
                    "accuracy": best_subspan_em(pred, ex.answers),
                    "em": exact_match(pred, ex.answers), "f1": f1(pred, ex.answers),
                }  # fmt: skip
            rows.append(row)
            contexts.append({"qid": ex.qid, "gold_pos": pos, "context": res.context})

        write_jsonl(RESULTS_DIR / track / f"{run_id}_gold{pos}.jsonl", rows)
        write_jsonl(DATA_DIR / "outputs" / f"{run_id}_gold{pos}.jsonl", contexts)

        def mean(key: str, rows: list[dict] = rows) -> float:
            vals = [r[key] for r in rows if r.get(key) is not None]
            return float(np.mean(vals)) if vals else float("nan")

        comp_tok = mean("comp_tokens")
        summary = {
            "run_id": run_id, "date": f"{datetime.now():%Y-%m-%d %H:%M}", "git": git_commit(),
            "track": track, "dataset": "litm_nq20", "method": comp.name,
            "params": json.dumps(comp.params(), sort_keys=True), "gold_pos": pos,
            "n": len(rows), "seed": args.seed,
            "llm_model": "" if args.dry_run else settings.model,
            "ctx_recall": round(mean("ctx_recall"), 4), "accuracy": round(mean("accuracy"), 4),
            "em": round(mean("em"), 4), "f1": round(mean("f1"), 4),
            "orig_tokens": round(mean("orig_tokens"), 1),
            "comp_tokens": round(comp_tok, 1),
            "ratio": round(mean("orig_tokens") / max(comp_tok, 1), 2),
            "compress_ms": round(1000 * mean("compress_s"), 1),
            "llm_ms": round(1000 * mean("llm_s"), 1),
            "cost_usd_per_1k_q": round(comp_tok * settings.price_in / 1000, 4),
        }  # fmt: skip
        path = append_summary(track, summary)
        print(f"gold@{pos}: acc={summary['accuracy']} ratio={summary['ratio']} -> {path}")


if __name__ == "__main__":
    main()
