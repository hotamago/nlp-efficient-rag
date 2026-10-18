"""Track B: efficient retrieval over the pooled Lost-in-the-Middle corpus (~50.7k passages).

Queries are the same seeded 500 questions as Track A. Metrics:
- answer_hit@k: a top-k passage contains a gold answer string (DPR-style top-k accuracy)
- gold_mrr@10: reciprocal rank of the annotated gold passage
- index_mb, build_s (index construction from cached embeddings), encode_s (corpus encoding),
  query_ms (batched; per-query average) and query_ms_single (p50 over 50 one-by-one queries)

  uv run python scripts/run_retrieval.py --systems bm25 bge-small qwen3-256
"""

import argparse
import json
import time
from datetime import datetime

import numpy as np

from effrag.config import RESULTS_DIR, SEED
from effrag.data import litm_corpus, load_litm_nq
from effrag.metrics import best_subspan_em
from effrag.results import git_commit

QWEN_Q = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
)
BGE_Q = "Represent this sentence for searching relevant passages: "
QWEN = "Qwen/Qwen3-Embedding-0.6B"
STATIC = "sentence-transformers/static-retrieval-mrl-en-v1"
BGE = "BAAI/bge-small-en-v1.5"

SYSTEMS = {
    "bm25": ("bm25", {}),
    "static-1024": ("dense", {"model": STATIC}),
    "static-256": ("dense", {"model": STATIC, "dim": 256}),
    "bge-small": ("dense", {"model": BGE, "query_prompt": BGE_Q}),
    "bge-small-hnsw": ("dense", {"model": BGE, "query_prompt": BGE_Q, "index": "hnsw"}),
    "bge-small-int8": ("dense", {"model": BGE, "query_prompt": BGE_Q, "quant": "int8"}),
    "bge-small-binary": ("dense", {"model": BGE, "query_prompt": BGE_Q, "quant": "binary"}),
    "qwen3-1024": ("dense", {"model": QWEN, "query_prompt": QWEN_Q}),
    "qwen3-256": ("dense", {"model": QWEN, "query_prompt": QWEN_Q, "dim": 256}),
    "qwen3-64": ("dense", {"model": QWEN, "query_prompt": QWEN_Q, "dim": 64}),
    "qwen3-1024-int8": ("dense", {"model": QWEN, "query_prompt": QWEN_Q, "quant": "int8"}),
    "qwen3-1024-binary": ("dense", {"model": QWEN, "query_prompt": QWEN_Q, "quant": "binary"}),
    "qwen3-256-binary": ("dense", {"model": QWEN, "query_prompt": QWEN_Q, "dim": 256,
                                   "quant": "binary"}),
}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    # Qwen3-Embedding variants are defined but not run by default (encoding 50k passages with a
    # 0.6B model is the most expensive part of Track B).
    default = [s for s in SYSTEMS if not s.startswith("qwen3")]
    ap.add_argument("--systems", nargs="+", default=default)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    from effrag.retrieval import BM25Retriever, DenseRetriever

    corpus = litm_corpus(0)
    examples = load_litm_nq(0, n=args.n, seed=args.seed)
    queries = [e.question for e in examples]
    key_to_id = {(d.title, d.text): i for i, d in enumerate(corpus)}
    gold_ids = [key_to_id[next((d.title, d.text) for d in e.docs if d.is_gold)] for e in examples]
    print(f"corpus={len(corpus)} queries={len(queries)}")

    out_path = RESULTS_DIR / "retrieval" / "summary.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for name in args.systems:
        kind, kw = SYSTEMS[name]
        r = BM25Retriever(corpus) if kind == "bm25" else DenseRetriever(corpus, **kw)
        r.search(queries[:8], k=20)  # warm-up
        t0 = time.perf_counter()
        hits = r.search(queries, k=20)
        query_ms = 1000 * (time.perf_counter() - t0) / len(queries)
        single = []
        for q in queries[:50]:
            t1 = time.perf_counter()
            r.search([q], k=20)
            single.append(1000 * (time.perf_counter() - t1))

        def answer_hit(k: int, hits: list[list[int]] = hits) -> float:
            return float(np.mean([
                any(best_subspan_em(corpus[i].text, e.answers) for i in h[:k])
                for e, h in zip(examples, hits, strict=True)
            ]))  # fmt: skip

        mrr = np.mean(
            [
                1 / (h[:10].index(g) + 1) if g in h[:10] else 0
                for g, h in zip(gold_ids, hits, strict=True)
            ]
        )
        row = {
            "date": f"{datetime.now():%Y-%m-%d %H:%M}", "git": git_commit(), "system": name,
            "params": json.dumps(kw, sort_keys=True), "n": len(queries), "seed": args.seed,
            "corpus": len(corpus), "answer_hit@1": round(answer_hit(1), 4),
            "answer_hit@5": round(answer_hit(5), 4), "answer_hit@20": round(answer_hit(20), 4),
            "gold_mrr@10": round(float(mrr), 4), "index_mb": round(r.index_bytes() / 2**20, 2),
            "build_s": round(r.build_s, 2), "encode_s": round(getattr(r, "encode_s", 0.0), 1),
            "query_ms": round(query_ms, 3), "query_ms_single": round(float(np.median(single)), 3),
        }  # fmt: skip
        with out_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(row)


if __name__ == "__main__":
    main()
