"""Evaluate the soft compressor.

Gate (Tevatron dev, k=20, not LITM): base closed-book vs base full-text vs soft memory.
    uv run python scripts/eval_soft.py gate --run data/soft/run1
    uv run python scripts/eval_soft.py diag --run data/soft/run1   # locate the gate failure

LITM NQ-20 (same 500 questions / positions as Track A); writes results/compression/ rows with
llm_model "<base>+soft". Documents are encoded once per question and reused for every position
(encode-once); `hybrid` keeps the cross-encoder's top-1 document as text.
    uv run python scripts/eval_soft.py litm --run data/soft/run1 --variants soft hybrid
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from effrag.config import DATA_DIR, SEED, limit_gpu_memory
from effrag.data import load_litm_nq
from effrag.metrics import best_subspan_em, exact_match, f1
from effrag.results import new_run_id, record_run
from effrag.soft.model import SoftRAG
from effrag.tokens import count_tokens


def gate(model: SoftRAG, run: str, n: int) -> None:
    data = [json.loads(x) for x in open(DATA_DIR / "soft" / "dev.jsonl")][:n]
    acc = {"closedbook": [], "full_text": [], "soft": []}
    for ex in tqdm(data, desc="gate"):
        docs = [tuple(d) for d in ex["docs"]]
        q, ans = ex["question"], ex["answers"]
        acc["closedbook"].append(best_subspan_em(model.generate(q, [], base=True), ans))
        acc["full_text"].append(best_subspan_em(model.generate(q, docs, base=True), ans))
        acc["soft"].append(best_subspan_em(model.generate(q, docs), ans))
    res = {k: round(100 * float(np.mean(v)), 1) for k, v in acc.items()}
    ok = res["soft"] >= res["closedbook"] + 10 and res["soft"] >= 0.7 * res["full_text"]
    res |= {"n": len(data), "pass": bool(ok)}
    print(json.dumps(res))
    (Path(run) / "gate.json").write_text(json.dumps(res))


def diag(model: SoftRAG, run: str, n: int) -> None:
    """Locate the gate failure on Tevatron dev: is the gold passage lost in its 8 memory vectors
    (gold-only soft vs gold-only text), or among the 19 compressed negatives (hybrid, gold open)?"""
    data = [json.loads(x) for x in open(DATA_DIR / "soft" / "dev.jsonl")][:n]
    acc = {"soft_gold_only": [], "text_gold_only": [], "hybrid_gold_open": [], "soft_20": []}
    for ex in tqdm(data, desc="diag"):
        docs = [tuple(d) for d in ex["docs"]]
        q, ans, g = ex["question"], ex["answers"], ex["gold"]
        gold_open = [j == g for j in range(len(docs))]
        acc["soft_gold_only"].append(best_subspan_em(model.generate(q, [docs[g]]), ans))
        acc["text_gold_only"].append(best_subspan_em(model.generate(q, [docs[g]], base=True), ans))
        acc["hybrid_gold_open"].append(
            best_subspan_em(model.generate(q, docs, open_mask=gold_open), ans)
        )
        acc["soft_20"].append(best_subspan_em(model.generate(q, docs), ans))
    res = {k: round(100 * float(np.mean(v)), 1) for k, v in acc.items()} | {"n": len(data)}
    print(json.dumps(res))
    (Path(run) / "diag.json").write_text(json.dumps(res))


def litm(model: SoftRAG, run: str, variants: list[str], positions: list[int], n: int) -> None:
    reranker = None
    if "hybrid" in variants:
        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device="cuda")
    mem_cache: dict[tuple[str, str], torch.Tensor] = {}
    base = model.cfg.base_model.split("/")[-1].lower()
    for variant in variants:
        run_id = new_run_id(f"soft_{variant}")
        params = {"n_mem": model.cfg.n_mem, "variant": variant, "run": Path(run).name}
        for pos in positions:
            rows = []
            for ex in tqdm(load_litm_nq(pos, n=n, seed=SEED), desc=f"{variant} gold@{pos}"):
                docs = [(d.title, d.text) for d in ex.docs]
                open_mask = [False] * len(docs)
                if variant == "hybrid":
                    s = reranker.predict([(ex.question, d.render()) for d in ex.docs])
                    open_mask[int(np.argmax(s))] = True
                t0 = time.perf_counter()
                todo = [d for d, o in zip(docs, open_mask, strict=True) if not o]
                new = [d for d in todo if d not in mem_cache]
                if new:
                    with torch.no_grad():
                        for d, m in zip(new, model.encode_docs(new), strict=True):
                            mem_cache[d] = m
                compress_s = time.perf_counter() - t0
                mem = torch.stack([mem_cache[d] for d in todo]) if todo else None
                t1 = time.perf_counter()
                pred = model.generate(ex.question, docs, mem, open_mask)
                llm_s = time.perf_counter() - t1
                full = "\n".join(f"Document [{i}]{d.render()}" for i, d in enumerate(ex.docs, 1))
                # decoder context positions: titles/open text as tokens + n_mem vectors per doc
                kept = "\n".join(
                    f"Document [{i}](Title: {t}) " + (x if o else "")
                    for i, ((t, x), o) in enumerate(zip(docs, open_mask, strict=True), 1)
                )
                comp = count_tokens(kept) + model.cfg.n_mem * len(todo)
                rows.append({
                    "run_id": run_id, "qid": ex.qid, "gold_pos": pos, "question": ex.question,
                    "answers": ex.answers, "orig_tokens": count_tokens(full), "comp_tokens": comp,
                    "compress_s": compress_s, "ctx_recall": None, "prediction": pred,
                    "llm_s": llm_s, "prompt_tokens": None,
                    "accuracy": best_subspan_em(pred, ex.answers),
                    "em": exact_match(pred, ex.answers), "f1": f1(pred, ex.answers),
                })  # fmt: skip
            record_run(rows, run_id=run_id, method="soft", params=params, gold_pos=pos,
                       llm_model=f"{base}+soft", seed=SEED)  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gate", "diag", "litm"])
    ap.add_argument("--run", default=str(DATA_DIR / "soft" / "run1"))
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--variants", nargs="+", default=["soft", "hybrid"])
    ap.add_argument("--positions", type=int, nargs="+", default=[0, 9, 19])
    args = ap.parse_args()
    limit_gpu_memory()
    model = SoftRAG.load(args.run).eval()
    with torch.inference_mode():
        if args.mode == "gate":
            gate(model, args.run, args.n or 200)
        elif args.mode == "diag":
            diag(model, args.run, args.n or 200)
        else:
            litm(model, args.run, args.variants, args.positions, args.n or 500)


if __name__ == "__main__":
    main()
