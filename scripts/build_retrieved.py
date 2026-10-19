"""Track C: retrieve top-k passages from the pooled corpus for the 500 evaluation questions.

Writes {qid: [corpus ids in rank order]} to data/cache/retrieved/, read by
`effrag.data.load_retrieved` and `run_compression.py --retriever`.

  uv run python scripts/build_retrieved.py --system bge-small-int8
"""

import argparse
import json

from effrag.config import SEED
from effrag.data import litm_corpus, load_litm_nq, retrieved_path
from effrag.metrics import best_subspan_em

BGE_Q = "Represent this sentence for searching relevant passages: "
SYSTEMS = {
    "bge-small-int8": {"model": "BAAI/bge-small-en-v1.5", "query_prompt": BGE_Q, "quant": "int8"},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="bge-small-int8", choices=list(SYSTEMS))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()

    from effrag.retrieval import DenseRetriever

    corpus = litm_corpus(0)
    examples = load_litm_nq(0, n=args.n, seed=args.seed)
    r = DenseRetriever(corpus, **SYSTEMS[args.system])
    hits = r.search([e.question for e in examples], k=args.k)
    path = retrieved_path(args.system, args.n, args.seed, args.k)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({e.qid: h for e, h in zip(examples, hits, strict=True)}))
    recall = sum(
        any(best_subspan_em(corpus[i].text, e.answers) for i in h)
        for e, h in zip(examples, hits, strict=True)
    ) / len(examples)
    print(f"{path}: answer_hit@{args.k}={recall:.3f}")


if __name__ == "__main__":
    main()
