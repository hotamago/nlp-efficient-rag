"""Build training/dev data for the soft compressor from Tevatron/wikipedia-nq (DPR NQ).

Each example: question, answer, k in {5, 10, 20} passages (1 positive + BM25 hard negatives, padded
with random passages from other questions), gold at a random position, and an optional "open"
passage that stays as text (gold with p=0.6, else a negative) for the hybrid variant.
Questions that also appear in the Lost-in-the-Middle test set are removed.

    uv run python scripts/build_soft_train.py --n-train 15000 --n-dev 200
"""

from __future__ import annotations

import argparse
import gzip
import json
import random

from huggingface_hub import hf_hub_download

from effrag.config import DATA_DIR, SEED
from effrag.data import load_litm_nq
from effrag.metrics import normalize_answer

REPO = "Tevatron/wikipedia-nq"


def read(split: str):
    path = hf_hub_download(REPO, f"nq-{split}.jsonl.gz", repo_type="dataset")
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            r["positive_passages"] = r["positive_passages"][:1]
            r["negative_passages"] = r["negative_passages"][:19]  # k <= 20, keeps RAM low
            yield r


def build(rows: list[dict], n: int, rng: random.Random, ks=(5, 10, 20), p_open=0.5) -> list[dict]:
    pool = [p for r in rows for p in r["negative_passages"][:3]]  # random filler passages
    out = []
    for r in rows:
        if len(out) >= n:
            break
        if not r["positive_passages"] or not r["answers"]:
            continue
        k = rng.choice(ks)
        negs = list(r["negative_passages"][: k - 1])
        while len(negs) < k - 1:
            negs.append(rng.choice(pool))
        gold = rng.randrange(k)
        docs = negs[:gold] + [r["positive_passages"][0]] + negs[gold:]
        open_idx = None
        if rng.random() < p_open:
            others = [i for i in range(k) if i != gold]
            open_idx = gold if rng.random() < 0.6 else rng.choice(others)
        out.append({
            "question": r["query"], "answers": r["answers"], "gold": gold, "open": open_idx,
            "docs": [[d["title"], d["text"]] for d in docs],
        })  # fmt: skip
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=15000)
    ap.add_argument("--n-dev", type=int, default=200)
    args = ap.parse_args()
    rng = random.Random(SEED)

    test_q = {normalize_answer(ex.question) for ex in load_litm_nq(0, n=None)}
    out_dir = DATA_DIR / "soft"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, n, ks, p_open in (("train", args.n_train, (5, 10, 20), 0.5),
                                 ("dev", args.n_dev, (20,), 0.0)):  # fmt: skip
        rows = [r for r in read(split) if normalize_answer(r["query"]) not in test_q]
        rng.shuffle(rows)
        data = build(rows, n, rng, ks, p_open)
        with (out_dir / f"{split}.jsonl").open("w") as f:
            for ex in data:
                f.write(json.dumps(ex) + "\n")
        print(f"{split}: {len(data)} examples (from {len(rows)} questions not in LITM test)")


if __name__ == "__main__":
    main()
