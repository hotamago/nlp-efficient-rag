"""Dataset loaders.

Main benchmark: NQ-Open multi-document QA from "Lost in the Middle" (Liu et al., TACL 2024),
20 Contriever-retrieved Wikipedia passages per question, gold passage at a fixed position.
Download first with `uv run python scripts/download_data.py`.
"""

import gzip
import json
import random
from dataclasses import dataclass

from effrag.config import DATA_DIR, SEED

LITM_DIR = DATA_DIR / "raw" / "litm_nq20"


@dataclass
class Doc:
    title: str
    text: str
    has_answer: bool = False
    is_gold: bool = False

    def render(self) -> str:
        return f"(Title: {self.title}) {self.text}"


@dataclass
class Example:
    qid: str
    question: str
    answers: list[str]
    docs: list[Doc]


def load_litm_nq(gold_pos: int = 0, n: int | None = 500, seed: int = SEED) -> list[Example]:
    """Load a fixed random subset (same question ids for every gold position given the seed)."""
    path = LITM_DIR / f"nq-open-20_total_documents_gold_at_{gold_pos}.jsonl.gz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing, run scripts/download_data.py")
    with gzip.open(path, "rt") as f:
        rows = [json.loads(line) for line in f]
    idx = list(range(len(rows)))
    if n is not None and n < len(rows):
        idx = sorted(random.Random(seed).sample(idx, n))
    return [
        Example(
            qid=f"nq{i}",
            question=rows[i]["question"],
            answers=rows[i]["answers"],
            docs=[
                Doc(c["title"], c["text"], bool(c.get("hasanswer")), bool(c.get("isgold")))
                for c in rows[i]["ctxs"]
            ],
        )
        for i in idx
    ]


def litm_corpus(gold_pos: int = 0) -> list[Doc]:
    """Pool and de-duplicate every passage in the dataset into one retrieval corpus."""
    seen: set[tuple[str, str]] = set()
    corpus: list[Doc] = []
    for ex in load_litm_nq(gold_pos, n=None):
        for d in ex.docs:
            key = (d.title, d.text)
            if key not in seen:
                seen.add(key)
                corpus.append(Doc(d.title, d.text))
    return corpus
