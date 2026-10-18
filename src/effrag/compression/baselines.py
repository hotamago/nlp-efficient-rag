"""Reference points and training-free baselines."""

from effrag.compression.base import Compressor, render_docs
from effrag.data import Doc
from effrag.tokens import _enc


class FullContext(Compressor):
    name = "full"

    def _compress(self, question: str, docs: list[Doc]) -> str:
        return render_docs(docs)


class ClosedBook(Compressor):
    name = "closedbook"

    def _compress(self, question: str, docs: list[Doc]) -> str:
        return ""


class OracleGold(Compressor):
    """Upper bound: only the passage annotated as gold."""

    name = "oracle"

    def _compress(self, question: str, docs: list[Doc]) -> str:
        return render_docs([d for d in docs if d.is_gold])


class Truncate(Compressor):
    """Keep the first `budget` tokens of the full context."""

    name = "truncate"

    def __init__(self, budget: int = 700):
        self.budget = budget

    def params(self) -> dict:
        return {"budget": self.budget}

    def _compress(self, question: str, docs: list[Doc]) -> str:
        ids = _enc().encode(render_docs(docs), disallowed_special=())
        return _enc().decode(ids[: self.budget])


class DenseTopK(Compressor):
    """Document-level selection: keep the k passages most similar to the question.

    Kept passages stay in their original order (`reorder=False`) so the gold-position
    variable survives, matching the per-position tables of LongLLMLingua.
    """

    name = "dense_topk"

    def __init__(self, k: int = 5, model: str = "BAAI/bge-small-en-v1.5", reorder: bool = False):
        from sentence_transformers import SentenceTransformer

        from effrag.config import device

        self.k, self.model_name, self.reorder = k, model, reorder
        self.model = SentenceTransformer(model, device=device())

    def params(self) -> dict:
        return {"k": self.k, "model": self.model_name, "reorder": self.reorder}

    def _compress(self, question: str, docs: list[Doc]) -> str:
        q = self.model.encode([question], normalize_embeddings=True)
        d = self.model.encode([x.render() for x in docs], normalize_embeddings=True)
        top = (d @ q[0]).argsort()[::-1][: self.k]
        order = top if self.reorder else sorted(top)
        return render_docs([docs[i] for i in order])
