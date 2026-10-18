"""Reference points and training-free baselines."""

import numpy as np

from effrag.compression.base import Compressor, DocwiseCompressor, render_docs
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


class TopKSelector(DocwiseCompressor):
    """Document-level selection: keep the k passages scored highest for the question.

    Kept passages stay in their original order, as in the per-position tables of LongLLMLingua.
    """

    def __init__(self, k: int = 5):
        super().__init__()
        self.k = k

    def _scores(self, question: str, docs: list[Doc]) -> np.ndarray:
        raise NotImplementedError

    def _compress_docs(self, question: str, docs: list[Doc]) -> list[str]:
        top = set(np.argsort(-self._scores(question, docs))[: self.k].tolist())
        return [d.render() if i in top else "" for i, d in enumerate(docs)]


class BM25TopK(TopKSelector):
    name = "bm25_topk"

    def params(self) -> dict:
        return {"k": self.k}

    def _scores(self, question: str, docs: list[Doc]) -> np.ndarray:
        import bm25s
        import Stemmer

        stemmer = Stemmer.Stemmer("english")
        corpus = bm25s.tokenize(
            [d.render() for d in docs], stopwords="en", stemmer=stemmer, show_progress=False
        )
        retriever = bm25s.BM25()
        retriever.index(corpus, show_progress=False)
        q = bm25s.tokenize(question, stopwords="en", stemmer=stemmer, return_ids=False)[0]
        q = [t for t in q if t in retriever.vocab_dict]
        return retriever.get_scores(q) if q else np.zeros(len(docs))


class DenseTopK(TopKSelector):
    """Bi-encoder similarity (sentence-transformers)."""

    name = "dense_topk"

    def __init__(self, k: int = 5, model: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        from effrag.config import device

        super().__init__(k)
        self.model_name = model
        self.model = SentenceTransformer(model, device=device())

    def params(self) -> dict:
        return {"k": self.k, "model": self.model_name}

    def _scores(self, question: str, docs: list[Doc]) -> np.ndarray:
        q = self.model.encode([question], normalize_embeddings=True)
        d = self.model.encode([x.render() for x in docs], normalize_embeddings=True)
        return d @ q[0]


class RerankTopK(TopKSelector):
    """Cross-encoder reranker scores."""

    name = "rerank_topk"

    def __init__(self, k: int = 5, model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"):
        from sentence_transformers import CrossEncoder

        from effrag.config import device

        super().__init__(k)
        self.model_name = model
        self.model = CrossEncoder(model, device=device())

    def params(self) -> dict:
        return {"k": self.k, "model": self.model_name}

    def _scores(self, question: str, docs: list[Doc]) -> np.ndarray:
        return np.asarray(self.model.predict([(question, d.render()) for d in docs]))
