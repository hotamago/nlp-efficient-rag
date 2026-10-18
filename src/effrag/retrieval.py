"""Retrievers over a passage corpus: BM25S (lexical) and dense (sentence-transformers + FAISS)."""

import time

import numpy as np

from effrag.data import Doc


class BM25Retriever:
    name = "bm25s"

    def __init__(self, corpus: list[Doc]):
        import bm25s
        import Stemmer

        self.corpus = corpus
        self.stemmer = Stemmer.Stemmer("english")
        t0 = time.perf_counter()
        tokens = bm25s.tokenize(
            [d.render() for d in corpus], stopwords="en", stemmer=self.stemmer, show_progress=False
        )
        self.index = bm25s.BM25()
        self.index.index(tokens, show_progress=False)
        self.build_s = time.perf_counter() - t0
        self._bm25s = bm25s

    def search(self, queries: list[str], k: int = 10) -> list[list[int]]:
        q = self._bm25s.tokenize(queries, stopwords="en", stemmer=self.stemmer, show_progress=False)
        ids, _ = self.index.retrieve(q, k=k, show_progress=False)
        return ids.tolist()


class DenseRetriever:
    """Exact inner-product search; `dim` truncates Matryoshka embeddings."""

    name = "dense"

    def __init__(
        self,
        corpus: list[Doc],
        model: str = "BAAI/bge-small-en-v1.5",
        dim: int | None = None,
        batch_size: int = 128,
    ):
        import faiss
        from sentence_transformers import SentenceTransformer

        from effrag.config import device

        self.corpus, self.model_name = corpus, model
        self.model = SentenceTransformer(model, device=device(), truncate_dim=dim)
        t0 = time.perf_counter()
        emb = self.model.encode(
            [d.render() for d in corpus], batch_size=batch_size, normalize_embeddings=True
        )
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(np.asarray(emb, dtype=np.float32))
        self.build_s = time.perf_counter() - t0

    def search(self, queries: list[str], k: int = 10) -> list[list[int]]:
        q = self.model.encode(queries, normalize_embeddings=True)
        _, ids = self.index.search(np.asarray(q, dtype=np.float32), k)
        return ids.tolist()
