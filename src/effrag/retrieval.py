"""Retrievers over a passage corpus: BM25S (lexical) and dense bi-encoders with FAISS.

Dense variants cover the efficiency knobs studied in Track B:
- Matryoshka truncation (`dim`), e.g. Qwen3-Embedding or static-retrieval-mrl
- embedding quantization: float32, int8 (scalar) and binary (Hamming search + float rescoring)
- exact (flat) vs approximate (HNSW) search
"""

import hashlib
import time
from pathlib import Path

import numpy as np

from effrag.config import DATA_DIR
from effrag.data import Doc


class BM25Retriever:
    name = "bm25s"

    def __init__(self, corpus: list[Doc]):
        import bm25s
        import Stemmer

        self.stemmer = Stemmer.Stemmer("english")
        t0 = time.perf_counter()
        tokens = bm25s.tokenize(
            [d.render() for d in corpus], stopwords="en", stemmer=self.stemmer, show_progress=False
        )
        self.index = bm25s.BM25()
        self.index.index(tokens, show_progress=False)
        self.build_s = time.perf_counter() - t0
        self._bm25s = bm25s

    def index_bytes(self) -> int:
        s = self.index.scores
        return sum(s[k].nbytes for k in ("data", "indices", "indptr"))

    def search(self, queries: list[str], k: int = 10) -> list[list[int]]:
        q = self._bm25s.tokenize(queries, stopwords="en", stemmer=self.stemmer, show_progress=False)
        ids, _ = self.index.retrieve(q, k=k, show_progress=False, n_threads=1)
        return ids.tolist()


class DenseRetriever:
    """Bi-encoder retrieval. Corpus embeddings are cached on disk per model."""

    name = "dense"

    def __init__(
        self,
        corpus: list[Doc],
        model: str = "BAAI/bge-small-en-v1.5",
        dim: int | None = None,
        quant: str = "float32",  # float32 | int8 | binary
        index: str = "flat",  # flat | hnsw
        query_prompt: str | None = None,
        rescore_mult: int = 4,
        batch_size: int = 64,
    ):
        import faiss
        from sentence_transformers import SentenceTransformer

        from effrag.config import device

        self.model_name, self.dim, self.quant, self.index_type = model, dim, quant, index
        self.query_prompt, self.rescore_mult = query_prompt, rescore_mult
        self.model = SentenceTransformer(model, device=device())
        full = self._corpus_embeddings(corpus, batch_size)
        t0 = time.perf_counter()
        emb = self._truncate(full)
        self.float_emb = emb
        if quant == "binary":
            self.bits = np.packbits(emb > 0, axis=1)
            self.index = faiss.IndexBinaryFlat(emb.shape[1])
            self.index.add(self.bits)
        else:
            if quant == "int8":
                emb = self._int8(emb, fit=True)
            if index == "hnsw":
                self.index = faiss.IndexHNSWFlat(emb.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
            else:
                self.index = faiss.IndexFlatIP(emb.shape[1])
            self.index.add(np.ascontiguousarray(emb, dtype=np.float32))
        self.build_s = time.perf_counter() - t0

    def _corpus_embeddings(self, corpus: list[Doc], batch_size: int) -> np.ndarray:
        h = hashlib.md5("".join(d.title + d.text for d in corpus).encode()).hexdigest()[:8]
        path = DATA_DIR / "cache" / "emb" / f"{self.model_name.replace('/', '__')}_{h}.npy"
        if path.exists():
            self.encode_s = float(Path(str(path) + ".time").read_text())
            return np.load(path)
        t0 = time.perf_counter()
        emb = self.model.encode(
            [d.render() for d in corpus], batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=True, convert_to_numpy=True,
        )  # fmt: skip
        self.encode_s = time.perf_counter() - t0
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, emb.astype(np.float32))
        Path(str(path) + ".time").write_text(str(self.encode_s))
        return emb.astype(np.float32)

    def _truncate(self, emb: np.ndarray) -> np.ndarray:
        if self.dim is None or self.dim >= emb.shape[1]:
            return emb
        emb = emb[:, : self.dim]
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)

    def _int8(self, emb: np.ndarray, fit: bool = False) -> np.ndarray:
        # Scalar quantization to 256 levels per dimension; dequantized values feed the index so
        # the accuracy effect is measured; storage is counted as 1 byte/dim.
        if fit:
            self.lo, self.hi = emb.min(0), emb.max(0)
        scale = (self.hi - self.lo) / 255
        q = np.clip(np.round((emb - self.lo) / scale), 0, 255)
        return q * scale + self.lo

    def index_bytes(self) -> int:
        n, d = self.float_emb.shape
        per_vec = {"float32": 4 * d, "int8": d, "binary": d // 8}[self.quant]
        return n * per_vec

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        kw = {"prompt": self.query_prompt} if self.query_prompt else {}
        q = self.model.encode(queries, normalize_embeddings=True, convert_to_numpy=True, **kw)
        return self._truncate(q.astype(np.float32))

    def search(self, queries: list[str], k: int = 10) -> list[list[int]]:
        q = self.encode_queries(queries)
        if self.quant == "binary":
            _, cand = self.index.search(np.packbits(q > 0, axis=1), k * self.rescore_mult)
            # Rescore Hamming candidates: float query x {-1, +1} doc vectors (no float docs kept)
            d = q.shape[1]
            out = []
            for qi, c in zip(q, cand, strict=True):
                signs = np.unpackbits(self.bits[c], axis=1)[:, :d].astype(np.float32) * 2 - 1
                scores = signs @ qi
                out.append(c[np.argsort(-scores)[:k]].tolist())
            return out
        _, ids = self.index.search(np.ascontiguousarray(q), k)
        return ids.tolist()
