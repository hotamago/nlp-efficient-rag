import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from effrag.data import Doc
from effrag.tokens import count_tokens


def render_docs(docs: list[Doc]) -> str:
    return "\n".join(f"Document [{i}]{d.render()}" for i, d in enumerate(docs, 1))


@dataclass
class CompressionResult:
    context: str
    orig_tokens: int
    comp_tokens: int
    latency_s: float

    @property
    def ratio(self) -> float:
        return self.orig_tokens / max(self.comp_tokens, 1)


class Compressor(ABC):
    """Turns (question, retrieved docs) into the context string placed in the prompt."""

    name: str = "base"

    def params(self) -> dict:
        return {}

    def compress(self, question: str, docs: list[Doc]) -> CompressionResult:
        orig = count_tokens(render_docs(docs))
        t0 = time.perf_counter()
        context = self._compress(question, docs)
        latency = time.perf_counter() - t0
        return CompressionResult(context, orig, count_tokens(context), latency)

    @abstractmethod
    def _compress(self, question: str, docs: list[Doc]) -> str: ...


class DocwiseCompressor(Compressor):
    """Compressor whose output for each document does not depend on document order.

    Subclasses return one string per document (its rendered, compressed form, "" to drop it).
    Results are memoised per (question, document set), so the five gold-position files reuse
    one computation; the latency reported is the one measured when it was first computed.
    Kept documents stay in the input order and are renumbered.
    """

    def __init__(self) -> None:
        self._memo: dict[tuple, tuple[dict[tuple[str, str], str], float]] = {}

    @abstractmethod
    def _compress_docs(self, question: str, docs: list[Doc]) -> list[str]: ...

    def compress_docs(self, question: str, docs: list[Doc]) -> tuple[list[str], float]:
        """Per-document compressed text (aligned with `docs`, "" = dropped) and latency."""
        keys = [(d.title, d.text) for d in docs]
        memo_key = (question, frozenset(keys))
        if memo_key not in self._memo:
            t0 = time.perf_counter()
            kept = self._compress_docs(question, docs)
            self._memo[memo_key] = (dict(zip(keys, kept, strict=True)), time.perf_counter() - t0)
        kept_by_key, latency = self._memo[memo_key]
        return [kept_by_key[k] for k in keys], latency

    def compress(self, question: str, docs: list[Doc]) -> CompressionResult:
        per_doc, latency = self.compress_docs(question, docs)
        kept = [s for s in per_doc if s]
        context = "\n".join(f"Document [{i}]{s}" for i, s in enumerate(kept, 1))
        return CompressionResult(
            context, count_tokens(render_docs(docs)), count_tokens(context), latency
        )

    def _compress(self, question: str, docs: list[Doc]) -> str:
        return self.compress(question, docs).context
