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
