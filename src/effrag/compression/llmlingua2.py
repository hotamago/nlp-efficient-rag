"""LLMLingua-2 (Pan et al., Findings ACL 2024): token classification, task-agnostic."""

from effrag.compression.base import Compressor
from effrag.data import Doc


class LLMLingua2(Compressor):
    name = "llmlingua2"

    def __init__(
        self,
        rate: float = 0.5,
        model: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    ):
        from llmlingua import PromptCompressor

        from effrag.config import device

        self.rate, self.model_name = rate, model
        self.pc = PromptCompressor(model_name=model, use_llmlingua2=True, device_map=device())

    def params(self) -> dict:
        return {"rate": self.rate, "model": self.model_name}

    def _compress(self, question: str, docs: list[Doc]) -> str:
        out = self.pc.compress_prompt(
            [f"Document [{i}]{d.render()}" for i, d in enumerate(docs, 1)],
            rate=self.rate,
            force_tokens=["\n", "?", "."],
            use_context_level_filter=False,
        )
        return out["compressed_prompt"]
