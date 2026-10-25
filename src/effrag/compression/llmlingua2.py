"""LLMLingua-2 (Pan et al., Findings ACL 2024, arXiv 2403.12968): token classification,
task-agnostic (does not look at the question)."""

from effrag.compression.base import DocwiseCompressor
from effrag.data import Doc


class LLMLingua2(DocwiseCompressor):
    name = "llmlingua2"

    def __init__(
        self,
        rate: float = 0.5,
        model: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        device: str | None = None,
    ):
        from llmlingua import PromptCompressor

        from effrag.config import device as default_device

        super().__init__()
        self.rate, self.model_name = rate, model
        # `device` lets the demo keep this one on CPU while vLLM shares the GPU.
        self.pc = PromptCompressor(
            model_name=model, use_llmlingua2=True, device_map=device or default_device()
        )

    def params(self) -> dict:
        return {"rate": self.rate, "model": self.model_name}

    def _compress_docs(self, question: str, docs: list[Doc]) -> list[str]:
        # One call over all docs: the keep-threshold is global, so the rate holds overall.
        out = self.pc.compress_prompt(
            [d.render() for d in docs],
            rate=self.rate,
            force_tokens=["\n", "?", "."],
            use_context_level_filter=False,
            return_word_label=False,
        )
        return out["compressed_prompt_list"]
