"""LongLLMLingua (Jiang et al., ACL 2024, arXiv 2310.06839): question-aware coarse-to-fine
compression with a small causal LM. The paper used Llama-2-7B; we use a 0.5B model."""

from effrag.compression.base import Compressor
from effrag.data import Doc


class LongLLMLingua(Compressor):
    name = "longllmlingua"

    def __init__(
        self,
        target_token: int = 730,
        model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str | None = None,
    ):
        from llmlingua import PromptCompressor

        from effrag.config import device as default_device

        self.target_token, self.model_name = target_token, model
        # `device` lets the demo keep this one on CPU while vLLM shares the GPU.
        self.pc = PromptCompressor(model_name=model, device_map=device or default_device())
        _patch_legacy_cache(self.pc.model)
        _patch_prefix_length(self.pc)

    def params(self) -> dict:
        return {"target_token": self.target_token, "model": self.model_name}

    def _compress(self, question: str, docs: list[Doc]) -> str:
        # Settings follow the LongLLMLingua README example for multi-document QA.
        out = self.pc.compress_prompt(
            [f"Document [{i}]{d.render()}" for i, d in enumerate(docs, 1)],
            question=question,
            target_token=self.target_token,
            condition_compare=True,
            condition_in_question="after",
            rank_method="longllmlingua",
            use_sentence_level_filter=False,
            context_budget="+100",
            dynamic_context_compression_ratio=0.4,
            reorder_context="original",
        )
        return out["compressed_prompt"]


def _patch_legacy_cache(model) -> None:
    """llmlingua 0.2.2 slices past_key_values as tuples; newer transformers expect Cache objects."""
    from transformers import DynamicCache

    forward = model.forward

    def wrapped(*args, past_key_values=None, **kwargs):
        if isinstance(past_key_values, list | tuple):
            past_key_values = DynamicCache.from_legacy_cache(tuple(map(tuple, past_key_values)))
        out = forward(*args, past_key_values=past_key_values, **kwargs)
        if hasattr(out.past_key_values, "to_legacy_cache"):
            out.past_key_values = out.past_key_values.to_legacy_cache()
        return out

    model.forward = wrapped


def _patch_prefix_length(pc) -> None:
    """llmlingua asserts that prefix and context tokenize separately; BPE merges across the
    boundary break that for some inputs. Fall back to the prefix's own token length."""
    original = pc.get_prefix_length

    def safe(prefix: str, text: str) -> int:
        try:
            return original(prefix, text)
        except AssertionError:
            return pc.get_token_length(prefix, False)

    pc.get_prefix_length = safe
