"""Provence (Chirkova et al., ICLR 2025): question-aware sentence pruning + reranking."""

from effrag.compression.base import Compressor, render_docs
from effrag.data import Doc


class Provence(Compressor):
    name = "provence"

    def __init__(self, threshold: float = 0.1, model: str = "naver/provence-reranker-debertav3-v1"):
        from transformers import AutoModel

        from effrag.config import device

        self.threshold, self.model_name = threshold, model
        self.model = AutoModel.from_pretrained(model, trust_remote_code=True).to(device()).eval()

    def params(self) -> dict:
        return {"threshold": self.threshold, "model": self.model_name}

    def _compress(self, question: str, docs: list[Doc]) -> str:
        out = self.model.process(
            [question],
            [[d.text for d in docs]],
            title=[[d.title for d in docs]],
            threshold=self.threshold,
            always_select_title=False,
            enable_warnings=False,
        )
        kept = [
            Doc(d.title, pruned)
            for d, pruned in zip(docs, out["pruned_context"][0], strict=True)
            if pruned
        ]
        return render_docs(kept)
