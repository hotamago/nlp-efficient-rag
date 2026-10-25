"""Provence-family pruners: question-aware sentence pruning + reranking in one pass.

- Provence (Chirkova et al., ICLR 2025, arXiv 2501.16214): naver/provence-reranker-debertav3-v1
- XProvence (arXiv 2601.18886): naver/xprovence-reranker-bgem3-v2 (multilingual)
- OpenProvence (MIT, hotchpotch/open_provence): hotchpotch/open-provence-reranker-*
"""

from effrag.compression.base import DocwiseCompressor
from effrag.data import Doc


class Provence(DocwiseCompressor):
    name = "provence"

    def __init__(
        self,
        threshold: float = 0.1,
        model: str = "naver/provence-reranker-debertav3-v1",
        revision: str | None = None,
        device: str | None = None,
    ):
        from transformers import AutoModel

        from effrag.config import device as default_device

        super().__init__()
        self.threshold, self.model_name, self.revision = threshold, model, revision
        self.model = AutoModel.from_pretrained(model, trust_remote_code=True, revision=revision)
        self.model = self.model.to(device or default_device()).eval()

    def params(self) -> dict:
        return {"threshold": self.threshold, "model": self.model_name}

    def _compress_docs(self, question: str, docs: list[Doc]) -> list[str]:
        if "naver/" in self.model_name:
            out = self.model.process(
                [question],
                [[d.text for d in docs]],
                title=[[d.title for d in docs]],
                threshold=self.threshold,
                always_select_title=False,
                enable_warnings=False,
            )
            pruned = out["pruned_context"][0]
        else:  # OpenProvence: one (question, passage) pair per item
            out = self.model.process(
                question=[question] * len(docs),
                context=[d.text for d in docs],
                threshold=self.threshold,
                language="en",  # "auto" misdetects some passages as Japanese
                show_progress=False,
            )
            pruned = out["pruned_context"]
        return [
            f"(Title: {d.title}) {p.strip()}" if p and p.strip() else ""
            for d, p in zip(docs, pruned, strict=True)
        ]
