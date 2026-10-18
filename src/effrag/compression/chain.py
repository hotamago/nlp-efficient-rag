"""Two-stage compression: a document selector followed by a finer compressor
(e.g. reranker top-k then LLMLingua-2, the "coarse + fine" recipe of LongLLMLingua)."""

from effrag.compression.base import DocwiseCompressor
from effrag.data import Doc


class Chain(DocwiseCompressor):
    name = "chain"

    def __init__(self, stages: list[dict]):
        from effrag.compression import get_compressor

        super().__init__()
        self.stages_cfg = stages
        self.stages = [
            get_compressor(s["method"], **{k: v for k, v in s.items() if k != "method"})
            for s in stages
        ]
        if not all(isinstance(s, DocwiseCompressor) for s in self.stages):
            raise ValueError("chain stages must be docwise compressors")

    def params(self) -> dict:
        return {"stages": [{"method": s.name, **s.params()} for s in self.stages]}

    def _compress_docs(self, question: str, docs: list[Doc]) -> list[str]:
        out = [d.render() for d in docs]
        for stage in self.stages:
            alive = [i for i, s in enumerate(out) if s]
            kept = stage._compress_docs(question, [docs[i] for i in alive])
            out = [""] * len(docs)
            for i, s in zip(alive, kept, strict=True):
                out[i] = s
        return out
