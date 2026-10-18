"""Compressor registry. Heavy models are imported lazily."""

from effrag.compression.base import CompressionResult, Compressor, render_docs


def get_compressor(name: str, **kwargs) -> Compressor:
    if name == "full":
        from effrag.compression.baselines import FullContext as cls
    elif name == "closedbook":
        from effrag.compression.baselines import ClosedBook as cls
    elif name == "oracle":
        from effrag.compression.baselines import OracleGold as cls
    elif name == "truncate":
        from effrag.compression.baselines import Truncate as cls
    elif name == "dense_topk":
        from effrag.compression.baselines import DenseTopK as cls
    elif name == "llmlingua2":
        from effrag.compression.llmlingua2 import LLMLingua2 as cls
    elif name == "provence":
        from effrag.compression.provence import Provence as cls
    else:
        raise ValueError(f"unknown compressor: {name}")
    return cls(**kwargs)


__all__ = ["CompressionResult", "Compressor", "get_compressor", "render_docs"]
