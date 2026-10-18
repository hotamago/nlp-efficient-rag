"""Runtime settings loaded from environment variables / `.env`."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
SEED = 42


@dataclass(frozen=True)
class LLMSettings:
    base_url: str | None = field(default_factory=lambda: os.getenv("LLM_BASE_URL") or None)
    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", "EMPTY"))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", ""))
    temperature: float = 0.0
    max_tokens: int = 64
    # USD per 1M input tokens, only used to estimate cost in reports.
    price_in: float = field(default_factory=lambda: float(os.getenv("LLM_PRICE_IN", "0")))


def device() -> str:
    forced = os.getenv("EFFRAG_DEVICE")
    if forced:
        return forced
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
