"""OpenAI-compatible generator with an on-disk response cache (works with APIs, vLLM, Ollama)."""

import hashlib
import json
import time
from pathlib import Path

from openai import OpenAI

from effrag.config import DATA_DIR, LLMSettings

# Prompt from Lost in the Middle (Liu et al., 2024), reused by LongLLMLingua.
QA_TEMPLATE = (
    "Write a high-quality answer for the given question using only the provided search results "
    "(some of which might be irrelevant).\n\n{context}\n\nQuestion: {question}\nAnswer:"
)
CLOSEDBOOK_TEMPLATE = "Question: {question}\nAnswer:"


def build_prompt(question: str, context: str) -> str:
    if not context.strip():
        return CLOSEDBOOK_TEMPLATE.format(question=question)
    return QA_TEMPLATE.format(context=context, question=question)


class LLM:
    def __init__(self, settings: LLMSettings | None = None, cache_dir: Path | None = None):
        self.s = settings or LLMSettings()
        if not self.s.model:
            raise ValueError("LLM_MODEL is empty, set it in .env")
        self.client = OpenAI(base_url=self.s.base_url, api_key=self.s.api_key)
        self.cache_dir = cache_dir or DATA_DIR / "cache" / "llm"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, prompt: str) -> Path:
        raw = json.dumps([self.s.model, self.s.temperature, self.s.max_tokens, prompt])
        return self.cache_dir / f"{hashlib.sha256(raw.encode()).hexdigest()}.json"

    def generate(self, prompt: str, retries: int = 5) -> dict:
        """Return {"text", "latency_s", "prompt_tokens", "cached"}."""
        path = self._key(prompt)
        if path.exists():
            return {**json.loads(path.read_text()), "cached": True}
        for attempt in range(retries):
            try:
                t0 = time.perf_counter()
                resp = self.client.chat.completions.create(
                    model=self.s.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.s.temperature,
                    max_tokens=self.s.max_tokens,
                )
                out = {
                    "text": resp.choices[0].message.content or "",
                    "latency_s": time.perf_counter() - t0,
                    "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
                }
                path.write_text(json.dumps(out))
                return {**out, "cached": False}
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError("unreachable")
