"""Canonical token counter so every method is measured with the same ruler."""

from functools import lru_cache

import tiktoken


@lru_cache(maxsize=1)
def _enc() -> tiktoken.Encoding:
    return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(_enc().encode(text, disallowed_special=()))
