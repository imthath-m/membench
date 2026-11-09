"""Token counting for recalled context.

`count_tokens` is a dependency-free whitespace word count, labelled
`whitespace:approx` in the report. `default_tokenizer()` returns tiktoken's
`o200k_base` encoding when the `membench[tokens]` extra is installed and its
vocabulary is cached or downloadable (cached under
`~/.cache/membench/tiktoken` unless `TIKTOKEN_CACHE_DIR` is set); otherwise
it falls back to the whitespace count, so a run never needs the network.
Every tokenizer carries a `name` that the run manifest records.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger("membench")

TIKTOKEN_ENCODING = "o200k_base"
_CACHE_DIR = Path.home() / ".cache" / "membench" / "tiktoken"


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


count_tokens.name = "whitespace:approx"


class TiktokenCounter:
    """Exact token count with a tiktoken encoding."""

    def __init__(self, encoding: str = TIKTOKEN_ENCODING) -> None:
        import tiktoken

        os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(_CACHE_DIR))
        self._encoding = tiktoken.get_encoding(encoding)
        self.name = f"tiktoken:{encoding}"

    def __call__(self, text: str) -> int:
        return len(self._encoding.encode(text, disallowed_special=())) if text else 0


def default_tokenizer() -> Callable[[str], int]:
    """tiktoken `o200k_base` if available, else the approximate whitespace count."""
    try:
        return TiktokenCounter()
    except Exception as exc:  # noqa: BLE001 - not installed, or vocab download failed offline
        logger.warning("tiktoken unavailable (%s); token counts are approximate", exc)
        return count_tokens
