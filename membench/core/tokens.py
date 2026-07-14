"""Lightweight, dependency-free token count approximation.

Avoids network-fetched tokenizer vocabularies (e.g. tiktoken) so adapters
stay usable fully offline. Approximates tokens as whitespace-separated
words, which is good enough for relative token-cost comparisons.
"""

from __future__ import annotations


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(text.split())
