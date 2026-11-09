"""Lexical baseline: Okapi BM25 over every written event (pure Python, offline).

A floor for retrieval: no embeddings, no LLM. Context is the top-k event
texts, best first.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from membench.core.adapter import Event, MemoryAdapter, RecalledItem, RecallResult

_TOKEN = re.compile(r"\w+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Adapter(MemoryAdapter):
    """Okapi BM25 (k1, b) ranking of events against the query."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.top_k = int(self.config.get("top_k", 3))
        self.k1 = float(self.config.get("k1", 1.5))
        self.b = float(self.config.get("b", 0.75))
        self._docs: dict[str, tuple[Event, Counter[str], int]] = {}

    def write(self, event: Event) -> None:
        terms = _tokens(event.content)
        self._docs[event.id] = (event, Counter(terms), len(terms))

    def recall(self, query: str, *, session_id: str, at: str) -> RecallResult:
        n = len(self._docs)
        if n == 0:
            return RecallResult(context="")
        avg_len = sum(length for _, _, length in self._docs.values()) / n or 1.0
        df = Counter(term for _, tf, _ in self._docs.values() for term in tf)
        terms = set(_tokens(query))
        scored = []
        for event, tf, length in self._docs.values():
            norm = self.k1 * (1 - self.b + self.b * length / avg_len)
            score = sum(
                math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) * tf[t] * (self.k1 + 1)
                / (tf[t] + norm)
                for t in terms if t in tf
            )
            if score > 0:
                scored.append((score, event))
        scored.sort(key=lambda pair: -pair[0])  # stable: ties keep write order
        items = [RecalledItem([e.id], e.content, s) for s, e in scored[: self.top_k]]
        return RecallResult(context="\n".join(i.text for i in items), items=items)

    def forget(self, event_ids: list[str]) -> None:
        for event_id in event_ids:
            self._docs.pop(event_id, None)

    def reset(self) -> None:
        self._docs = {}

    def describe(self) -> dict[str, Any]:
        return {"top_k": self.top_k, "k1": self.k1, "b": self.b}
