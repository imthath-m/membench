"""Baseline adapter: append every event, recall returns the whole log."""

from __future__ import annotations

from membench.core.adapter import Event, MemoryAdapter, RecallResult
from membench.core.tokens import count_tokens


class FullContextAdapter(MemoryAdapter):
    """No memory system at all — just replays the full event log."""

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self._events: list[Event] = []

    def write(self, event: Event) -> None:
        self._events.append(event)

    def recall(self, query: str) -> RecallResult:
        lines = [f"[{e.role}] {e.content}" for e in self._events]
        context = "\n".join(lines)
        return RecallResult(
            context=context,
            items=list(self._events),
            token_count=count_tokens(context),
        )

    def forget(self, key: str) -> None:
        self._events = [e for e in self._events if e.key != key]
