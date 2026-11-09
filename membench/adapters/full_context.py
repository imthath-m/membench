"""Baseline adapter: append every event, recall returns the whole log."""

from __future__ import annotations

from membench.core.adapter import Event, MemoryAdapter, RecalledItem, RecallResult


class FullContextAdapter(MemoryAdapter):
    """No memory system at all — just replays the full event log."""

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self._events: list[Event] = []

    def write(self, event: Event) -> None:
        self._events.append(event)

    def recall(self, query: str, *, session_id: str, at: str) -> RecallResult:
        lines = [f"[{e.role}] {e.content}" for e in self._events]
        return RecallResult(
            context="\n".join(lines),
            items=[RecalledItem(event_ids=[e.id], text=e.content) for e in self._events],
        )

    def forget(self, event_ids: list[str]) -> None:
        targets = set(event_ids)
        self._events = [e for e in self._events if e.id not in targets]

    def reset(self) -> None:
        self._events = []
