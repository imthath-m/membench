"""Upper-bound baseline: recall returns exactly the query's gold `evidence` events.

The one adapter that sees scoring ground truth, by design: the runner sets
`evidence` before each recall (`sees_evidence = True`). Retrieval is perfect,
so QA accuracy on it isolates the reader. A query with no `evidence` gets an
empty context; forgotten events are never returned.
"""

from __future__ import annotations

from membench.core.adapter import Event, MemoryAdapter, RecalledItem, RecallResult


class OracleAdapter(MemoryAdapter):
    """Perfect retrieval from the pack's evidence labels (an upper bound)."""

    sees_evidence = True

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self._events: dict[str, Event] = {}
        self.evidence: list[str] = []

    def write(self, event: Event) -> None:
        self._events[event.id] = event

    def recall(self, query: str, *, session_id: str, at: str) -> RecallResult:
        events = [self._events[i] for i in self.evidence if i in self._events]
        return RecallResult(
            context="\n".join(f"[{e.timestamp}] [{e.role}] {e.content}" for e in events),
            items=[RecalledItem(event_ids=[e.id], text=e.content) for e in events],
        )

    def forget(self, event_ids: list[str]) -> None:
        for event_id in event_ids:
            self._events.pop(event_id, None)

    def reset(self) -> None:
        self._events = {}
