"""Mem0 OSS adapter: write/recall/forget over `mem0ai`'s local `Memory`
client. No network calls beyond what Mem0 itself makes per its own
config (e.g. local vector store + configured LLM/embedder).
"""

from __future__ import annotations

from typing import Any

from membench.core.adapter import Event, MemoryAdapter, RecallResult
from membench.core.tokens import count_tokens

try:
    from mem0 import Memory
except ImportError:  # pragma: no cover - exercised only when mem0 missing
    Memory = None


class Mem0Adapter(MemoryAdapter):
    """Mem0 OSS baseline backend."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if Memory is None:
            raise ImportError(
                "mem0ai is not installed. Install it with "
                "`pip install membench[mem0]` (or `pip install mem0ai`) "
                "to use the mem0 backend."
            )
        super().__init__(config)
        self.user_id = self.config.get("user_id", "membench")
        self.top_k = int(self.config.get("top_k", 3))
        mem0_config = self.config.get("mem0_config")
        self._client = Memory.from_config(mem0_config) if mem0_config else Memory()
        self._ids: dict[str, str] = {}

    def write(self, event: Event) -> None:
        result = self._client.add(
            event.content, user_id=self.user_id, metadata={"key": event.key}
        )
        memories = result.get("results", []) if isinstance(result, dict) else result
        if memories:
            self._ids[event.key] = memories[0]["id"]

    def recall(self, query: str) -> RecallResult:
        result = self._client.search(query, user_id=self.user_id, limit=self.top_k)
        memories = result.get("results", []) if isinstance(result, dict) else result
        context = "\n".join(m["memory"] for m in memories)
        return RecallResult(
            context=context,
            items=memories,
            token_count=count_tokens(context),
        )

    def forget(self, key: str) -> None:
        memory_id = self._ids.pop(key, None)
        if memory_id is not None:
            self._client.delete(memory_id)
