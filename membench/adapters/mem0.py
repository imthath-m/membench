"""Mem0 OSS adapter: write/recall/forget over `mem0ai`'s local `Memory`
client. No network calls beyond what Mem0 itself makes per its own
config (e.g. local vector store + configured LLM/embedder).
"""

from __future__ import annotations

from typing import Any

from membench.core.adapter import Event, MemoryAdapter, RecalledItem, RecallResult
from membench.core.usage import meter_openai, new_usage

try:
    from mem0 import Memory
except ImportError:  # pragma: no cover - exercised only when mem0 missing
    Memory = None

try:  # Mem0's default LLM/embedder calls go through the OpenAI SDK.
    import openai

    _RETRYABLE: tuple[type[BaseException], ...] = (
        openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError
    )
except ImportError:  # pragma: no cover - exercised only when openai missing
    _RETRYABLE = ()


class Mem0Adapter(MemoryAdapter):
    """Mem0 OSS baseline backend."""

    retryable = _RETRYABLE
    packages = ("mem0ai", "openai")
    uses_llm = True

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if Memory is None:
            raise ImportError(
                "mem0ai is not installed. Install it with "
                "`pip install membench[mem0]` (or `pip install mem0ai`) "
                "to use the mem0 backend."
            )
        super().__init__(config)
        self.user_id = self.config.get("user_id", self.namespace)
        self.top_k = int(self.config.get("top_k", 3))
        mem0_config = self.config.get("mem0_config")
        self._client = Memory.from_config(mem0_config) if mem0_config else Memory()
        self._sources: dict[str, list[str]] = {}  # mem0 memory id -> Event.ids
        self._llm = getattr(self._client, "llm", None)
        self._embedder = getattr(self._client, "embedding_model", None)
        self._usage = new_usage()
        for part in (self._llm, self._embedder):  # extraction + embedding tokens
            meter_openai(getattr(part, "client", None), self._usage)

    def write(self, event: Event) -> None:
        result = self._client.add(event.content, user_id=self.user_id)
        memories = result.get("results", []) if isinstance(result, dict) else result
        for memory in memories or []:  # one add may create or update several memories
            self._sources.setdefault(memory["id"], []).append(event.id)

    def recall(self, query: str, *, session_id: str, at: str) -> RecallResult:
        result = self._client.search(
            query, filters={"user_id": self.user_id}, top_k=self.top_k
        )
        memories = result.get("results", []) if isinstance(result, dict) else result
        items = [
            RecalledItem(self._sources.get(m["id"], []), m["memory"], m.get("score"))
            for m in memories
        ]
        return RecallResult(context="\n".join(i.text for i in items), items=items)

    def forget(self, event_ids: list[str]) -> None:
        """Delete every memory derived from a target event."""
        targets = set(event_ids)
        for memory_id, sources in list(self._sources.items()):
            if targets.intersection(sources):
                self._client.delete(memory_id)
                del self._sources[memory_id]

    def reset(self) -> None:
        self._client.delete_all(user_id=self.user_id)
        self._sources.clear()

    def usage(self) -> dict[str, int]:
        return dict(self._usage)

    def describe(self) -> dict[str, Any]:
        llm = getattr(self._llm, "config", None)
        base_url = getattr(getattr(self._llm, "client", None), "base_url", None)
        return {
            "llm_model": getattr(llm, "model", None),
            "llm_base_url": str(base_url) if base_url else None,
            "temperature": getattr(llm, "temperature", None),
            "embedding_model": getattr(getattr(self._embedder, "config", None), "model", None),
        }
