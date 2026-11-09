"""LangMem adapter: a memory-store manager (LLM extraction) over a LangGraph store.

Each write runs LangMem's `create_memory_store_manager` on the event as one
message; the manager searches existing memories and inserts or updates
extracted ones in a LangGraph `BaseStore` (an `InMemoryStore` with an embedding
index). Recall is `store.search`. Each episode uses its own namespace.
"""

from __future__ import annotations

from typing import Any

from membench.core.adapter import Event, MemoryAdapter, RecalledItem, RecallResult
from membench.core.usage import new_usage

try:
    from langchain_core.callbacks import UsageMetadataCallbackHandler
    from langgraph.store.memory import InMemoryStore
    from langmem import create_memory_store_manager
except ImportError:  # pragma: no cover - exercised only when langmem missing
    create_memory_store_manager = None

try:  # the default chat model and embedder go through the OpenAI SDK
    import openai

    _RETRYABLE: tuple[type[BaseException], ...] = (
        openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError
    )
except ImportError:  # pragma: no cover - exercised only when openai missing
    _RETRYABLE = ()


class LangMemAdapter(MemoryAdapter):
    """LangMem memory-store manager; models are `provider:model` strings."""

    retryable = _RETRYABLE
    packages = ("langmem", "langgraph", "langchain-core")
    uses_llm = True

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if create_memory_store_manager is None:
            raise ImportError("langmem is not installed: `pip install membench[langmem]`.")
        super().__init__(config)
        self.top_k = int(self.config.get("top_k", 3))
        self.model = self.config.get("model", "openai:gpt-4o-mini")
        self.embedding = self.config.get("embedding", "openai:text-embedding-3-small")
        dims = int(self.config.get("dims", 1536))
        self._store = InMemoryStore(index={"dims": dims, "embed": self.embedding})
        self._manager = create_memory_store_manager(
            self.model, namespace=("membench", "{langgraph_user_id}"), store=self._store
        )
        self._episode, self._sources = 0, {}  # memory key -> Event.ids it came from
        self._usage = UsageMetadataCallbackHandler()

    @property
    def _namespace(self) -> tuple[str, str]:
        return ("membench", f"{self.namespace}-e{self._episode}")

    def write(self, event: Event) -> None:
        role = "assistant" if event.role == "assistant" else "user"
        puts = self._manager.invoke(
            {"messages": [{"role": role, "content": event.content}]},
            config={"configurable": {"langgraph_user_id": self._namespace[1]},
                    "callbacks": [self._usage]},
        )
        for put in puts:  # inserted or updated memories
            self._sources.setdefault(put["key"], []).append(event.id)

    def recall(self, query: str, *, session_id: str, at: str) -> RecallResult:
        found = self._store.search(self._namespace, query=query, limit=self.top_k)
        items = [RecalledItem(self._sources.get(m.key, []), _text(m.value), m.score) for m in found]
        return RecallResult(context="\n".join(i.text for i in items), items=items)

    def forget(self, event_ids: list[str]) -> None:
        targets = set(event_ids)
        for key, sources in list(self._sources.items()):
            if targets.intersection(sources):
                self._store.delete(self._namespace, key)
                del self._sources[key]

    def reset(self) -> None:
        for item in self._store.search(self._namespace, limit=10_000):
            self._store.delete(self._namespace, item.key)
        self._episode, self._sources = self._episode + 1, {}

    def usage(self) -> dict[str, int]:
        totals = new_usage()  # chat-model tokens; embedding tokens are not reported
        for meta in self._usage.usage_metadata.values():
            totals["llm_input_tokens"] += meta.get("input_tokens", 0)
            totals["llm_output_tokens"] += meta.get("output_tokens", 0)
        return totals

    def describe(self) -> dict[str, Any]:
        return {"llm_model": self.model, "embedding_model": self.embedding, "top_k": self.top_k}


def _text(value: dict[str, Any]) -> str:
    content = value.get("content", value)  # {"kind": "Memory", "content": {"content": ...}}
    return str(content.get("content", content) if isinstance(content, dict) else content)
