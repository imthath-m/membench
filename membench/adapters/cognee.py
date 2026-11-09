"""Cognee adapter: `add` writes, `flush()` runs `cognify`, recall is `search`.

Cognee's API is async; every call runs on one event loop owned by the adapter.
Each episode writes to its own dataset (`<namespace>-e<n>`); `reset()` prunes
Cognee's data and system stores (graph, vectors, metadata). Recall defaults to
`CHUNKS` search (chunk payloads trace back to their source event); any other
`search_type` (e.g. `GRAPH_COMPLETION`) recalls the context its LLM would see.
"""

from __future__ import annotations

import asyncio
from typing import Any

from membench.core.adapter import Event, MemoryAdapter, RecalledItem, RecallResult

try:
    import cognee
except ImportError:  # pragma: no cover - exercised only when cognee missing
    cognee = None

try:  # Cognee's default LLM / embedder go through OpenAI-compatible clients
    import openai

    _RETRYABLE: tuple[type[BaseException], ...] = (
        openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError
    )
except ImportError:  # pragma: no cover - exercised only when openai missing
    _RETRYABLE = ()


class CogneeAdapter(MemoryAdapter):
    """Local Cognee (LLM and embedder configured through Cognee's own env vars)."""

    retryable = _RETRYABLE
    packages = ("cognee",)
    uses_llm = True

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if cognee is None:
            raise ImportError("cognee is not installed: `pip install membench[cognee]`.")
        super().__init__(config)
        self.top_k = int(self.config.get("top_k", 3))
        self.search_type = self.config.get("search_type", "CHUNKS")
        self._loop = asyncio.new_event_loop()
        self._episode, self._dirty = 0, False
        self._data: dict[str, tuple[Any, list[str]]] = {}  # data id -> (UUID, Event.ids)

    def _run(self, coro: Any) -> Any:
        return self._loop.run_until_complete(coro)

    @property
    def dataset(self) -> str:
        return f"{self.namespace}-e{self._episode}"

    def write(self, event: Event) -> None:
        info = self._run(cognee.add(f"[{event.timestamp}] {event.role}: {event.content}",
                                    dataset_name=self.dataset))
        for data_id in info.data_ids or []:  # identical content dedups to one data item
            self._data.setdefault(str(data_id), (data_id, []))[1].append(event.id)
        self._dirty = True

    def flush(self) -> None:
        if self._dirty:
            self._run(cognee.cognify(datasets=[self.dataset]))
            self._dirty = False

    def recall(self, query: str, *, session_id: str, at: str) -> RecallResult:
        if not self._data:
            return RecallResult(context="")
        results = self._run(cognee.search(
            query, query_type=cognee.SearchType[self.search_type], datasets=[self.dataset],
            top_k=self.top_k, only_context=self.search_type != "CHUNKS",  # chunks: payloads
        ))
        items = []
        for result in results:  # one entry per dataset; chunks or a context string inside
            found = result.get("search_result", result) if isinstance(result, dict) else result
            for chunk in found if isinstance(found, list) else [found]:
                chunk = chunk if isinstance(chunk, dict) else {"text": str(chunk)}
                sources = self._data.get(str(chunk.get("document_id")), (None, []))[1]
                items.append(RecalledItem(list(sources), chunk.get("text", ""), chunk.get("score")))
        return RecallResult(context="\n".join(i.text for i in items), items=items)

    def forget(self, event_ids: list[str]) -> None:
        targets = set(event_ids)
        for key, (data_id, sources) in list(self._data.items()):
            if targets.intersection(sources):
                self._run(cognee.forget(data_id=data_id, dataset=self.dataset))
                del self._data[key]

    def reset(self) -> None:
        self._run(cognee.prune.prune_data())
        self._run(cognee.prune.prune_system(metadata=True))
        self._episode, self._dirty, self._data = self._episode + 1, False, {}

    def describe(self) -> dict[str, Any]:
        return {"search_type": self.search_type, "top_k": self.top_k}
