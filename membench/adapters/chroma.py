"""Baseline adapter: embed events into a local persistent Chroma
collection; recall does top-k vector search.

Uses Chroma's local `PersistentClient` and its default local embedding
function — no network calls and no API key required at run time.
"""

from __future__ import annotations

from typing import Any

from membench.core.adapter import Event, MemoryAdapter, RecallResult
from membench.core.tokens import count_tokens

try:
    import chromadb
except ImportError:  # pragma: no cover - exercised only when chromadb missing
    chromadb = None


class ChromaAdapter(MemoryAdapter):
    """Local, offline vector-store baseline (no network, no API key)."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if chromadb is None:
            raise ImportError(
                "chromadb is not installed. Install it with "
                "`pip install membench[chroma]` (or `pip install chromadb`) "
                "to use the chroma backend."
            )
        super().__init__(config)
        self.top_k = int(self.config.get("top_k", 3))
        persist_dir = self.config.get("persist_dir", ".membench_chroma")
        collection_name = self.config.get("collection_name", "membench")
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(collection_name)

    def write(self, event: Event) -> None:
        self._collection.upsert(
            ids=[event.key],
            documents=[event.content],
            metadatas=[
                {
                    "session_id": event.session_id,
                    "role": event.role,
                    "timestamp": event.timestamp,
                }
            ],
        )

    def recall(self, query: str) -> RecallResult:
        count = self._collection.count()
        if count == 0:
            return RecallResult(context="", items=[], token_count=0)
        results = self._collection.query(
            query_texts=[query], n_results=min(self.top_k, count)
        )
        documents = results.get("documents", [[]])[0]
        context = "\n".join(documents)
        return RecallResult(
            context=context,
            items=documents,
            token_count=count_tokens(context),
        )

    def forget(self, key: str) -> None:
        self._collection.delete(ids=[key])
