"""Baseline adapter: embed events into a local persistent Chroma
collection; recall does top-k vector search.

Uses Chroma's local `PersistentClient` and its default local embedding
function — no network calls and no API key required at run time.
"""

from __future__ import annotations

from typing import Any

from membench.core.adapter import Event, MemoryAdapter, RecalledItem, RecallResult

try:
    import chromadb
except ImportError:  # pragma: no cover - exercised only when chromadb missing
    chromadb = None


class ChromaAdapter(MemoryAdapter):
    """Local, offline vector-store baseline (no network, no API key)."""

    packages = ("chromadb",)

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
        self._collection_name = self.config.get("collection_name", self.namespace)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(self._collection_name)

    def write(self, event: Event) -> None:
        self._collection.upsert(
            ids=[event.id],
            documents=[event.content],
            metadatas=[
                {
                    "session_id": event.session_id,
                    "role": event.role,
                    "timestamp": event.timestamp,
                }
            ],
        )

    def recall(self, query: str, *, session_id: str, at: str) -> RecallResult:
        count = self._collection.count()
        if count == 0:
            return RecallResult(context="")
        results = self._collection.query(
            query_texts=[query], n_results=min(self.top_k, count)
        )
        ids, documents = results["ids"][0], results["documents"][0]
        distances = results["distances"][0]
        items = [
            RecalledItem(event_ids=[i], text=doc, score=-float(dist))  # closer = higher
            for i, doc, dist in zip(ids, documents, distances)
        ]
        return RecallResult(context="\n".join(documents), items=items)

    def forget(self, event_ids: list[str]) -> None:
        self._collection.delete(ids=list(event_ids))

    def reset(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(self._collection_name)

    def describe(self) -> dict[str, Any]:
        # The collection uses Chroma's default embedder: the ONNX all-MiniLM-L6-v2.
        return {"embedding_model": "all-MiniLM-L6-v2", "top_k": self.top_k}
