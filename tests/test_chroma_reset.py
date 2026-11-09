"""Chroma reset() against a stubbed chromadb (runs without the chroma extra)."""

import membench.adapters.chroma as chroma_module


class FakeCollection:
    def __init__(self) -> None:
        self.ids: list[str] = []

    def upsert(self, ids, documents, metadatas):
        self.ids.extend(ids)


class FakePersistentClient:
    def __init__(self, path):
        self.collections: dict[str, FakeCollection] = {}
        self.deleted: list[str] = []

    def get_or_create_collection(self, name):
        return self.collections.setdefault(name, FakeCollection())

    def delete_collection(self, name):
        self.deleted.append(name)
        del self.collections[name]


class FakeChromadb:
    PersistentClient = FakePersistentClient


def test_reset_recreates_collection(monkeypatch):
    monkeypatch.setattr(chroma_module, "chromadb", FakeChromadb)
    adapter = chroma_module.ChromaAdapter({"collection_name": "c1"})
    adapter.write(
        chroma_module.Event(
            id="e1", session_id="s1", role="user", content="x", timestamp="t"
        )
    )

    adapter.reset()

    assert adapter._client.deleted == ["c1"]
    assert adapter._collection.ids == []
