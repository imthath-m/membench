from pathlib import Path

import pytest

import membench.adapters.mem0 as mem0_module
from membench.core.adapter import Event


class FakeMem0Client:
    """Stand-in for `mem0.Memory` — no network, in-memory store only."""

    def __init__(self) -> None:
        self._store: list[dict] = []

    @classmethod
    def from_config(cls, config: dict) -> "FakeMem0Client":
        return cls()

    def add(self, content, user_id=None, metadata=None):
        memory_id = f"m{len(self._store)}"
        self._store.append({"id": memory_id, "memory": content})
        return {"results": [{"id": memory_id, "memory": content}]}

    def search(self, query, *, top_k=20, filters=None):  # mem0ai>=2.0 signature
        assert filters and "user_id" in filters
        return {"results": self._store[:top_k]}

    def delete(self, memory_id):
        self._store = [m for m in self._store if m["id"] != memory_id]

    def delete_all(self, user_id=None):
        self.deleted_all_for = user_id
        self._store = []


def _event(key: str, content: str, session_id: str = "s1") -> Event:
    return Event(
        session_id=session_id,
        key=key,
        role="user",
        content=content,
        timestamp="2024-01-01T00:00:00Z",
    )


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(mem0_module, "Memory", FakeMem0Client)
    return mem0_module.Mem0Adapter({"top_k": 2})


def test_write_and_recall(adapter):
    adapter.write(_event("k1", "My favorite color is blue."))
    adapter.write(_event("k2", "I live in Austin."))

    result = adapter.recall("What is my favorite color?")

    assert "blue" in result.context
    assert result.token_count > 0
    assert len(result.items) <= 2


def test_forget_removes_memory(adapter):
    adapter.write(_event("k1", "My favorite color is blue."))
    adapter.forget("k1")

    result = adapter.recall("favorite color")

    assert result.context == ""
    assert result.items == []


def test_recall_on_empty_adapter(adapter):
    result = adapter.recall("anything")

    assert result.context == ""
    assert result.items == []
    assert result.token_count == 0


def test_reset_clears_memories_for_user_id(monkeypatch):
    monkeypatch.setattr(mem0_module, "Memory", FakeMem0Client)
    adapter = mem0_module.Mem0Adapter({"user_id": "run-42"})
    adapter.write(_event("k1", "My favorite color is blue."))

    adapter.reset()

    assert adapter._client.deleted_all_for == "run-42"
    assert adapter.recall("favorite color").context == ""
    assert adapter._ids == {}


def test_missing_install_raises_clear_error(monkeypatch):
    monkeypatch.setattr(mem0_module, "Memory", None)

    with pytest.raises(ImportError, match=r"membench\[mem0\]"):
        mem0_module.Mem0Adapter()


def test_adapter_under_100_lines():
    line_count = len(Path(mem0_module.__file__).read_text().splitlines())
    assert line_count < 100
