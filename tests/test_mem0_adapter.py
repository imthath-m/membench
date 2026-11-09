from pathlib import Path
from types import SimpleNamespace

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


def _event(event_id: str, content: str, session_id: str = "s1") -> Event:
    return Event(
        id=event_id,
        session_id=session_id,
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

    result = adapter.recall("What is my favorite color?", session_id="s1", at="2024-01-01T00:00:00Z")

    assert "blue" in result.context
    assert len(result.items) <= 2


def test_forget_removes_memory(adapter):
    adapter.write(_event("k1", "My favorite color is blue."))
    adapter.forget(["k1"])

    result = adapter.recall("favorite color", session_id="s1", at="2024-01-01T00:00:00Z")

    assert result.context == ""
    assert result.items == []


def test_recall_on_empty_adapter(adapter):
    result = adapter.recall("anything", session_id="s1", at="2024-01-01T00:00:00Z")

    assert result.context == ""
    assert result.items == []


def test_reset_clears_memories_for_user_id(monkeypatch):
    monkeypatch.setattr(mem0_module, "Memory", FakeMem0Client)
    adapter = mem0_module.Mem0Adapter({"user_id": "run-42"})
    adapter.write(_event("k1", "My favorite color is blue."))

    adapter.reset()

    assert adapter._client.deleted_all_for == "run-42"
    assert adapter.recall("favorite color", session_id="s1", at="2024-01-01T00:00:00Z").context == ""
    assert adapter._sources == {}


def test_missing_install_raises_clear_error(monkeypatch):
    monkeypatch.setattr(mem0_module, "Memory", None)

    with pytest.raises(ImportError, match=r"membench\[mem0\]"):
        mem0_module.Mem0Adapter()


def test_adapter_under_100_lines():
    line_count = len(Path(mem0_module.__file__).read_text().splitlines())
    assert line_count < 100


def test_mem0_declares_openai_rate_limit_and_timeout_retryable():
    openai = pytest.importorskip("openai")
    assert openai.RateLimitError in mem0_module.Mem0Adapter.retryable
    assert openai.APITimeoutError in mem0_module.Mem0Adapter.retryable


# --- usage + describe (Phase 10 cost accounting) ----------------------------

class _FakeEndpoint:
    def __init__(self, **usage):
        self.usage = SimpleNamespace(**usage)

    def create(self, **kwargs):
        return SimpleNamespace(usage=self.usage)


class MeteredFakeMem0Client(FakeMem0Client):
    """Fake `Memory` exposing `.llm` / `.embedding_model` like mem0ai does."""

    def __init__(self) -> None:
        super().__init__()
        chat = SimpleNamespace(completions=_FakeEndpoint(prompt_tokens=100, completion_tokens=20))
        self.llm = SimpleNamespace(
            config=SimpleNamespace(model="gpt-5-mini", temperature=0.1),
            client=SimpleNamespace(chat=chat, base_url="https://api.openai.com/v1/"),
        )
        self.embedding_model = SimpleNamespace(
            config=SimpleNamespace(model="text-embedding-3-small"),
            client=SimpleNamespace(embeddings=_FakeEndpoint(prompt_tokens=7)),
        )

    def add(self, content, user_id=None, metadata=None):
        self.llm.client.chat.completions.create(model="gpt-5-mini")  # fact extraction
        self.embedding_model.client.embeddings.create(model="text-embedding-3-small")
        return super().add(content, user_id=user_id, metadata=metadata)


def test_usage_counts_extraction_and_embedding_tokens(monkeypatch):
    monkeypatch.setattr(mem0_module, "Memory", MeteredFakeMem0Client)
    adapter = mem0_module.Mem0Adapter()

    adapter.write(_event("k1", "My favorite color is blue."))
    adapter.write(_event("k2", "I live in Austin."))

    assert adapter.usage() == {
        "llm_input_tokens": 200, "llm_output_tokens": 40, "embedding_tokens": 14,
    }
    assert adapter.describe() == {
        "llm_model": "gpt-5-mini",
        "llm_base_url": "https://api.openai.com/v1/",
        "temperature": 0.1,
        "embedding_model": "text-embedding-3-small",
    }


def test_usage_is_zero_without_openai_clients(adapter):
    adapter.write(_event("k1", "My favorite color is blue."))
    assert adapter.usage() == {"llm_input_tokens": 0, "llm_output_tokens": 0, "embedding_tokens": 0}
