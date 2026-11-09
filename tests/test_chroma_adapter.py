import pytest

from membench.core.adapter import Event

chromadb = pytest.importorskip("chromadb", reason="chromadb not installed")

from membench.adapters.chroma import ChromaAdapter


def _event(event_id: str, content: str, session_id: str = "s1") -> Event:
    return Event(
        id=event_id,
        session_id=session_id,
        role="user",
        content=content,
        timestamp="2024-01-01T00:00:00Z",
    )


@pytest.fixture
def adapter(tmp_path):
    return ChromaAdapter({"persist_dir": str(tmp_path / "chroma"), "top_k": 2})


def test_write_and_recall_top_k(adapter):
    adapter.write(_event("k1", "My favorite color is blue."))
    adapter.write(_event("k2", "I live in Austin."))

    result = adapter.recall("What is my favorite color?", session_id="s1", at="2024-01-01T00:00:00Z")

    assert result.context != ""
    assert len(result.items) <= 2


def test_forget_removes_from_index(adapter):
    adapter.write(_event("k1", "My favorite color is blue."))
    adapter.forget(["k1"])

    result = adapter.recall("favorite color", session_id="s1", at="2024-01-01T00:00:00Z")

    assert result.context == ""
    assert result.items == []


def test_recall_on_empty_adapter(adapter):
    result = adapter.recall("anything", session_id="s1", at="2024-01-01T00:00:00Z")

    assert result.context == ""
    assert result.items == []
