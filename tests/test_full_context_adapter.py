from membench.adapters.full_context import FullContextAdapter
from membench.core.adapter import Event


def _event(key: str, content: str, session_id: str = "s1") -> Event:
    return Event(
        session_id=session_id,
        key=key,
        role="user",
        content=content,
        timestamp="2024-01-01T00:00:00Z",
    )


def test_write_and_recall_returns_full_log():
    adapter = FullContextAdapter()
    adapter.write(_event("k1", "hello"))
    adapter.write(_event("k2", "world"))

    result = adapter.recall("anything")

    assert "hello" in result.context
    assert "world" in result.context
    assert result.token_count > 0
    assert len(result.items) == 2


def test_forget_removes_event_from_context():
    adapter = FullContextAdapter()
    adapter.write(_event("k1", "hello"))
    adapter.write(_event("k2", "world"))

    adapter.forget("k1")
    result = adapter.recall("anything")

    assert "hello" not in result.context
    assert "world" in result.context


def test_recall_on_empty_adapter():
    adapter = FullContextAdapter()
    result = adapter.recall("anything")

    assert result.context == ""
    assert result.items == []
    assert result.token_count == 0
