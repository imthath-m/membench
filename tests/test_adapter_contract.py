"""Contract v2 checks run against all 3 adapters (mem0 via a fake client)."""

from pathlib import Path

import pytest

import membench.adapters.mem0 as mem0_module
from membench.adapters.full_context import FullContextAdapter
from membench.core.runner import run_pack
from membench.packs.schema import parse_pack
from tests.test_mem0_adapter import FakeMem0Client

ADAPTERS_DIR = Path(mem0_module.__file__).parent

UPDATE_PACK = {
    "name": "update_resolution",
    "steps": [
        {"type": "write", "session_id": "s1", "fact_key": "fact_plan", "role": "user",
         "content": "I'm on the Basic plan."},
        {"type": "write", "session_id": "s2", "fact_key": "fact_plan", "role": "user",
         "content": "We upgraded the account to Premium last week.", "fact_type": "update"},
        {"type": "query", "session_id": "s2", "query": "Which plan is the customer on?",
         "expect_contains": ["Premium"], "expect_key": "fact_plan"},
    ],
}

EPISODES_PACK = {
    "name": "forget_episodes",
    "episodes": [
        {"steps": [
            {"type": "write", "session_id": "s1", "id": "pet", "fact_key": "fact_pet",
             "role": "user", "content": "My dog is called Rex."},
            {"type": "write", "session_id": "s1", "id": "city", "fact_key": "fact_city",
             "role": "user", "content": "I live in Austin."},
            {"type": "forget", "session_id": "s1", "target": "pet"},
            {"type": "query", "session_id": "s1", "query": "Where do I live?",
             "expect_contains": ["Austin"], "expect_key": "fact_city"},
        ]},
        {"steps": [
            {"type": "write", "session_id": "s2", "fact_key": "fact_food", "role": "user",
             "content": "My favorite food is ramen."},
            {"type": "query", "session_id": "s2", "query": "What is my favorite food?",
             "expect_contains": ["ramen"], "expect_key": "fact_food"},
        ]},
    ],
}


def _full_context(tmp_path, monkeypatch):
    return FullContextAdapter()


def _chroma(tmp_path, monkeypatch):
    pytest.importorskip("chromadb", reason="chromadb not installed")
    from membench.adapters.chroma import ChromaAdapter

    return ChromaAdapter({"persist_dir": str(tmp_path / "chroma"), "top_k": 5})


def _mem0(tmp_path, monkeypatch):
    monkeypatch.setattr(mem0_module, "Memory", FakeMem0Client)
    return mem0_module.Mem0Adapter({"top_k": 5})


ADAPTERS = pytest.mark.parametrize("make", [_full_context, _chroma, _mem0],
                                   ids=["full-context", "chroma", "mem0"])


def _recording(adapter):
    """Record every Event the adapter receives, then pass it through."""
    adapter.received = []
    write = adapter.write

    def recording_write(event):
        adapter.received.append(event)
        write(event)

    adapter.write = recording_write
    return adapter


def test_every_adapter_file_is_under_100_lines():
    files = sorted(ADAPTERS_DIR.glob("*.py"))
    assert len(files) >= 4  # __init__ + 3 adapters
    for path in files:
        assert len(path.read_text().splitlines()) < 100, path.name


@ADAPTERS
def test_update_resolution_is_left_to_the_backend(make, tmp_path, monkeypatch):
    pack = parse_pack(UPDATE_PACK)
    adapter = _recording(make(tmp_path, monkeypatch))

    result = run_pack(pack, adapter, backend_name="contract")

    ids = [e.id for e in adapter.received]
    assert len(ids) == 2 and len(set(ids)) == 2  # the update got its own id
    assert all("fact_plan" not in repr(e) for e in adapter.received)  # no fact_key leak
    query = result.traces[-1]
    assert query.error is None
    assert "Basic" in query.detail["context"] and "Premium" in query.detail["context"]
    recalled = {i for item in query.detail["items"] for i in item["event_ids"]}
    assert recalled == set(ids)  # items map back to Event.id


@ADAPTERS
def test_forget_and_two_episodes_run(make, tmp_path, monkeypatch):
    pack = parse_pack(EPISODES_PACK)
    adapter = make(tmp_path, monkeypatch)

    result = run_pack(pack, adapter, backend_name="contract")

    assert all(t.error is None for t in result.traces)
    assert [t.episode for t in result.traces] == [0, 0, 0, 0, 1, 1]
    first_query, second_query = (t for t in result.traces if t.step_type == "query")
    assert "Rex" not in first_query.detail["context"]  # forgotten
    assert "Austin" in first_query.detail["context"]
    assert "ramen" in second_query.detail["context"]
    assert "Austin" not in second_query.detail["context"]  # reset between episodes
    assert result.scores["recall_precision"] == 1.0
