"""Shared adapter contract suite, run against every registered backend.

Offline adapters (full-context, chroma, bm25, oracle) run for real; vendor
adapters run on the in-memory SDK fakes in `tests/vendor_fakes.py`. Each
backend must: find a written event after flush, drop a forgotten event, come
back empty after reset, and keep episodes isolated.
"""

from __future__ import annotations

import pytest

import membench.adapters.cognee as cognee_module
import membench.adapters.langmem as langmem_module
import membench.adapters.letta as letta_module
import membench.adapters.mem0 as mem0_module
import membench.adapters.mem0_platform as mem0_platform_module
import membench.adapters.zep as zep_module
from membench.adapters.bm25 import BM25Adapter
from membench.adapters.full_context import FullContextAdapter
from membench.adapters.letta_agent import LettaAgentAdapter
from membench.adapters.oracle import OracleAdapter
from membench.core.adapter import Event
from membench.core.registry import list_backends
from membench.core.runner import run_pack
from membench.packs.schema import builtin_pack_path, load_pack, parse_pack
from tests.test_mem0_adapter import FakeMem0Client
from tests.vendor_fakes import (
    FAKE_ZEP_MODULE,
    FakeCognee,
    FakeLetta,
    FakeMemoryClient,
    FakeStore,
    FakeUsageHandler,
    FakeZep,
    fake_memory_store_manager,
)

FAST_POLL = {"poll_interval_s": 0, "flush_timeout_s": 5}


def _chroma(tmp_path, monkeypatch):
    pytest.importorskip("chromadb", reason="chromadb not installed")
    from membench.adapters.chroma import ChromaAdapter

    return ChromaAdapter({"persist_dir": str(tmp_path / "chroma"), "top_k": 3})


def _mem0(tmp_path, monkeypatch):
    monkeypatch.setattr(mem0_module, "Memory", FakeMem0Client)
    return mem0_module.Mem0Adapter({"top_k": 3})


def _mem0_platform(tmp_path, monkeypatch):
    monkeypatch.setattr(mem0_platform_module, "MemoryClient", FakeMemoryClient)
    return mem0_platform_module.Mem0PlatformAdapter(FAST_POLL)


def _zep(tmp_path, monkeypatch):
    monkeypatch.setattr(zep_module, "Zep", FakeZep)
    monkeypatch.setattr(zep_module, "zep_cloud", FAKE_ZEP_MODULE, raising=False)
    return zep_module.ZepAdapter(FAST_POLL)


def _letta(tmp_path, monkeypatch, cls=letta_module.LettaAdapter):
    monkeypatch.setattr(letta_module, "Letta", FakeLetta)
    return cls()


def _letta_agent(tmp_path, monkeypatch):
    return _letta(tmp_path, monkeypatch, LettaAgentAdapter)


def _langmem(tmp_path, monkeypatch):
    monkeypatch.setattr(langmem_module, "create_memory_store_manager", fake_memory_store_manager)
    monkeypatch.setattr(langmem_module, "InMemoryStore", FakeStore, raising=False)
    monkeypatch.setattr(langmem_module, "UsageMetadataCallbackHandler", FakeUsageHandler,
                        raising=False)
    return langmem_module.LangMemAdapter()


def _cognee(tmp_path, monkeypatch):
    monkeypatch.setattr(cognee_module, "cognee", FakeCognee())
    return cognee_module.CogneeAdapter()


FACTORIES = {
    "full-context": lambda tmp_path, monkeypatch: FullContextAdapter(),
    "chroma": _chroma,
    "bm25": lambda tmp_path, monkeypatch: BM25Adapter(),
    "oracle": lambda tmp_path, monkeypatch: OracleAdapter(),
    "mem0": _mem0,
    "mem0-platform": _mem0_platform,
    "zep": _zep,
    "letta": _letta,
    "letta-agent": _letta_agent,
    "langmem": _langmem,
    "cognee": _cognee,
}
# Recalled items trace to no Event.id: the agent writes its own notes.
UNTRACEABLE = {"letta-agent"}

BACKENDS = pytest.mark.parametrize("backend", sorted(FACTORIES))

PET = {"type": "write", "session_id": "s1", "id": "pet", "fact_key": "fact_pet", "role": "user",
       "content": "My dog is called Rex.", "timestamp": "2025-03-03T09:00:00+00:00"}
CITY = {"type": "write", "session_id": "s1", "id": "city", "fact_key": "fact_city",
        "role": "user", "content": "I live in Austin.", "timestamp": "2025-03-03T09:01:00+00:00"}


def _query(text: str, evidence: list[str], expect: list[str]) -> dict:
    return {"type": "query", "session_id": "s1", "query": text, "evidence": evidence,
            "expect_contains": expect, "timestamp": "2025-03-04T09:00:00+00:00"}


def _run(backend, tmp_path, monkeypatch, pack: dict):
    adapter = FACTORIES[backend](tmp_path, monkeypatch)
    adapter.reset()  # as `membench run` does before replaying a pack
    result = run_pack(parse_pack(pack), adapter, backend_name=backend)
    assert all(t.error is None for t in result.traces), [t.error for t in result.traces]
    return [t for t in result.traces if t.step_type == "query"]


def _recalled_ids(trace) -> set[str]:
    return {i for item in trace.detail["items"] for i in item["event_ids"]}


def test_every_registered_backend_is_in_the_suite():
    assert sorted(FACTORIES) == list_backends()


@BACKENDS
def test_write_flush_recall_finds_it(backend, tmp_path, monkeypatch):
    pack = {"name": "contract_find", "steps": [
        PET, CITY, _query("What is my dog called?", ["pet"], ["Rex"]),
    ]}

    (query,) = _run(backend, tmp_path, monkeypatch, pack)

    assert "Rex" in query.detail["context"]
    assert query.detail["ingest_latency_ms"] is not None  # flush ran before recall
    if backend not in UNTRACEABLE:
        assert "pet" in _recalled_ids(query)


@BACKENDS
def test_forget_removes_it(backend, tmp_path, monkeypatch):
    pack = {"name": "contract_forget", "steps": [
        PET, CITY, {"type": "forget", "session_id": "s1", "target": "pet"},
        _query("Where do I live and what is my dog called?", ["city"], ["Austin"]),
    ]}

    (query,) = _run(backend, tmp_path, monkeypatch, pack)

    assert "Rex" not in query.detail["context"]
    assert "Austin" in query.detail["context"]
    assert "pet" not in _recalled_ids(query)


@BACKENDS
def test_reset_empties(backend, tmp_path, monkeypatch):
    adapter = FACTORIES[backend](tmp_path, monkeypatch)
    adapter.reset()
    adapter.write(Event("pet", "s1", "user", "My dog is called Rex.", "2025-03-03T09:00:00+00:00"))
    adapter.flush()

    adapter.reset()
    adapter.flush()
    if adapter.sees_evidence:
        adapter.evidence = ["pet"]
    result = adapter.recall("What is my dog called?", session_id="s1",
                            at="2025-03-04T09:00:00+00:00")

    assert "Rex" not in result.context
    assert result.items == []


@BACKENDS
def test_episode_isolation(backend, tmp_path, monkeypatch):
    pack = {"name": "contract_episodes", "episodes": [
        {"steps": [PET, _query("What is my dog called?", ["pet"], ["Rex"])]},
        {"steps": [CITY, _query("Where do I live and what is my dog called?", ["city"],
                                ["Austin"])]},
    ]}

    first, second = _run(backend, tmp_path, monkeypatch, pack)

    assert "Rex" in first.detail["context"]
    assert "Austin" in second.detail["context"]
    assert "Rex" not in second.detail["context"]  # nothing leaks across reset()


@BACKENDS
def test_mini_pack_runs_clean(backend, tmp_path, monkeypatch):
    adapter = FACTORIES[backend](tmp_path, monkeypatch)
    adapter.reset()

    result = run_pack(load_pack(builtin_pack_path("mini")), adapter, backend_name=backend)

    assert result.scores["error_rate"] == 0.0
    assert result.scores["recall_precision"] is not None
