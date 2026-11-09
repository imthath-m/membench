"""Vendor-specific behaviour of the hosted / LLM-backed adapters, on SDK fakes.

The shared contract (find / forget / reset / episode isolation) lives in
`test_contract_suite.py`; these cover each adapter's own mechanics.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import membench.adapters.cognee as cognee_module
import membench.adapters.langmem as langmem_module
import membench.adapters.letta as letta_module
import membench.adapters.mem0_platform as mem0_platform_module
import membench.adapters.zep as zep_module
from membench.adapters.letta_agent import LettaAgentAdapter
from membench.core.adapter import Event
from membench.core.cost import _usd
from membench.core.poll import drain
from tests.test_contract_suite import FACTORIES

DAY1, DAY2, DAY3 = (f"2025-03-0{d}T09:00:00+00:00" for d in (1, 2, 3))


def _event(event_id: str, content: str, timestamp: str = DAY1) -> Event:
    return Event(event_id, "s1", "user", content, timestamp)


@pytest.mark.parametrize("module, cls, extra", [
    (mem0_platform_module, "Mem0PlatformAdapter", ("MemoryClient", "mem0-platform")),
    (zep_module, "ZepAdapter", ("Zep", "zep")),
    (letta_module, "LettaAdapter", ("Letta", "letta")),
    (langmem_module, "LangMemAdapter", ("create_memory_store_manager", "langmem")),
    (cognee_module, "CogneeAdapter", ("cognee", "cognee")),
])
def test_missing_sdk_raises_clear_error(module, cls, extra, monkeypatch):
    attr, name = extra
    monkeypatch.setattr(module, attr, None)
    with pytest.raises(ImportError, match=rf"membench\[{name}\]"):
        getattr(module, cls)()


def test_hosted_and_llm_flags():
    for module, cls in [(mem0_platform_module, "Mem0PlatformAdapter"), (zep_module, "ZepAdapter")]:
        assert getattr(module, cls).deployment == "hosted" and getattr(module, cls).uses_llm
    assert letta_module.LettaAdapter.deployment == "hosted"
    assert not letta_module.LettaAdapter.uses_llm  # passages only: embeddings, no agent step
    assert LettaAgentAdapter.uses_llm
    assert langmem_module.LangMemAdapter.uses_llm and cognee_module.CogneeAdapter.uses_llm


# --- shared flush poller ------------------------------------------------------

def test_drain_polls_until_done_and_drops_failed_jobs():
    polls = {"a": 2, "b": 0}

    def done(job):
        if job == "bad":
            raise RuntimeError("failed")
        polls[job] -= 1
        return polls[job] < 0

    pending = ["a", "b"]
    drain(pending, done, interval_s=0)
    assert pending == []

    pending = ["bad", "a"]
    with pytest.raises(RuntimeError):
        drain(pending, done, interval_s=0)
    assert pending == ["a"]  # the failed job is not re-raised on the next flush


def test_drain_times_out():
    with pytest.raises(TimeoutError):
        drain(["x"], lambda job: False, interval_s=0, timeout_s=0)


# --- Mem0 Platform ------------------------------------------------------------

def test_mem0_platform_sends_metadata_timestamp_and_needs_flush(tmp_path, monkeypatch):
    adapter = FACTORIES["mem0-platform"](tmp_path, monkeypatch)
    adapter.write(_event("pet", "My dog is called Rex."))

    assert adapter.recall("dog", session_id="s1", at=DAY2).context == ""  # still processing
    adapter.flush()
    result = adapter.recall("dog", session_id="s1", at=DAY2)

    assert result.items[0].event_ids == ["pet"]
    (memory,) = adapter._client.memories
    assert memory["user_id"] == "membench-e0" and memory["metadata"] == {"membench_event_id": "pet"}


def test_mem0_platform_failed_event_fails_the_flush(tmp_path, monkeypatch):
    adapter = FACTORIES["mem0-platform"](tmp_path, monkeypatch)
    adapter.write(_event("pet", "My dog is called Rex."))
    adapter._client.client.get = lambda path: SimpleNamespace(
        raise_for_status=lambda: SimpleNamespace(json=lambda: {"status": "FAILED", "error": "x"}))

    with pytest.raises(RuntimeError, match="mem0 add failed"):
        adapter.flush()
    adapter.flush()  # the failed event is not retried forever


def test_mem0_platform_reset_moves_to_a_new_user(tmp_path, monkeypatch):
    adapter = FACTORIES["mem0-platform"](tmp_path, monkeypatch)
    adapter.reset()
    assert adapter.user_id == "membench-e1"
    retryable = pytest.importorskip("mem0.exceptions")
    assert retryable.RateLimitError in mem0_platform_module.Mem0PlatformAdapter.retryable


# --- Zep ------------------------------------------------------------------------

def test_zep_recall_hides_facts_not_yet_valid_and_shows_validity(tmp_path, monkeypatch):
    adapter = FACTORIES["zep"](tmp_path, monkeypatch)
    adapter.write(_event("old", "My dog is called Rex.", DAY1))
    adapter.write(_event("new", "My dog is called Max.", DAY3))
    adapter.flush()

    result = adapter.recall("dog", session_id="s1", at=DAY2)

    assert [item.event_ids for item in result.items] == [["old"]]
    assert result.context == f"My dog is called Rex. (valid {DAY1} to now)"


def test_zep_failed_task_fails_the_flush_and_reset_deletes_the_user(tmp_path, monkeypatch):
    adapter = FACTORIES["zep"](tmp_path, monkeypatch)
    adapter.write(_event("pet", "My dog is called Rex."))
    user = adapter._user
    adapter._client.task.get = lambda uuid: SimpleNamespace(status="failed", error="boom")

    with pytest.raises(RuntimeError, match="boom"):
        adapter.flush()
    adapter.reset()

    assert user not in adapter._client.users and adapter._user is None


# --- Letta --------------------------------------------------------------------

def test_letta_archival_dates_passages_and_adds_core_blocks(tmp_path, monkeypatch):
    adapter = FACTORIES["letta"](tmp_path, monkeypatch)
    adapter.write(_event("old", "My dog is called Rex.", DAY1))
    adapter.write(_event("new", "My dog is called Max.", DAY3))
    state = adapter._client.agents_state[adapter._agent]
    state["blocks"][0].value = "Has a dog."

    result = adapter.recall("dog", session_id="s1", at=DAY2)

    assert [item.event_ids for item in result.items] == [["old"]]  # created_at <= at
    assert result.context.startswith("<human>\nHas a dog.\n</human>\n")
    assert state["name"] == "membench-e0"
    assert adapter.usage() is None


def test_letta_agent_counts_llm_tokens_and_asks_the_agent_to_forget(tmp_path, monkeypatch):
    adapter = FACTORIES["letta-agent"](tmp_path, monkeypatch)
    adapter.write(_event("pet", "My dog is called Rex."))
    adapter.forget(["pet"])
    adapter.forget(["unknown"])  # nothing to ask about

    assert adapter.usage() == {"llm_input_tokens": 200, "llm_output_tokens": 20,
                               "embedding_tokens": 0}
    assert adapter.recall("dog", session_id="s1", at=DAY2).context == ""


# --- LangMem --------------------------------------------------------------------

def test_langmem_namespace_per_episode_and_usage(tmp_path, monkeypatch):
    adapter = FACTORIES["langmem"](tmp_path, monkeypatch)
    adapter.write(_event("pet", "My dog is called Rex."))
    assert adapter._store.items.keys() == {("membench", "membench-e0")}

    adapter.reset()
    adapter.write(_event("city", "I live in Austin."))

    assert ("membench", "membench-e1") in adapter._store.items
    assert adapter.usage() == {"llm_input_tokens": 100, "llm_output_tokens": 10,
                               "embedding_tokens": 0}
    assert adapter.describe()["llm_model"] == "openai:gpt-4o-mini"


# --- Cognee ---------------------------------------------------------------------

def test_cognee_flush_cognifies_only_new_writes(tmp_path, monkeypatch):
    adapter = FACTORIES["cognee"](tmp_path, monkeypatch)
    calls = []
    cognify = cognee_module.cognee.cognify

    async def counting_cognify(*, datasets):
        calls.append(datasets)
        await cognify(datasets=datasets)

    cognee_module.cognee.cognify = counting_cognify
    adapter.write(_event("pet", "My dog is called Rex."))
    adapter.flush()
    adapter.flush()

    assert calls == [["membench-e0"]]


def test_cognee_graph_search_returns_context_without_event_ids(tmp_path, monkeypatch):
    FACTORIES["cognee"](tmp_path, monkeypatch)  # installs the fake cognee module
    adapter = cognee_module.CogneeAdapter({"search_type": "GRAPH_COMPLETION"})
    adapter.write(_event("pet", "My dog is called Rex."))
    adapter.flush()

    result = adapter.recall("dog", session_id="s1", at=DAY2)

    assert "Rex" in result.context and result.items[0].event_ids == []


# --- cost ---------------------------------------------------------------------

def test_provider_prefixed_models_are_priced():
    prices = {"models": {"gpt-4o-mini": {"input": 1.0, "output": 2.0}}}
    for handle in ("gpt-4o-mini", "openai:gpt-4o-mini", "openai/gpt-4o-mini"):
        assert _usd(prices, handle, 1_000_000, 1_000_000) == 3.0
    assert _usd(prices, "unknown/model", 1, 0) is None


# --- published default configs -------------------------------------------------

CONFIGS = Path(__file__).resolve().parent.parent / "configs" / "default"


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.yaml")), ids=lambda p: p.stem)
def test_default_configs_equal_the_adapter_defaults(path, tmp_path, monkeypatch):
    config = yaml.safe_load(path.read_text())
    cls = type(FACTORIES[path.stem](tmp_path, monkeypatch))  # installs the SDK fake

    configured, bare = cls(config), cls({})

    assert configured.describe() == bare.describe()
    assert getattr(configured, "poll", None) == getattr(bare, "poll", None)
    if "memory_blocks" in config:
        assert config["memory_blocks"] == letta_module.DEFAULT_BLOCKS


def test_every_vendor_backend_has_a_default_config_and_doc():
    docs = Path(__file__).resolve().parent.parent / "docs"
    for backend in ("mem0-platform", "zep", "letta", "letta-agent", "langmem", "cognee"):
        assert (CONFIGS / f"{backend}.yaml").exists(), backend
    for backend in ("mem0-platform", "zep", "letta", "letta-agent", "langmem", "cognee",
                    "bm25", "oracle"):
        assert (docs / f"{backend}.md").exists(), backend
