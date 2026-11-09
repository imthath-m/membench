"""bm25 (lexical floor) and oracle (evidence-only upper bound) baselines."""

from membench.adapters.bm25 import BM25Adapter
from membench.adapters.full_context import FullContextAdapter
from membench.adapters.oracle import OracleAdapter
from membench.core.adapter import Event
from membench.core.runner import run_pack
from membench.packs.schema import builtin_pack_path, load_pack, parse_pack

AT = "2025-03-04T00:00:00+00:00"


def _event(event_id: str, content: str) -> Event:
    return Event(event_id, "s1", "user", content, "2025-03-03T00:00:00+00:00")


def test_bm25_ranks_the_lexical_match_first_and_keeps_top_k():
    adapter = BM25Adapter({"top_k": 2})
    adapter.write(_event("a", "I live in Austin."))
    adapter.write(_event("b", "My dog is called Rex and my dog likes walks."))
    adapter.write(_event("c", "Nice weather today."))

    result = adapter.recall("what is my dog called", session_id="s1", at=AT)

    assert [item.event_ids for item in result.items] == [["b"]]  # no overlap -> not returned
    assert result.items[0].score > 0
    assert result.context == "My dog is called Rex and my dog likes walks."


def test_bm25_rare_term_outranks_common_term():
    adapter = BM25Adapter({"top_k": 3})
    adapter.write(_event("common", "my my my"))
    adapter.write(_event("rare", "my plan is Premium"))
    adapter.write(_event("other", "my city is Austin"))

    result = adapter.recall("premium plan", session_id="s1", at=AT)

    assert result.items[0].event_ids == ["rare"]


def test_bm25_empty_and_describe():
    adapter = BM25Adapter()
    assert adapter.recall("anything", session_id="s1", at=AT).context == ""
    assert adapter.describe() == {"top_k": 3, "k1": 1.5, "b": 0.75}


def test_oracle_returns_only_the_evidence_events():
    adapter = OracleAdapter()
    adapter.write(_event("a", "I live in Austin."))
    adapter.write(_event("b", "My dog is called Rex."))

    adapter.evidence = ["b", "missing"]
    result = adapter.recall("anything", session_id="s1", at=AT)

    assert [item.event_ids for item in result.items] == [["b"]]
    assert result.context == "[2025-03-03T00:00:00+00:00] [user] My dog is called Rex."
    adapter.evidence = []
    assert adapter.recall("anything", session_id="s1", at=AT).context == ""


def test_runner_hands_evidence_only_to_an_oracle():
    pack = parse_pack({"name": "oracle", "steps": [
        {"type": "write", "session_id": "s1", "id": "a", "fact_key": "f_a", "role": "user",
         "content": "Austin"},
        {"type": "write", "session_id": "s1", "id": "b", "fact_key": "f_b", "role": "user",
         "content": "Rex"},
        {"type": "query", "session_id": "s1", "query": "dog?", "evidence": ["b"],
         "expect_contains": ["Rex"]},
    ]})
    full = FullContextAdapter()

    oracle_trace = run_pack(pack, OracleAdapter(), backend_name="oracle").traces[-1]
    run_pack(pack, full, backend_name="full-context")

    assert oracle_trace.detail["context"].endswith("Rex")
    assert "Austin" not in oracle_trace.detail["context"]
    assert not hasattr(full, "evidence")  # real backends never see ground truth


def test_oracle_is_an_upper_bound_on_retrieval_for_mini():
    result = run_pack(load_pack(builtin_pack_path("mini")), OracleAdapter(), backend_name="oracle")

    assert result.scores["evidence_recall_at_k"] == 1.0
    assert result.scores["recall_precision"] == 1.0
    assert result.scores["staleness"] == 0.0  # only the current value is recalled
