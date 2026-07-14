from membench.adapters.full_context import FullContextAdapter
from membench.core.runner import RunResult, StepTrace, run_pack
from membench.packs.schema import builtin_pack_path, load_pack


def test_run_pack_produces_a_trace_per_step():
    pack = load_pack(builtin_pack_path("mini"))
    adapter = FullContextAdapter()

    result = run_pack(pack, adapter, backend_name="full-context")

    assert isinstance(result, RunResult)
    assert result.pack_name == "mini"
    assert result.backend_name == "full-context"
    assert len(result.traces) == len(pack.steps)
    assert all(isinstance(t, StepTrace) for t in result.traces)
    assert result.scores == {}


def test_query_traces_carry_latency_and_tokens():
    pack = load_pack(builtin_pack_path("mini"))
    adapter = FullContextAdapter()

    result = run_pack(pack, adapter, backend_name="full-context")
    query_traces = [t for t in result.traces if t.step_type == "query"]

    assert len(query_traces) == 3
    for trace in query_traces:
        assert trace.latency_ms >= 0
        assert "token_count" in trace.detail
        assert "context" in trace.detail


def test_full_context_recalls_updated_fact():
    pack = load_pack(builtin_pack_path("mini"))
    adapter = FullContextAdapter()

    result = run_pack(pack, adapter, backend_name="full-context")
    last_query = [t for t in result.traces if t.step_type == "query"][-2]

    assert "green" in last_query.detail["context"]
