from membench.core.runner import StepTrace
from membench.core.scoring.token_cost import token_cost


def _trace(step_type: str, token_count: int = 0) -> StepTrace:
    detail = {"token_count": token_count} if step_type == "query" else {}
    return StepTrace(
        step_index=0, step_type=step_type, session_id="s1", latency_ms=1.0, detail=detail
    )


def test_mean_of_query_token_counts():
    traces = [_trace("query", 10), _trace("query", 20)]

    assert token_cost(traces) == 15.0


def test_write_traces_are_ignored():
    traces = [_trace("write"), _trace("query", 10)]

    assert token_cost(traces) == 10.0


def test_no_queries_returns_none():
    traces = [_trace("write")]

    assert token_cost(traces) is None


def test_no_traces_returns_none():
    assert token_cost([]) is None
