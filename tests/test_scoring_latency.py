from membench.core.runner import StepTrace
from membench.core.scoring.latency import p95_latency_ms


def _trace(latency_ms: float) -> StepTrace:
    return StepTrace(step_index=0, step_type="query", session_id="s1", latency_ms=latency_ms, detail={})


def test_p95_of_twenty_evenly_spaced_values():
    # 1..20 ms, nearest-rank p95 -> the 19th smallest value.
    traces = [_trace(float(i)) for i in range(1, 21)]

    assert p95_latency_ms(traces) == 19.0


def test_single_value():
    traces = [_trace(42.0)]

    assert p95_latency_ms(traces) == 42.0


def test_no_traces_returns_none():
    assert p95_latency_ms([]) is None


def test_includes_write_and_query_latencies():
    traces = [_trace(1.0), _trace(100.0)]

    assert p95_latency_ms(traces) == 100.0
