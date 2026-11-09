from membench.core.runner import StepTrace
from membench.core.scoring.latency import MIN_P95_SAMPLES, latency_scores, percentile


def _trace(step_type: str, latency_ms: float, ingest_ms=None, error=None) -> StepTrace:
    detail = {"ingest_latency_ms": ingest_ms} if step_type == "query" else {}
    return StepTrace(0, step_type, "s1", latency_ms, detail, error=error)


def test_percentile_nearest_rank():
    values = [float(i) for i in range(1, 21)]
    assert percentile(values, 0.95) == 19.0
    assert percentile(values, 0.5) == 10.0
    assert percentile([42.0], 0.5) == 42.0
    assert percentile([], 0.5) is None


def test_stages_are_separate_and_errors_excluded():
    traces = [
        _trace("write", 1.0), _trace("write", 3.0), _trace("write", 500.0, error="boom"),
        _trace("query", 10.0, ingest_ms=7.0), _trace("query", 20.0),
        _trace("forget", 99.0),
    ]
    scores = latency_scores(traces)
    assert scores["write_p50_ms"] == 1.0
    assert scores["recall_p50_ms"] == 10.0
    assert scores["ingest_p50_ms"] == 7.0


def test_p95_is_none_below_min_samples():
    few = latency_scores([_trace("write", float(i)) for i in range(MIN_P95_SAMPLES - 1)])
    assert few["write_p95_ms"] is None and few["write_p50_ms"] is not None
    many = latency_scores([_trace("write", float(i)) for i in range(1, MIN_P95_SAMPLES + 1)])
    assert many["write_p95_ms"] == 95.0


def test_no_traces_scores_none():
    assert all(v is None for v in latency_scores([]).values())
