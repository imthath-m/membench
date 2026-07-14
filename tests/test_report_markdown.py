from membench.core.report.markdown import render_markdown


def _report(backend="full-context", pack="mini", **score_overrides):
    scores = {
        "recall_precision": 1.0,
        "update_correctness": 0.5,
        "token_cost": 42.0,
        "p95_latency_ms": 3.5,
        "semantic_recall": None,
        "contradiction_rate": None,
    }
    scores.update(score_overrides)
    return {
        "pack_name": pack,
        "backend_name": backend,
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:01+00:00",
        "traces": [
            {"step_index": 0, "step_type": "write", "session_id": "s1"},
            {"step_index": 1, "step_type": "query", "session_id": "s1"},
        ],
        "scores": scores,
    }


def test_single_report_has_metrics_and_metadata():
    md = render_markdown([_report()])
    assert "MemBench scorecard: mini" in md
    assert "full-context" in md
    assert "Recall precision" in md
    assert "100%" in md  # recall_precision 1.0
    assert "42.00" in md  # token_cost


def test_single_report_renders_none_as_na():
    md = render_markdown([_report()])
    assert "n/a" in md  # semantic_recall / contradiction_rate are None


def test_multi_report_is_side_by_side_comparison():
    md = render_markdown([_report(backend="full-context"), _report(backend="chroma")])
    assert "comparison" in md.lower()
    assert "full-context" in md
    assert "chroma" in md
    # shared pack name surfaces in the heading
    assert "MemBench comparison: mini" in md


def test_empty_reports_does_not_crash():
    assert "No reports" in render_markdown([])
