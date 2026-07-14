from membench.core.report.html import render_html


def _report(backend="full-context", pack="mini"):
    return {
        "pack_name": pack,
        "backend_name": backend,
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:01+00:00",
        "traces": [
            {"step_index": 0, "step_type": "write", "session_id": "s1"},
            {"step_index": 1, "step_type": "query", "session_id": "s1"},
        ],
        "scores": {
            "recall_precision": 1.0,
            "update_correctness": 0.5,
            "token_cost": 42.0,
            "p95_latency_ms": 3.5,
            "semantic_recall": None,
            "contradiction_rate": None,
        },
    }


def test_single_html_is_self_contained_document():
    html = render_html([_report()])
    assert html.lstrip().startswith("<!doctype html>")
    assert "<style>" in html
    # no external resources
    assert "src=" not in html
    assert "href=" not in html
    assert "cdn" not in html.lower()
    assert "<script" not in html.lower()


def test_single_html_has_metrics_and_na():
    html = render_html([_report()])
    assert "Recall precision" in html
    assert "100%" in html
    assert "n/a" in html  # None metrics


def test_multi_html_comparison_names_both_backends():
    html = render_html([_report(backend="full-context"), _report(backend="chroma")])
    assert "full-context" in html
    assert "chroma" in html
    assert "comparison" in html.lower()
    # still self-contained
    assert "src=" not in html and "<script" not in html.lower()
