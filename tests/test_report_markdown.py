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


def test_single_report_shows_error_rate_and_invalid_flag():
    report = _report(error_rate=0.25)
    report["valid"] = False
    md = render_markdown([report])
    assert "| Error rate | 25% |" in md
    assert "**Valid:** no" in md


def _categorized(backend="full-context", tokenizer="whitespace:approx"):
    report = _report(backend=backend)
    report["manifest"] = {"tokenizer": tokenizer}
    report["by_category"] = {
        "temporal": {"n": 2, "recall_precision": 0.5, "evidence_recall_at_k": 1.0,
                     "semantic_recall": None, "qa_accuracy": None},
        "update": {"n": 3, "recall_precision": 1.0, "evidence_recall_at_k": None,
                   "semantic_recall": None, "qa_accuracy": 0.25},
    }
    return report


def test_single_report_has_per_category_table_and_approx_tokens():
    md = render_markdown([_categorized()])
    assert "## By category" in md
    assert "| temporal | 2 | 50% | 100% | n/a | n/a |" in md
    assert "| update | 3 | 100% | n/a | n/a | 25% |" in md
    assert "42.00 (approx)" in md
    assert "**Tokenizer:** whitespace:approx" in md
    assert "(approx)" not in render_markdown([_categorized(tokenizer="tiktoken:o200k_base")])


def test_comparison_has_recall_precision_by_category():
    md = render_markdown([_categorized("a"), _categorized("b")])
    assert "## Recall precision by category" in md
    assert "| temporal | 50% | 50% |" in md
