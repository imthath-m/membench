"""Render RunResult JSON report(s) as Markdown.

Reports derive strictly from the JSON `scores` block and run metadata — no
rescoring happens here (the runner is the source of truth). A single report
becomes a scorecard table, a per-category table and a one-line-per-metric
breakdown; multiple reports become a side-by-side backend/pack comparison,
one column each, plus recall precision by category.
"""

from __future__ import annotations

from membench.core.runner import MAX_ERROR_RATE

# (key, human label, kind) — `kind` drives value formatting. `ratio` metrics
# live in 0..1 and render as percentages; everything else is a plain number.
_METRICS: list[tuple[str, str, str]] = [
    ("recall_precision", "Recall precision", "ratio"),
    ("update_correctness", "Update correctness", "ratio"),
    ("staleness", "Staleness", "ratio"),
    ("temporal_accuracy", "Temporal accuracy", "ratio"),
    ("forget_compliance", "Forget compliance", "ratio"),
    ("evidence_recall_at_k", "Evidence recall@k", "ratio"),
    ("abstention", "Abstention", "ratio"),
    ("semantic_recall", "Semantic recall", "ratio"),
    ("contradiction_rate", "Contradiction rate", "ratio"),
    ("qa_accuracy", "QA accuracy", "ratio"),
    ("token_cost", "Token cost (avg tokens/query)", "number"),
    ("write_p50_ms", "Write p50 (ms)", "number"),
    ("write_p95_ms", "Write p95 (ms)", "number"),
    ("ingest_p50_ms", "Ingest p50 (ms)", "number"),
    ("ingest_p95_ms", "Ingest p95 (ms)", "number"),
    ("recall_p50_ms", "Recall p50 (ms)", "number"),
    ("recall_p95_ms", "Recall p95 (ms)", "number"),
    ("error_rate", "Error rate", "ratio"),
]

_P95_NOTE = "95th percentile in milliseconds (n/a below 100 samples)"
_METRIC_NOTES: dict[str, str] = {
    "recall_precision": "fraction of expect_contains queries whose recall matched",
    "update_correctness": "fraction of post-update queries reflecting the new fact",
    "staleness": "fraction of post-update queries still recalling a stale value (lower is better)",
    "temporal_accuracy": "fraction of temporal queries whose recall matched",
    "forget_compliance": "fraction of post-forget queries free of the forgotten content",
    "evidence_recall_at_k": "fraction of evidence writes among the recalled items",
    "abstention": "fraction of unanswerable queries the reader declined (n/a without reader)",
    "semantic_recall": "judge score for semantic recall (n/a when judge is off)",
    "contradiction_rate": "judge rate of contradictions in recall (n/a when judge is off)",
    "qa_accuracy": "judge score of reader answers vs gold (n/a without reader + judge)",
    "token_cost": "mean recalled-context tokens per query (`approx`: whitespace count)",
    "write_p50_ms": "median write latency in milliseconds",
    "write_p95_ms": _P95_NOTE,
    "ingest_p50_ms": "median flush (async ingest) latency before a query, in milliseconds",
    "ingest_p95_ms": _P95_NOTE,
    "recall_p50_ms": "median recall latency in milliseconds",
    "recall_p95_ms": _P95_NOTE,
    "error_rate": (
        "fraction of steps that failed or timed out "
        f"(run invalid above {MAX_ERROR_RATE:.0%})"
    ),
}


# (key, label, kind) columns of the per-category table.
_CATEGORY_METRICS: list[tuple[str, str, str]] = [
    ("n", "Queries", "count"),
    ("recall_precision", "Recall precision", "ratio"),
    ("evidence_recall_at_k", "Evidence recall@k", "ratio"),
    ("semantic_recall", "Semantic recall", "ratio"),
    ("qa_accuracy", "QA accuracy", "ratio"),
]


def _fmt_value(kind: str, value: object) -> str:
    if value is None:
        return "n/a"
    if kind == "ratio":
        return f"{float(value) * 100:.0f}%"
    if kind == "count":
        return str(value)
    return f"{float(value):.2f}"


def _fmt_score(report: dict, key: str, kind: str) -> str:
    """A score cell; token cost from the whitespace fallback is marked `approx`."""
    text = _fmt_value(kind, report.get("scores", {}).get(key))
    tokenizer = str(report.get("manifest", {}).get("tokenizer", ""))
    if key == "token_cost" and text != "n/a" and "approx" in tokenizer:
        text += " (approx)"
    return text


def _categories(reports: list[dict]) -> list[str]:
    return sorted({c for r in reports for c in r.get("by_category", {})})


def _category_cell(report: dict, category: str, key: str, kind: str) -> str:
    return _fmt_value(kind, report.get("by_category", {}).get(category, {}).get(key))


def _count_queries(report: dict) -> int:
    return sum(1 for t in report.get("traces", []) if t.get("step_type") == "query")


def _fmt_valid(report: dict) -> str:
    return "yes" if report.get("valid", True) else "no (error rate above threshold)"


def _label(report: dict) -> str:
    return f"{report.get('backend_name', '?')}"


def render_markdown(reports: list[dict]) -> str:
    """Render one or more RunResult-shaped dicts as a Markdown document."""
    if not reports:
        return "# MemBench report\n\n_No reports provided._\n"
    if len(reports) == 1:
        return _render_single(reports[0])
    return _render_comparison(reports)


def _render_single(report: dict) -> str:
    lines = [
        f"# MemBench scorecard: {report.get('pack_name', '?')}",
        "",
        f"- **Backend:** {report.get('backend_name', '?')}",
        f"- **Started:** {report.get('started_at', '?')}",
        f"- **Finished:** {report.get('finished_at', '?')}",
        (
            f"- **Steps:** {len(report.get('traces', []))} "
            f"({_count_queries(report)} queries)"
        ),
        f"- **Valid:** {_fmt_valid(report)}",
        f"- **Tokenizer:** {report.get('manifest', {}).get('tokenizer', '?')}",
        "",
        "## Scores",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for key, label, kind in _METRICS:
        lines.append(f"| {label} | {_fmt_score(report, key, kind)} |")

    categories = _categories([report])
    if categories:
        lines += [
            "", "## By category", "",
            "| Category | " + " | ".join(label for _, label, _ in _CATEGORY_METRICS) + " |",
            "| --- | " + " | ".join("---" for _ in _CATEGORY_METRICS) + " |",
        ]
        for category in categories:
            cells = [_category_cell(report, category, k, kind) for k, _, kind in _CATEGORY_METRICS]
            lines.append(f"| {category} | " + " | ".join(cells) + " |")

    lines += ["", "## Breakdown", ""]
    for key, label, _ in _METRICS:
        lines.append(f"- **{label}:** {_METRIC_NOTES[key]}")
    lines.append("")
    return "\n".join(lines)


def _render_comparison(reports: list[dict]) -> str:
    packs = {r.get("pack_name") for r in reports}
    if len(packs) == 1:
        heading = f"# MemBench comparison: {next(iter(packs))}"
    else:
        heading = "# MemBench comparison"

    labels = [_label(r) for r in reports]
    lines = [
        heading,
        "",
        "| Metric | " + " | ".join(labels) + " |",
        "| --- | " + " | ".join("---" for _ in labels) + " |",
    ]
    for key, label, kind in _METRICS:
        cells = [_fmt_score(r, key, kind) for r in reports]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    categories = _categories(reports)
    if categories:
        lines += [
            "", "## Recall precision by category", "",
            "| Category | " + " | ".join(labels) + " |",
            "| --- | " + " | ".join("---" for _ in labels) + " |",
        ]
        for category in categories:
            cells = [_category_cell(r, category, "recall_precision", "ratio") for r in reports]
            lines.append(f"| {category} | " + " | ".join(cells) + " |")

    lines += ["", "## Runs", ""]
    for r in reports:
        lines.append(
            f"- **{r.get('backend_name', '?')}** on `{r.get('pack_name', '?')}` — "
            f"{len(r.get('traces', []))} steps ({_count_queries(r)} queries)"
        )
    lines.append("")
    return "\n".join(lines)
