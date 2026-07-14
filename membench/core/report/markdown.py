"""Render RunResult JSON report(s) as Markdown.

Reports derive strictly from the JSON `scores` block and run metadata — no
rescoring happens here (the runner is the source of truth). A single report
becomes a scorecard table plus a one-line-per-metric breakdown; multiple
reports become a side-by-side backend/pack comparison, one column each.
"""

from __future__ import annotations

# (key, human label, kind) — `kind` drives value formatting. `ratio` metrics
# live in 0..1 and render as percentages; everything else is a plain number.
_METRICS: list[tuple[str, str, str]] = [
    ("recall_precision", "Recall precision", "ratio"),
    ("update_correctness", "Update correctness", "ratio"),
    ("semantic_recall", "Semantic recall", "ratio"),
    ("contradiction_rate", "Contradiction rate", "ratio"),
    ("token_cost", "Token cost (avg tokens/query)", "number"),
    ("p95_latency_ms", "p95 latency (ms)", "number"),
]

_METRIC_NOTES: dict[str, str] = {
    "recall_precision": "fraction of expect_contains queries whose recall matched",
    "update_correctness": "fraction of post-update queries reflecting the new fact",
    "semantic_recall": "judge score for semantic recall (n/a when judge is off)",
    "contradiction_rate": "judge rate of contradictions in recall (n/a when judge is off)",
    "token_cost": "mean recalled-context tokens per query",
    "p95_latency_ms": "95th-percentile per-step latency in milliseconds",
}


def _fmt_value(kind: str, value: object) -> str:
    if value is None:
        return "n/a"
    if kind == "ratio":
        return f"{float(value) * 100:.0f}%"
    return f"{float(value):.2f}"


def _count_queries(report: dict) -> int:
    return sum(1 for t in report.get("traces", []) if t.get("step_type") == "query")


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
    scores = report.get("scores", {})
    lines = [
        f"# MemBench scorecard: {report.get('pack_name', '?')}",
        "",
        f"- **Backend:** {report.get('backend_name', '?')}",
        f"- **Started:** {report.get('started_at', '?')}",
        f"- **Finished:** {report.get('finished_at', '?')}",
        f"- **Steps:** {len(report.get('traces', []))} "
        f"({_count_queries(report)} queries)",
        "",
        "## Scores",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for key, label, kind in _METRICS:
        lines.append(f"| {label} | {_fmt_value(kind, scores.get(key))} |")

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
        cells = [_fmt_value(kind, r.get("scores", {}).get(key)) for r in reports]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += ["", "## Runs", ""]
    for r in reports:
        lines.append(
            f"- **{r.get('backend_name', '?')}** on `{r.get('pack_name', '?')}` — "
            f"{len(r.get('traces', []))} steps ({_count_queries(r)} queries)"
        )
    lines.append("")
    return "\n".join(lines)
