"""Render RunResult JSON report(s) as Markdown.

Reports derive strictly from the JSON `scores` / `stats` blocks and run
metadata — no rescoring happens here (the runner is the source of truth). A
single report becomes a scorecard table (value, 95% CI, n, and mean ± SD when
the run was repeated), a per-category table and a one-line-per-metric
breakdown; multiple reports become a side-by-side backend/pack comparison,
one column each, plus recall precision by category. A score whose n is below
the minimum (see `core.stats`) is labelled `indicative`; p95 latency below
100 samples is suppressed. For a paired significance test use `membench
compare`.
"""

from __future__ import annotations

from membench.core.runner import MAX_ERROR_RATE
from membench.core.scoring.latency import MIN_P95_SAMPLES
from membench.core.stats import MIN_CATEGORY_N, MIN_HEADLINE_N

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


def _value(report: dict, key: str, kind: str) -> str:
    """A score value; token cost from the whitespace fallback is marked `approx`."""
    text = _fmt_value(kind, report.get("scores", {}).get(key))
    tokenizer = str(report.get("manifest", {}).get("tokenizer", ""))
    if key == "token_cost" and text != "n/a" and "approx" in tokenizer:
        text += " (approx)"
    return text


def _stat(report: dict, key: str, category: str | None = None) -> dict | None:
    stats = report.get("stats") or {}
    if category is None:
        return stats.get("scores", {}).get(key)
    return stats.get("by_category", {}).get(category, {}).get(key)


def _min_n(key: str, category: str | None = None) -> int:
    if category is not None:
        return MIN_CATEGORY_N
    return MIN_P95_SAMPLES if key.endswith("_p95_ms") else MIN_HEADLINE_N


def _fmt_ci(kind: str, stat: dict | None) -> str:
    if not stat or stat.get("ci") is None:
        return ""
    low, high = stat["ci"]
    if kind == "ratio":
        return f"[{low * 100:.0f}–{high * 100:.0f}%]"
    return f"[{low:.2f}–{high:.2f}]"


def _note(stat: dict | None, min_n: int, value: object) -> str:
    """`indicative` / `suppressed` label for a score below its minimum n."""
    if not stat or not stat.get("indicative") or stat.get("n", 0) == 0:
        return ""
    return f"{'suppressed' if value is None else 'indicative'} (n < {min_n})"


def _fmt_score(report: dict, key: str, kind: str) -> str:
    """Compact cell: value, 95% CI, n and an indicative label (when stats exist)."""
    return _cell(_value(report, key, kind), kind, _stat(report, key), _min_n(key),
                 report.get("scores", {}).get(key))


def _cell(text: str, kind: str, stat: dict | None, min_n: int, value: object) -> str:
    if stat is None or value is None:
        return text
    parts = [text, _fmt_ci(kind, stat), f"n={stat['n']}"]
    note = _note(stat, min_n, value)
    if note:
        parts.append("(indicative)")
    return " ".join(p for p in parts if p)


def _fmt_repeats(report: dict, key: str, kind: str) -> str:
    entry = (report.get("repeat_summary") or {}).get("scores", {}).get(key) or {}
    mean, sd = entry.get("mean"), entry.get("sd")
    if mean is None or sd is None:
        return "n/a"
    if kind == "ratio":
        return f"{mean * 100:.0f}% ± {sd * 100:.1f}pp"
    return f"{mean:.2f} ± {sd:.2f}"


def _categories(reports: list[dict]) -> list[str]:
    return sorted({c for r in reports for c in r.get("by_category", {})})


def _category_cell(report: dict, category: str, key: str, kind: str) -> str:
    value = report.get("by_category", {}).get(category, {}).get(key)
    if key == "n":
        return _fmt_value(kind, value)
    return _cell(_fmt_value(kind, value), kind, _stat(report, key, category),
                 MIN_CATEGORY_N, value)


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
    ]
    repeats = (report.get("repeat_summary") or {}).get("n")
    if repeats:
        lines.append(f"- **Repeats:** {repeats} (value / CI / n are from repeat 0)")
    lines += ["", "## Scores", ""]
    header = ["Metric", "Value", "95% CI", "n", "Note"]
    if repeats:
        header.append(f"Mean ± SD ({repeats} repeats)")
    lines += ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    for key, label, kind in _METRICS:
        lines.append("| " + " | ".join(_score_row(report, key, label, kind, repeats)) + " |")

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
    lines += ["", _STATS_NOTE, ""]
    return "\n".join(lines)


_STATS_NOTE = (
    "95% CIs: Wilson for pass/fail rates, seeded bootstrap over queries (10k resamples) "
    f"otherwise. Indicative: n below {MIN_HEADLINE_N} overall, {MIN_CATEGORY_N} per "
    f"category, or {MIN_P95_SAMPLES} samples for p95 latency (suppressed)."
)


def _score_row(report: dict, key: str, label: str, kind: str, repeats: int | None) -> list[str]:
    """Metric | Value | 95% CI | n | Note [| Mean ± SD] cells of the scorecard."""
    stat = _stat(report, key)
    value = report.get("scores", {}).get(key)
    cells = [
        label,
        _value(report, key, kind),
        _fmt_ci(kind, stat) if value is not None else "",
        str(stat["n"]) if stat else "",
        _note(stat, _min_n(key), value),
    ]
    if repeats:
        cells.append(_fmt_repeats(report, key, kind))
    return cells


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
