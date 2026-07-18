"""Render RunResult JSON report(s) as a self-contained HTML document.

Like the Markdown emitter, this derives strictly from the JSON `scores`
block and run metadata — no rescoring. The output is a single, fully
self-contained page: inline `<style>` only, no external CDN / script / font
links, so it opens offline anywhere. Reuses the metric spec and formatting
from `markdown.py` to keep the two emitters in lockstep.
"""

from __future__ import annotations

from html import escape

from membench.core.report.markdown import (
    _METRIC_NOTES,
    _METRICS,
    _count_queries,
    _fmt_value,
    _label,
)

_STYLE = """
  body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         margin: 2rem auto; max-width: 52rem; color: #1a1a1a; line-height: 1.5; }
  h1 { font-size: 1.5rem; margin-bottom: 0.25rem; }
  h2 { font-size: 1.1rem; margin-top: 1.75rem; }
  table { border-collapse: collapse; width: 100%; margin-top: 0.5rem; }
  th, td { border: 1px solid #ddd; padding: 0.4rem 0.7rem; text-align: left; }
  th { background: #f5f5f5; }
  td.value, th.value { text-align: right; font-variant-numeric: tabular-nums; }
  .meta { color: #555; font-size: 0.92rem; }
  .meta li { margin: 0.1rem 0; }
  code { background: #f2f2f2; padding: 0 0.25rem; border-radius: 3px; }
""".strip("\n")


def render_html(reports: list[dict]) -> str:
    """Render one or more RunResult-shaped dicts as a self-contained HTML page."""
    if not reports:
        body = "<p><em>No reports provided.</em></p>"
        title = "MemBench report"
    elif len(reports) == 1:
        title, body = _render_single(reports[0])
    else:
        title, body = _render_comparison(reports)
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{escape(title)}</title>\n"
        f"<style>\n{_STYLE}\n</style>\n"
        "</head>\n<body>\n"
        f"{body}\n"
        "</body>\n</html>\n"
    )


def _row(label: str, cells: list[str]) -> str:
    tds = "".join(f'<td class="value">{escape(c)}</td>' for c in cells)
    return f"<tr><th>{escape(label)}</th>{tds}</tr>"


def _render_single(report: dict) -> tuple[str, str]:
    scores = report.get("scores", {})
    pack = report.get("pack_name", "?")
    title = f"MemBench scorecard: {pack}"

    meta = (
        "<ul class=\"meta\">"
        f"<li>Backend: <code>{escape(str(report.get('backend_name', '?')))}</code></li>"
        f"<li>Started: {escape(str(report.get('started_at', '?')))}</li>"
        f"<li>Finished: {escape(str(report.get('finished_at', '?')))}</li>"
        f"<li>Steps: {len(report.get('traces', []))} "
        f"({_count_queries(report)} queries)</li>"
        "</ul>"
    )

    rows = "".join(
        _row(label, [_fmt_value(kind, scores.get(key))])
        for key, label, kind in _METRICS
    )
    table = (
        "<table><thead><tr><th>Metric</th>"
        "<th class=\"value\">Value</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )

    breakdown = "".join(
        f"<li><strong>{escape(label)}:</strong> {escape(_METRIC_NOTES[key])}</li>"
        for key, label, _ in _METRICS
    )

    body = (
        f"<h1>{escape(title)}</h1>\n{meta}\n"
        f"<h2>Scores</h2>\n{table}\n"
        f"<h2>Breakdown</h2>\n<ul>{breakdown}</ul>"
    )
    return title, body


def _render_comparison(reports: list[dict]) -> tuple[str, str]:
    packs = {r.get("pack_name") for r in reports}
    if len(packs) == 1:
        title = f"MemBench comparison: {next(iter(packs))}"
    else:
        title = "MemBench comparison"

    labels = [_label(r) for r in reports]
    header = (
        "<tr><th>Metric</th>"
        + "".join(f'<th class="value">{escape(label)}</th>' for label in labels)
        + "</tr>"
    )
    rows = "".join(
        _row(label, [_fmt_value(kind, r.get("scores", {}).get(key)) for r in reports])
        for key, label, kind in _METRICS
    )
    table = f"<table><thead>{header}</thead><tbody>{rows}</tbody></table>"

    runs = "".join(
        f"<li><strong>{escape(str(r.get('backend_name', '?')))}</strong> on "
        f"<code>{escape(str(r.get('pack_name', '?')))}</code> — "
        f"{len(r.get('traces', []))} steps ({_count_queries(r)} queries)</li>"
        for r in reports
    )

    body = (
        f"<h1>{escape(title)}</h1>\n"
        f"<h2>Scores</h2>\n{table}\n"
        f"<h2>Runs</h2>\n<ul class=\"meta\">{runs}</ul>"
    )
    return title, body
