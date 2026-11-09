"""Per-query failure listing for one run report (`membench report --failures`).

Each failed query is sorted into the classes below, from the report's stored
per-query `outcomes` and the pack (evidence ids, stale / forgotten values,
gold answers); a query can land in more than one class. Each entry shows the
query, the expected evidence ids, the recalled item ids and a short context
excerpt; `excerpt_chars=0` lists ids only (no query, answer or context text,
e.g. for LoCoMo).
"""

from __future__ import annotations

from membench.core.scoring.abstention import is_abstention
from membench.packs.schema import Pack, QueryStep

CLASSES: list[tuple[str, str]] = [
    ("not_retrieved", "Not retrieved (expected content or evidence missing from recall)"),
    ("stale_surfaced", "Stale surfaced (a superseded value is still recalled)"),
    ("answered_wrong", "Retrieved but answered wrong (reader answer judged wrong)"),
    ("false_abstention", "False abstention (reader declined an answerable query)"),
    ("forget_violation", "Forget violation (forgotten content recalled)"),
]


def classify(report: dict, pack: Pack) -> dict[str, list[int]]:
    """{class: [step indices]} of the report's failed queries (repeat 0)."""
    if "outcomes" not in report:
        raise ValueError("report has no per-query outcomes (re-score it first: "
                         "membench rescore --in old.json --out new.json)")
    out = {name: [] for name, _ in CLASSES}

    def outcome(metric: str, index: int) -> object:
        return report["outcomes"].get(metric, {}).get(str(index))

    for trace in report["traces"]:
        index, step = trace["step_index"], pack.steps[trace["step_index"]]
        if not isinstance(step, QueryStep) or trace["error"] is not None:
            continue
        missed = outcome("recall_precision", index) is False or (
            outcome("recall_precision", index) is None
            and outcome("evidence_recall_at_k", index) == 0
        )
        answer = trace["detail"].get("reader_answer")
        for name, failed in (
            ("not_retrieved", missed),
            ("stale_surfaced", outcome("staleness", index) is True),
            ("answered_wrong", not missed and outcome("qa_accuracy", index) is False),
            ("false_abstention", step.answer is not None and not step.abstain
             and answer is not None and is_abstention(answer)),
            ("forget_violation", outcome("forget_compliance", index) is False),
        ):
            if failed:
                out[name].append(index)
    return out


def _excerpt(context: str, needles: list[str], chars: int) -> str:
    """`chars` characters of context, centred on the first needle found (else its start)."""
    if chars <= 0:
        return ""
    lower = context.lower()
    hits = [lower.find(n.lower()) for n in needles if n and n.lower() in lower]
    start = max(0, min(hits) - chars // 2) if hits else 0
    text = context[start:start + chars].replace("\n", " ⏎ ").replace("`", "'")
    return ("…" if start else "") + text + ("…" if start + chars < len(context) else "")


def render_failures(report: dict, pack: Pack, excerpt_chars: int = 200) -> str:
    classes = classify(report, pack)
    traces = {t["step_index"]: t for t in report["traces"]}
    errors = sum(1 for t in report["traces"] if t["step_type"] == "query" and t["error"])
    lines = [
        f"# Failures: {report['backend_name']} on {report['pack_name']}",
        "",
        "| Class | Queries |",
        "| --- | --- |",
        *(f"| {label} | {len(classes[name])} |" for name, label in CLASSES),
        "",
        f"Queries that errored (not classified): {errors}.",
        "",
    ]
    for name, label in CLASSES:
        if not classes[name]:
            continue
        lines += [f"## {label}", ""]
        for index in classes[name]:
            step, detail = pack.steps[index], traces[index]["detail"]
            recalled = sorted({i for item in detail["items"] for i in item["event_ids"]})
            needles = step.expect_absent if name in ("stale_surfaced", "forget_violation") \
                else step.expect_contains
            query = f": {step.query}" if excerpt_chars > 0 else ""
            lines += [
                f"- **step {index}** ({step.category or 'uncategorized'}){query}",
                (f"  - evidence ids: {', '.join(step.evidence) or 'none'}; "
                 f"recalled ids: {', '.join(recalled) or 'none'}"),
            ]
            if name in ("answered_wrong", "false_abstention") and excerpt_chars > 0:
                lines.append(f"  - reader: {detail.get('reader_answer')!r}; "
                             f"gold: {step.answer!r}")
            excerpt = _excerpt(detail["context"], list(needles), excerpt_chars)
            if excerpt:
                lines.append(f"  - context: `{excerpt}`")
        lines.append("")
    return "\n".join(lines)
