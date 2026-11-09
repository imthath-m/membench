"""Paired comparison of two runs (A vs B) on the same queries.

Pairs each query-level metric's per-query `outcomes` by step index (overall
and per category), then: paired bootstrap 95% CI of the mean difference
B − A, McNemar's exact test for pass/fail metrics (bootstrap p otherwise),
and Holm's correction across every metric × category row. A row counts as a
real difference only when its Holm-adjusted p is below `ALPHA` and its CI
excludes 0. Runs must share the pack (sha256 incl. subset) and, where both
used one, the judge and reader model + prompt version.
"""

from __future__ import annotations

from typing import Any

from membench.core.report.markdown import _METRICS
from membench.core.scoring import CATEGORY_METRICS, OUTCOMES
from membench.core.stats import ALPHA, MIN_CATEGORY_N, MIN_HEADLINE_N, holm, paired_diff

_LABELS = {key: label for key, label, _ in _METRICS}
_KINDS = {key: kind for key, _, kind in _METRICS}


def _check_comparable(a: dict, b: dict) -> None:
    for name, report in (("A", a), ("B", b)):
        if "outcomes" not in report:
            raise ValueError(
                f"report {name} has no per-query outcomes (made before membench Phase 13); "
                "re-score it first: membench rescore --in <report> --out <new report>"
            )
    ma, mb = a["manifest"], b["manifest"]
    if ma["pack"]["sha256"] != mb["pack"]["sha256"]:
        raise ValueError("runs used different packs (or subsets); a paired comparison needs "
                         "the same queries")
    for part in ("judge", "reader"):
        ja, jb = ma.get(part), mb.get(part)
        if ja and jb and (ja["model"], ja["prompt_version"]) != (jb["model"], jb["prompt_version"]):
            raise ValueError(f"runs used different {part} models or prompt versions; "
                             "their scores are not comparable")


def compare_reports(a: dict, b: dict) -> dict[str, Any]:
    """Paired comparison rows (B − A) for every metric × category both runs scored."""
    _check_comparable(a, b)
    seed = a["manifest"].get("seed", 0)
    category_of = {
        str(t["step_index"]): t["detail"].get("category") or "uncategorized"
        for t in a["traces"] if t["step_type"] == "query"
    }
    scopes = [("all", list(OUTCOMES))] + [
        (c, list(CATEGORY_METRICS)) for c in sorted(set(category_of.values()))
    ]
    rows: list[dict[str, Any]] = []
    for scope, metrics in scopes:
        for name in metrics:
            oa, ob = a["outcomes"].get(name, {}), b["outcomes"].get(name, {})
            keys = sorted(
                (k for k in oa.keys() & ob.keys() if scope == "all" or category_of[k] == scope),
                key=int,
            )
            if not keys:
                continue
            row = paired_diff([oa[k] for k in keys], [ob[k] for k in keys],
                              f"{seed}:compare:{scope}:{name}")
            min_n = MIN_HEADLINE_N if scope == "all" else MIN_CATEGORY_N
            rows.append({"metric": name, "category": scope, **row, "indicative": row["n"] < min_n})
    for row, p in zip(rows, holm([r["p"] for r in rows])):
        row["p_holm"] = p
        row["significant"] = p < ALPHA and not row["ci_includes_zero"]
    return {
        "pack": a["manifest"]["pack"]["name"],
        "a": a["backend_name"],
        "b": b["backend_name"],
        "alpha": ALPHA,
        "rows": rows,
    }


def _fmt(kind: str, value: float, signed: bool = False) -> str:
    sign = "+" if signed else ""
    if kind == "ratio":
        return f"{value * 100:{sign}.1f}{'pp' if signed else '%'}"
    return f"{value:{sign}.2f}"


def _fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def render_compare_markdown(comparison: dict[str, Any]) -> str:
    a, b = comparison["a"], comparison["b"]
    lines = [
        f"# MemBench paired comparison: {comparison['pack']}",
        "",
        f"- **A:** {a}",
        f"- **B:** {b}",
        (
            "- **Difference:** B − A over the queries both runs scored; 95% CI from a paired "
            "bootstrap (10k resamples, seeded)."
        ),
        (
            "- **p:** McNemar exact test for pass/fail metrics, bootstrap otherwise; "
            f"Holm-adjusted across all {len(comparison['rows'])} rows. A difference is real "
            f"when Holm p < {comparison['alpha']} and the CI excludes 0."
        ),
        (
            f"- **Indicative:** fewer than {MIN_HEADLINE_N} paired queries overall or "
            f"{MIN_CATEGORY_N} per category."
        ),
        "",
        f"| Metric | Category | n | {a} (A) | {b} (B) | B − A | 95% CI | p | p (Holm) | Verdict |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in comparison["rows"]:
        kind = _KINDS.get(r["metric"], "number")
        verdict = "real difference" if r["significant"] else (
            "CI includes 0" if r["ci_includes_zero"] else "not significant after Holm"
        )
        if r["indicative"]:
            verdict += " (indicative)"
        low, high = r["ci"]
        lines.append(
            f"| {_LABELS.get(r['metric'], r['metric'])} | {r['category']} | {r['n']} | "
            f"{_fmt(kind, r['a'])} | {_fmt(kind, r['b'])} | {_fmt(kind, r['diff'], True)} | "
            f"[{_fmt(kind, low, True)}, {_fmt(kind, high, True)}] | "
            f"{_fmt_p(r['p'])} ({r['test']}) | {_fmt_p(r['p_holm'])} | {verdict} |"
        )
    lines.append("")
    return "\n".join(lines)
