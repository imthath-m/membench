"""Recall precision: exact-fact assertions on query steps.

For every query step that declares `expect_contains`, check whether the
recalled context contains all of the expected substrings (case-insensitive).
Precision is the fraction of such queries that pass.
"""

from __future__ import annotations

from membench.core.runner import StepTrace
from membench.packs.schema import Pack, QueryStep


def _passes(expect_contains: list[str], context: str) -> bool:
    context_lower = context.lower()
    return all(needle.lower() in context_lower for needle in expect_contains)


def recall_precision(pack: Pack, traces: list[StepTrace]) -> float | None:
    """Fraction of `expect_contains`-bearing queries whose context matches.

    Returns None if the pack has no such queries (metric not applicable).
    """
    results: list[bool] = []
    for step, trace in zip(pack.steps, traces):
        if isinstance(step, QueryStep) and step.expect_contains:
            results.append(_passes(step.expect_contains, trace.detail["context"]))

    if not results:
        return None
    return sum(results) / len(results)
