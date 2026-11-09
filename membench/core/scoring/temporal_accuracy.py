"""Temporal reasoning: accuracy on `category: temporal` queries.

A temporal query is asked at its step `timestamp` (passed to `recall` as
`at`) about when something happened or what held at an earlier time. It
passes when the recalled context contains all its `expect_contains` strings
(case-insensitive), e.g. the value that held then or the date it changed.
With a reader, `by_category.temporal.qa_accuracy` judges the answer instead.
"""

from __future__ import annotations

from membench.core.runner import StepTrace
from membench.core.scoring.recall_precision import _passes
from membench.packs.schema import Pack, QueryStep


def temporal_accuracy(pack: Pack, traces: list[StepTrace]) -> float | None:
    """None if no temporal query declares `expect_contains`."""
    results = [
        _passes(step.expect_contains, trace.detail["context"])
        for step, trace in zip(pack.steps, traces)
        if isinstance(step, QueryStep) and step.category == "temporal" and step.expect_contains
    ]
    if not results:
        return None
    return sum(results) / len(results)
