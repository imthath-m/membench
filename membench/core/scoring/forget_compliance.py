"""Forget compliance: forgotten content stays out of later recalls.

Scored on `category: forget` queries (the schema requires an earlier forget
step in the same episode). A query passes when no recalled item traces back
to a forgotten event id and the context contains none of its
`expect_absent` strings. Higher is better; failed recalls are skipped.
"""

from __future__ import annotations

from membench.core.runner import StepTrace
from membench.core.scoring.staleness import contains_any
from membench.packs.schema import ForgetStep, Pack, QueryStep


def forget_compliance(pack: Pack, traces: list[StepTrace]) -> float | None:
    """None if the pack has no `forget`-category query."""
    forgotten: dict[int, set[str]] = {}  # episode -> forgotten event ids
    results: list[bool] = []
    for step, trace in zip(pack.steps, traces):
        if isinstance(step, ForgetStep):
            forgotten.setdefault(step.episode, set()).update(step.target)
        elif isinstance(step, QueryStep) and step.category == "forget" and trace.error is None:
            ids = forgotten.get(step.episode, set())
            leaked = any(ids.intersection(item["event_ids"]) for item in trace.detail["items"])
            results.append(
                not leaked and not contains_any(step.expect_absent, trace.detail["context"])
            )
    if not results:
        return None
    return sum(results) / len(results)
