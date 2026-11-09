"""Staleness: how often recall still surfaces a superseded value.

Over the post-update queries `update_correctness` scores (a query whose
`expect_key` was updated or contradicted earlier in its episode) that also
declare `expect_absent`: the fraction whose context still contains any of
those stale strings (case-insensitive). Lower is better. Deterministic; a
backend that keeps everything (full-context) passes `update_correctness`
but scores badly here.
"""

from __future__ import annotations

from membench.core.runner import StepTrace
from membench.packs.schema import Pack, QueryStep, WriteStep

_UPDATE_FACT_TYPES = {"update", "contradict"}


def contains_any(needles: list[str], context: str) -> bool:
    context_lower = context.lower()
    return any(needle.lower() in context_lower for needle in needles)


def staleness(pack: Pack, traces: list[StepTrace]) -> float | None:
    """None if no post-update query declares `expect_absent`."""
    updated_keys: set[tuple[int, str]] = set()  # (episode, fact_key)
    results: list[bool] = []
    for step, trace in zip(pack.steps, traces):
        if isinstance(step, WriteStep) and step.fact_type in _UPDATE_FACT_TYPES:
            updated_keys.add((step.episode, step.fact_key))
        elif (
            isinstance(step, QueryStep)
            and (step.episode, step.expect_key) in updated_keys
            and step.expect_absent
            and trace.error is None  # nothing recalled, nothing stale
        ):
            results.append(contains_any(step.expect_absent, trace.detail["context"]))
    if not results:
        return None
    return sum(results) / len(results)
