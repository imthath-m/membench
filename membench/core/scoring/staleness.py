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
from membench.core.stats import mean_of
from membench.packs.schema import Pack, QueryStep, WriteStep

_UPDATE_FACT_TYPES = {"update", "contradict"}


def contains_any(needles: list[str], context: str) -> bool:
    context_lower = context.lower()
    return any(needle.lower() in context_lower for needle in needles)


def staleness(pack: Pack, traces: list[StepTrace]) -> float | None:
    """None if no post-update query declares `expect_absent`."""
    return mean_of(staleness_outcomes(pack, traces))


def staleness_outcomes(pack: Pack, traces: list[StepTrace]) -> dict[int, bool]:
    """Stale (True) or not per scored post-update query, keyed by step index."""
    updated_keys: set[tuple[int, str]] = set()  # (episode, fact_key)
    results: dict[int, bool] = {}
    for step, trace in zip(pack.steps, traces):
        if isinstance(step, WriteStep) and step.fact_type in _UPDATE_FACT_TYPES:
            updated_keys.add((step.episode, step.fact_key))
        elif (
            isinstance(step, QueryStep)
            and (step.episode, step.expect_key) in updated_keys
            and step.expect_absent
            and trace.error is None  # nothing recalled, nothing stale
        ):
            results[trace.step_index] = contains_any(step.expect_absent, trace.detail["context"])
    return results
