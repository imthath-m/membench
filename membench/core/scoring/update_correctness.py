"""Update correctness: did recall reflect the updated fact, not the stale one.

A write step with `fact_type` of "update" or "contradict" marks that it
supersedes a previously written fact under the same `key`. Any later query
whose `expect_key` matches a key that has been updated is scored: does the
recalled context contain the (updated) `expect_contains` strings?
"""

from __future__ import annotations

from membench.core.runner import StepTrace
from membench.packs.schema import Pack, QueryStep, WriteStep

_UPDATE_FACT_TYPES = {"update", "contradict"}


def _passes(expect_contains: list[str], context: str) -> bool:
    context_lower = context.lower()
    return all(needle.lower() in context_lower for needle in expect_contains)


def update_correctness(pack: Pack, traces: list[StepTrace]) -> float | None:
    """Fraction of post-update queries whose recall reflects the new fact.

    Returns None if the pack never queries a key after updating it (metric
    not applicable).
    """
    updated_keys: set[str] = set()
    results: list[bool] = []

    for step, trace in zip(pack.steps, traces):
        if isinstance(step, WriteStep):
            if step.fact_type in _UPDATE_FACT_TYPES:
                updated_keys.add(step.key)
        elif isinstance(step, QueryStep):
            if step.expect_key in updated_keys and step.expect_contains:
                results.append(_passes(step.expect_contains, trace.detail["context"]))

    if not results:
        return None
    return sum(results) / len(results)
