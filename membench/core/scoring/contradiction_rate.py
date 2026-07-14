"""Contradiction rate: judge-scored how often recall surfaces contradictions.

For each query step declaring a `judge.contradiction` rubric, the judge is
asked whether the recalled context contains contradictory statements about
the same fact (e.g. a stale value alongside its update). The rate is the
fraction of judged queries in which the judge finds a contradiction — lower
is better. Returns None when the judge is unavailable or no query step
declares a contradiction rubric.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from membench.core.runner import StepTrace
from membench.packs.schema import Pack, QueryStep

if TYPE_CHECKING:
    from membench.core.scoring.judge import Judge


def contradiction_rate(
    pack: Pack, traces: list[StepTrace], judge: "Judge | None"
) -> float | None:
    if judge is None:
        return None

    results: list[bool] = []
    for step, trace in zip(pack.steps, traces):
        if isinstance(step, QueryStep) and step.judge and step.judge.contradiction:
            verdict = judge.score(step.judge.contradiction, trace.detail["context"])
            results.append(verdict.passed)  # passed == contradiction present

    if not results:
        return None
    return sum(results) / len(results)
