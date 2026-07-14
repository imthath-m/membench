"""Semantic recall: judge-scored whether recall conveyed the expected fact.

Unlike `recall_precision` (exact substring match), this asks an LLM judge
whether the recalled context semantically supports the query step's
`judge.semantic_recall` rubric. Score is the fraction of judged queries the
judge passes. Returns None when the judge is unavailable or no query step
declares a semantic-recall rubric.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from membench.core.runner import StepTrace
from membench.packs.schema import Pack, QueryStep

if TYPE_CHECKING:
    from membench.core.scoring.judge import Judge


def semantic_recall(
    pack: Pack, traces: list[StepTrace], judge: "Judge | None"
) -> float | None:
    if judge is None:
        return None

    results: list[bool] = []
    for step, trace in zip(pack.steps, traces):
        if isinstance(step, QueryStep) and step.judge and step.judge.semantic_recall:
            verdict = judge.score(step.judge.semantic_recall, trace.detail["context"])
            results.append(verdict.passed)

    if not results:
        return None
    return sum(results) / len(results)
