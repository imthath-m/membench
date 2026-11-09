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
from membench.core.scoring.judge import JudgeError, record_judge_error
from membench.core.stats import mean_of
from membench.packs.schema import Pack, QueryStep

if TYPE_CHECKING:
    from membench.core.scoring.judge import Judge


def semantic_recall(
    pack: Pack, traces: list[StepTrace], judge: Judge | None
) -> float | None:
    return mean_of(semantic_recall_outcomes(pack, traces, judge))


def semantic_recall_outcomes(
    pack: Pack, traces: list[StepTrace], judge: Judge | None
) -> dict[int, bool]:
    if judge is None:
        return {}

    results: dict[int, bool] = {}
    for step, trace in zip(pack.steps, traces):
        if isinstance(step, QueryStep) and step.judge and step.judge.semantic_recall:
            if trace.error is not None:  # a failed recall conveyed nothing
                results[trace.step_index] = False
                continue
            try:
                verdict = judge.score(step.judge.semantic_recall, trace.detail["context"])
            except JudgeError as exc:
                record_judge_error(trace.detail, "semantic_recall", exc)
                continue
            results[trace.step_index] = verdict.passed
    return results
