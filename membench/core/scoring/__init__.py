"""Scoring: deterministic metrics, LLM-judge metrics, latency and error rate.

Each metric lives in its own module and takes the pack (for step
intent/assertions) plus the run traces (for what actually happened) — never
an adapter. The judge metrics (semantic recall, contradiction rate, QA
accuracy) also take an optional `Judge`; when it's None (judge off or no API
key) they score as None rather than crashing. `score_all` adds the
per-category breakdown.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from membench.core.runner import StepTrace
from membench.core.scoring.abstention import abstention
from membench.core.scoring.contradiction_rate import contradiction_rate
from membench.core.scoring.error_rate import error_rate
from membench.core.scoring.evidence_recall import evidence_recall_at_k
from membench.core.scoring.forget_compliance import forget_compliance
from membench.core.scoring.judge import JudgeError, Verdict
from membench.core.scoring.latency import latency_scores
from membench.core.scoring.qa_accuracy import qa_accuracy
from membench.core.scoring.recall_precision import recall_precision
from membench.core.scoring.semantic_recall import semantic_recall
from membench.core.scoring.staleness import staleness
from membench.core.scoring.temporal_accuracy import temporal_accuracy
from membench.core.scoring.token_cost import token_cost
from membench.core.scoring.update_correctness import update_correctness
from membench.packs.schema import Pack, QueryStep

if TYPE_CHECKING:
    from membench.core.scoring.judge import Judge

__all__ = [
    "abstention",
    "contradiction_rate",
    "error_rate",
    "evidence_recall_at_k",
    "forget_compliance",
    "latency_scores",
    "qa_accuracy",
    "recall_precision",
    "score_all",
    "score_run",
    "semantic_recall",
    "staleness",
    "temporal_accuracy",
    "token_cost",
    "update_correctness",
]


def score_run(
    pack: Pack, traces: list[StepTrace], judge: Judge | None = None
) -> dict[str, Any]:
    """Compute all metrics for a completed run.

    Judge metrics are None when `judge` is None (off / no API key).
    """
    return {
        "recall_precision": recall_precision(pack, traces),
        "update_correctness": update_correctness(pack, traces),
        "staleness": staleness(pack, traces),
        "temporal_accuracy": temporal_accuracy(pack, traces),
        "forget_compliance": forget_compliance(pack, traces),
        "evidence_recall_at_k": evidence_recall_at_k(pack, traces),
        "abstention": abstention(pack, traces),
        "semantic_recall": semantic_recall(pack, traces, judge),
        "contradiction_rate": contradiction_rate(pack, traces, judge),
        "qa_accuracy": qa_accuracy(pack, traces, judge),
        "token_cost": token_cost(traces),
        **latency_scores(traces),
        "error_rate": error_rate(traces),
    }


class _MemoJudge:
    """Asks the wrapped judge once per (rubric, evidence) within one scoring pass."""

    def __init__(self, judge: Judge) -> None:
        self.model = judge.model
        self._judge = judge
        self._seen: dict[tuple[str, str], Verdict | JudgeError] = {}

    def score(self, rubric: str, evidence: str) -> Verdict:
        key = (rubric, evidence)
        if key not in self._seen:
            try:
                self._seen[key] = self._judge.score(rubric, evidence)
            except JudgeError as exc:
                self._seen[key] = exc
        outcome = self._seen[key]
        if isinstance(outcome, JudgeError):
            raise outcome
        return outcome


def score_all(
    pack: Pack, traces: list[StepTrace], judge: Judge | None = None
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """`score_run` plus a breakdown by query `category` ("uncategorized" if unset).

    Each category scores the query-level metrics over its own queries (all
    non-query steps kept, so update/forget context is intact).
    """
    judge = _MemoJudge(judge) if judge is not None else None
    scores = score_run(pack, traces, judge)
    categories = sorted({
        s.category or "uncategorized" for s in pack.steps if isinstance(s, QueryStep)
    })
    by_category: dict[str, dict[str, Any]] = {}
    for category in categories:
        keep = [
            i for i, s in enumerate(pack.steps)
            if not isinstance(s, QueryStep) or (s.category or "uncategorized") == category
        ]
        sub_pack = Pack(pack.name, pack.description, [pack.steps[i] for i in keep], pack.sha256)
        sub_traces = [traces[i] for i in keep]
        by_category[category] = {
            "n": sum(isinstance(s, QueryStep) for s in sub_pack.steps),
            "recall_precision": recall_precision(sub_pack, sub_traces),
            "evidence_recall_at_k": evidence_recall_at_k(sub_pack, sub_traces),
            "semantic_recall": semantic_recall(sub_pack, sub_traces, judge),
            "qa_accuracy": qa_accuracy(sub_pack, sub_traces, judge),
        }
    return scores, by_category
