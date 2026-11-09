"""Scoring: deterministic metrics, LLM-judge metrics, latency and error rate.

Each metric lives in its own module and takes the pack (for step
intent/assertions) plus the run traces (for what actually happened) — never
an adapter. The judge metrics (semantic recall, contradiction rate, QA
accuracy) also take an optional `Judge`; when it's None (judge off or no API
key) they score as None rather than crashing. `score_all` adds the
per-category breakdown; `score_detailed` also returns per-query outcomes and
their statistics (n, 95% CI, indicative flag).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from membench.core.runner import StepTrace
from membench.core.scoring.abstention import abstention, abstention_outcomes
from membench.core.scoring.contradiction_rate import (
    contradiction_rate,
    contradiction_rate_outcomes,
)
from membench.core.scoring.error_rate import error_rate
from membench.core.scoring.evidence_recall import evidence_recall_at_k, evidence_recall_outcomes
from membench.core.scoring.forget_compliance import (
    forget_compliance,
    forget_compliance_outcomes,
)
from membench.core.scoring.judge import JudgeError, Verdict
from membench.core.scoring.latency import MIN_P95_SAMPLES, latency_scores
from membench.core.scoring.qa_accuracy import qa_accuracy, qa_accuracy_outcomes
from membench.core.scoring.recall_precision import recall_precision, recall_precision_outcomes
from membench.core.scoring.semantic_recall import semantic_recall, semantic_recall_outcomes
from membench.core.scoring.staleness import staleness, staleness_outcomes
from membench.core.scoring.temporal_accuracy import temporal_accuracy, temporal_accuracy_outcomes
from membench.core.scoring.token_cost import token_cost, token_cost_outcomes
from membench.core.scoring.update_correctness import (
    update_correctness,
    update_correctness_outcomes,
)
from membench.core.stats import MIN_CATEGORY_N, MIN_HEADLINE_N, mean_of, stat
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
    "score_detailed",
    "score_run",
    "semantic_recall",
    "staleness",
    "temporal_accuracy",
    "token_cost",
    "update_correctness",
]


# Query-level metrics: name -> per-query outcomes {step_index: value}. Each
# score is the mean of its outcomes; CIs are computed over them.
OUTCOMES: dict[str, Callable[[Pack, list[StepTrace], Any], dict[int, Any]]] = {
    "recall_precision": lambda p, t, j: recall_precision_outcomes(p, t),
    "update_correctness": lambda p, t, j: update_correctness_outcomes(p, t),
    "staleness": lambda p, t, j: staleness_outcomes(p, t),
    "temporal_accuracy": lambda p, t, j: temporal_accuracy_outcomes(p, t),
    "forget_compliance": lambda p, t, j: forget_compliance_outcomes(p, t),
    "evidence_recall_at_k": lambda p, t, j: evidence_recall_outcomes(p, t),
    "abstention": lambda p, t, j: abstention_outcomes(p, t),
    "semantic_recall": semantic_recall_outcomes,
    "contradiction_rate": contradiction_rate_outcomes,
    "qa_accuracy": qa_accuracy_outcomes,
    "token_cost": lambda p, t, j: token_cost_outcomes(t),
}
CATEGORY_METRICS = ("recall_precision", "evidence_recall_at_k", "semantic_recall", "qa_accuracy")


@dataclass
class Scoring:
    """Everything a scoring pass produces (see `score_detailed`)."""

    scores: dict[str, Any]
    by_category: dict[str, dict[str, Any]]
    outcomes: dict[str, dict[str, Any]]  # metric -> {str(step_index): outcome}
    stats: dict[str, Any]  # {"scores": {metric: stat}, "by_category": {cat: {metric: stat}}}


def score_run(
    pack: Pack, traces: list[StepTrace], judge: Judge | None = None
) -> dict[str, Any]:
    """Compute all metrics for a completed run.

    Judge metrics are None when `judge` is None (off / no API key).
    """
    return score_detailed(pack, traces, judge).scores


class _MemoJudge:
    """Asks the wrapped judge once per (rubric, evidence) within one scoring pass."""

    def __init__(self, judge: Judge) -> None:
        self.model = judge.model
        self._judge = judge
        self._seen: dict[tuple[str, ...], Verdict | JudgeError] = {}

    def score(self, rubric: str, evidence: str) -> Verdict:
        return self._memo((rubric, evidence), lambda: self._judge.score(rubric, evidence))

    def ask(self, prompt: str, parse: Callable[[str], Verdict]) -> Verdict:
        return self._memo((prompt,), lambda: self._judge.ask(prompt, parse))

    def _memo(self, key: tuple[str, ...], call: Callable[[], Verdict]) -> Verdict:
        if key not in self._seen:
            try:
                self._seen[key] = call()
            except JudgeError as exc:
                self._seen[key] = exc
        outcome = self._seen[key]
        if isinstance(outcome, JudgeError):
            raise outcome
        return outcome


def score_all(
    pack: Pack, traces: list[StepTrace], judge: Judge | None = None
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """`score_run` plus a breakdown by query `category` ("uncategorized" if unset)."""
    scoring = score_detailed(pack, traces, judge)
    return scoring.scores, scoring.by_category


def _latency_n(traces: list[StepTrace]) -> dict[str, int]:
    ok = [t for t in traces if t.error is None]
    return {
        "write": sum(t.step_type == "write" for t in ok),
        "ingest": sum(
            t.step_type == "query" and t.detail.get("ingest_latency_ms") is not None for t in ok
        ),
        "recall": sum(t.step_type == "query" for t in ok),
    }


def score_detailed(
    pack: Pack, traces: list[StepTrace], judge: Judge | None = None, seed: int = 0
) -> Scoring:
    """Scores, per-category breakdown, per-query outcomes and their statistics.

    Each category scores `CATEGORY_METRICS` over its own queries' outcomes.
    `stats` holds n, a 95% CI (Wilson for pass/fail, seeded bootstrap
    otherwise; see `core.stats`) and an `indicative` flag (n below the
    minimum) for every score; latency entries carry n only.
    """
    judge = _MemoJudge(judge) if judge is not None else None
    outcomes = {name: fn(pack, traces, judge) for name, fn in OUTCOMES.items()}
    scores = {name: mean_of(values) for name, values in outcomes.items()}
    scores.update(latency_scores(traces))
    scores["error_rate"] = error_rate(traces)

    score_stats = {
        name: stat(list(values.values()), f"{seed}:all:{name}", MIN_HEADLINE_N)
        for name, values in outcomes.items()
    }
    for stage, n in _latency_n(traces).items():
        score_stats[f"{stage}_p50_ms"] = {"n": n, "ci": None, "method": None, "indicative": False}
        score_stats[f"{stage}_p95_ms"] = {
            "n": n, "ci": None, "method": None, "indicative": n < MIN_P95_SAMPLES,
        }
    score_stats["error_rate"] = stat(
        [t.error is not None for t in traces], f"{seed}:all:error_rate", MIN_HEADLINE_N
    )

    category_of = {
        t.step_index: s.category or "uncategorized"
        for s, t in zip(pack.steps, traces) if isinstance(s, QueryStep)
    }
    by_category: dict[str, dict[str, Any]] = {}
    category_stats: dict[str, dict[str, Any]] = {}
    for category in sorted(set(category_of.values())):
        by_category[category] = {"n": sum(c == category for c in category_of.values())}
        category_stats[category] = {}
        for name in CATEGORY_METRICS:
            values = [v for i, v in outcomes[name].items() if category_of[i] == category]
            by_category[category][name] = sum(values) / len(values) if values else None
            category_stats[category][name] = stat(
                values, f"{seed}:{category}:{name}", MIN_CATEGORY_N
            )
    return Scoring(
        scores=scores,
        by_category=by_category,
        outcomes={n: {str(i): v for i, v in o.items()} for n, o in outcomes.items()},
        stats={"scores": score_stats, "by_category": category_stats},
    )
