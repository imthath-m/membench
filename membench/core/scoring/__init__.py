"""Scoring: four deterministic metrics plus two LLM-judge metrics.

Each metric lives in its own module and takes the pack (for step
intent/assertions) plus the run traces (for what actually happened) — never
an adapter. The two judge metrics (semantic recall, contradiction rate) also
take an optional `Judge`; when it's None (judge off or no API key) they score
as None rather than crashing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from membench.core.runner import StepTrace
from membench.core.scoring.contradiction_rate import contradiction_rate
from membench.core.scoring.latency import p95_latency_ms
from membench.core.scoring.recall_precision import recall_precision
from membench.core.scoring.semantic_recall import semantic_recall
from membench.core.scoring.token_cost import token_cost
from membench.core.scoring.update_correctness import update_correctness
from membench.packs.schema import Pack

if TYPE_CHECKING:
    from membench.core.scoring.judge import Judge

__all__ = [
    "score_run",
    "recall_precision",
    "update_correctness",
    "token_cost",
    "p95_latency_ms",
    "semantic_recall",
    "contradiction_rate",
]


def score_run(
    pack: Pack, traces: list[StepTrace], judge: "Judge | None" = None
) -> dict[str, Any]:
    """Compute all metrics for a completed run.

    Judge metrics are None when `judge` is None (off / no API key).
    """
    return {
        "recall_precision": recall_precision(pack, traces),
        "update_correctness": update_correctness(pack, traces),
        "token_cost": token_cost(traces),
        "p95_latency_ms": p95_latency_ms(traces),
        "semantic_recall": semantic_recall(pack, traces, judge),
        "contradiction_rate": contradiction_rate(pack, traces, judge),
    }
