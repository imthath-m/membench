"""Deterministic scoring: the four non-judge metrics.

Each metric lives in its own module and takes the pack (for step
intent/assertions) plus the run traces (for what actually happened) — never
an adapter. LLM-judge metrics (semantic recall, contradiction rate) are out
of scope for this phase; they land alongside a `Judge` interface later.
"""

from __future__ import annotations

from typing import Any

from membench.core.runner import StepTrace
from membench.core.scoring.latency import p95_latency_ms
from membench.core.scoring.recall_precision import recall_precision
from membench.core.scoring.token_cost import token_cost
from membench.core.scoring.update_correctness import update_correctness
from membench.packs.schema import Pack

__all__ = [
    "score_run",
    "recall_precision",
    "update_correctness",
    "token_cost",
    "p95_latency_ms",
]


def score_run(pack: Pack, traces: list[StepTrace]) -> dict[str, Any]:
    """Compute all four deterministic metrics for a completed run."""
    return {
        "recall_precision": recall_precision(pack, traces),
        "update_correctness": update_correctness(pack, traces),
        "token_cost": token_cost(traces),
        "p95_latency_ms": p95_latency_ms(traces),
    }
