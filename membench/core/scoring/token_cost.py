"""Token cost: average tokens injected into context per query recall.

Reads `token_count` off each query step's trace (the runner counts each
recalled `context` with its configured tokenizer) rather than recomputing anything.
"""

from __future__ import annotations

from membench.core.runner import StepTrace
from membench.core.stats import mean_of


def token_cost(traces: list[StepTrace]) -> float | None:
    """Mean `token_count` across successful query traces; None if there were none."""
    return mean_of(token_cost_outcomes(traces))


def token_cost_outcomes(traces: list[StepTrace]) -> dict[int, int]:
    return {
        t.step_index: t.detail["token_count"]
        for t in traces
        if t.step_type == "query" and t.error is None
    }
