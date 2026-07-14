"""Token cost: average tokens injected into context per query recall.

Reads `token_count` off each query step's trace (populated from
`RecallResult.token_count` by the runner) rather than recomputing anything.
"""

from __future__ import annotations

from membench.core.runner import StepTrace


def token_cost(traces: list[StepTrace]) -> float | None:
    """Mean `token_count` across query traces. None if there were no queries."""
    counts = [t.detail["token_count"] for t in traces if t.step_type == "query"]
    if not counts:
        return None
    return sum(counts) / len(counts)
