"""Error rate: fraction of steps whose adapter call failed or timed out."""

from __future__ import annotations

from membench.core.runner import StepTrace


def error_rate(traces: list[StepTrace]) -> float | None:
    """Errored steps / all steps. None if there were no steps."""
    if not traces:
        return None
    return sum(t.error is not None for t in traces) / len(traces)
