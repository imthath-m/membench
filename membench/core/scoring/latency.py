"""p95 latency across all successful step traces (write + query)."""

from __future__ import annotations

import math

from membench.core.runner import StepTrace


def p95_latency_ms(traces: list[StepTrace]) -> float | None:
    """95th-percentile step latency, nearest-rank method. None if no successful traces."""
    latencies = sorted(t.latency_ms for t in traces if t.error is None)
    if not latencies:
        return None
    index = math.ceil(0.95 * len(latencies)) - 1
    index = max(0, min(index, len(latencies) - 1))
    return latencies[index]
