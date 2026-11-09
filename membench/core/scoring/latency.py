"""Latency percentiles per stage: `write`, `ingest` (flush) and `recall`.

Each stage pools its successful steps only. p95 is None (`n/a`) below
`MIN_P95_SAMPLES` samples: with fewer, nearest-rank p95 is just the max.
"""

from __future__ import annotations

import math

from membench.core.runner import StepTrace

MIN_P95_SAMPLES = 100


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile (q in 0..1); None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, min(math.ceil(q * len(ordered)) - 1, len(ordered) - 1))]


def latency_scores(traces: list[StepTrace]) -> dict[str, float | None]:
    ok = [t for t in traces if t.error is None]
    stages = {
        "write": [t.latency_ms for t in ok if t.step_type == "write"],
        "ingest": [
            t.detail["ingest_latency_ms"] for t in ok
            if t.step_type == "query" and t.detail.get("ingest_latency_ms") is not None
        ],
        "recall": [t.latency_ms for t in ok if t.step_type == "query"],
    }
    scores: dict[str, float | None] = {}
    for stage, values in stages.items():
        scores[f"{stage}_p50_ms"] = percentile(values, 0.5)
        scores[f"{stage}_p95_ms"] = (
            percentile(values, 0.95) if len(values) >= MIN_P95_SAMPLES else None
        )
    return scores
