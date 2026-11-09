"""Evidence recall@k: did recall return the writes that answer the query.

For each query declaring `evidence` (event ids), the fraction of those ids
found in the `event_ids` of the recalled items — k being however many items
the backend returned (its own top-k). Averaged over queries. Deterministic;
a failed recall scores 0.
"""

from __future__ import annotations

from membench.core.runner import StepTrace
from membench.packs.schema import Pack, QueryStep


def evidence_recall_at_k(pack: Pack, traces: list[StepTrace]) -> float | None:
    """None if no query declares `evidence`."""
    results: list[float] = []
    for step, trace in zip(pack.steps, traces):
        if isinstance(step, QueryStep) and step.evidence:
            recalled = {i for item in trace.detail["items"] for i in item["event_ids"]}
            results.append(len(recalled.intersection(step.evidence)) / len(set(step.evidence)))
    if not results:
        return None
    return sum(results) / len(results)
