"""Replay a scenario pack against an adapter, capturing per-step traces.

The runner owns timing and token accounting — adapters only move data.
Scoring is out of scope for Phase 1; `RunResult.scores` is a placeholder
populated in Phase 2.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from membench.core.adapter import Event, MemoryAdapter
from membench.packs.schema import Pack, QueryStep, WriteStep


@dataclass
class StepTrace:
    step_index: int
    step_type: str  # "write" | "query"
    session_id: str
    latency_ms: float
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    pack_name: str
    backend_name: str
    started_at: str
    finished_at: str
    traces: list[StepTrace] = field(default_factory=list)
    scores: dict[str, Any] = field(default_factory=dict)


def run_pack(pack: Pack, adapter: MemoryAdapter, backend_name: str) -> RunResult:
    """Replay every step in `pack` against `adapter` in order."""
    started_at = datetime.now(timezone.utc).isoformat()
    traces: list[StepTrace] = []

    for index, step in enumerate(pack.steps):
        if isinstance(step, WriteStep):
            event = Event(
                session_id=step.session_id,
                key=step.key,
                role=step.role,
                content=step.content,
                timestamp=step.timestamp,
                metadata=step.metadata,
            )
            start = time.perf_counter()
            adapter.write(event)
            latency_ms = (time.perf_counter() - start) * 1000
            traces.append(
                StepTrace(
                    step_index=index,
                    step_type="write",
                    session_id=step.session_id,
                    latency_ms=latency_ms,
                    detail={"key": step.key, "fact_type": step.fact_type},
                )
            )
        elif isinstance(step, QueryStep):
            start = time.perf_counter()
            result = adapter.recall(step.query)
            latency_ms = (time.perf_counter() - start) * 1000
            traces.append(
                StepTrace(
                    step_index=index,
                    step_type="query",
                    session_id=step.session_id,
                    latency_ms=latency_ms,
                    detail={
                        "query": step.query,
                        "context": result.context,
                        "token_count": result.token_count,
                        "item_count": len(result.items),
                        "expect_contains": step.expect_contains,
                        "expect_key": step.expect_key,
                    },
                )
            )
        else:  # pragma: no cover - schema guarantees this can't happen
            raise TypeError(f"unknown step type: {step!r}")

    finished_at = datetime.now(timezone.utc).isoformat()
    return RunResult(
        pack_name=pack.name,
        backend_name=backend_name,
        started_at=started_at,
        finished_at=finished_at,
        traces=traces,
        scores={},
    )
