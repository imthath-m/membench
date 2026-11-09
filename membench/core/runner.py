"""Replay a scenario pack against an adapter, capturing per-step traces.

The runner owns timing, token accounting, retries, timeouts, and scoring —
adapters only move data. A step that still fails after retries (or times
out) is recorded in its trace's `error` and the run continues.
`RunResult.scores` is populated from `core.scoring` once every step has been
replayed; a run whose `error_rate` exceeds `MAX_ERROR_RATE` is marked invalid.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from membench.core.adapter import Event, MemoryAdapter, RecallResult
from membench.core.retry import retry_call
from membench.packs.schema import Pack, QueryStep, WriteStep

logger = logging.getLogger("membench")

DEFAULT_STEP_TIMEOUT_S = 120.0
MAX_ERROR_RATE = 0.1


@dataclass
class StepTrace:
    step_index: int
    step_type: str  # "write" | "query"
    session_id: str
    latency_ms: float
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class RunResult:
    pack_name: str
    backend_name: str
    started_at: str
    finished_at: str
    traces: list[StepTrace] = field(default_factory=list)
    scores: dict[str, Any] = field(default_factory=dict)
    valid: bool = True


def _call_with_timeout(fn: Callable[[], Any], timeout: float | None) -> Any:
    """Run `fn` in a daemon thread; raise TimeoutError if it overruns.

    A timed-out call can't be killed: its thread is abandoned and the run
    moves on.
    """
    if timeout is None:
        return fn()
    outcome: dict[str, Any] = {}

    def target() -> None:
        try:
            outcome["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller's thread
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError(f"step exceeded {timeout:g}s timeout")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def _run_step(
    fn: Callable[[], Any], adapter: MemoryAdapter, index: int, timeout: float | None
) -> tuple[Any, float, int, str | None]:
    """Call one adapter method with retry + timeout.

    Returns (result, latency_ms, retries, error); `error` is None on success.
    """
    retries = 0

    def on_retry(attempt: int, exc: BaseException, delay: float) -> None:
        nonlocal retries
        retries += 1
        logger.warning(
            "step %d: retry %d after %s: %s (sleeping %.1fs)",
            index, attempt, type(exc).__name__, exc, delay,
        )

    start = time.perf_counter()
    try:
        result = _call_with_timeout(
            lambda: retry_call(fn, retryable=adapter.retryable, on_retry=on_retry), timeout
        )
        error = None
    except Exception as exc:  # noqa: BLE001 - any adapter failure is recorded
        result, error = None, f"{type(exc).__name__}: {exc}"
        logger.warning("step %d failed: %s", index, error)
    latency_ms = (time.perf_counter() - start) * 1000
    return result, latency_ms, retries, error


def run_pack(
    pack: Pack,
    adapter: MemoryAdapter,
    backend_name: str,
    judge: Any = None,
    step_timeout: float | None = DEFAULT_STEP_TIMEOUT_S,
) -> RunResult:
    """Replay every step in `pack` against `adapter` in order.

    `judge` is an optional `core.scoring.judge.Judge`; when None the judge
    metrics score as None. `step_timeout` (seconds, None for no limit) caps
    each adapter call, retries included.
    """
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
            _, latency_ms, retries, error = _run_step(
                lambda event=event: adapter.write(event), adapter, index, step_timeout
            )
            traces.append(
                StepTrace(
                    step_index=index,
                    step_type="write",
                    session_id=step.session_id,
                    latency_ms=latency_ms,
                    detail={"key": step.key, "fact_type": step.fact_type, "retries": retries},
                    error=error,
                )
            )
        elif isinstance(step, QueryStep):
            result, latency_ms, retries, error = _run_step(
                lambda query=step.query: adapter.recall(query), adapter, index, step_timeout
            )
            if result is None:  # failed step: nothing was recalled
                result = RecallResult(context="")
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
                        "retries": retries,
                    },
                    error=error,
                )
            )
        else:  # pragma: no cover - schema guarantees this can't happen
            raise TypeError(f"unknown step type: {step!r}")

    finished_at = datetime.now(timezone.utc).isoformat()

    from membench.core.scoring import score_run

    scores = score_run(pack, traces, judge)
    error_rate = scores["error_rate"]
    return RunResult(
        pack_name=pack.name,
        backend_name=backend_name,
        started_at=started_at,
        finished_at=finished_at,
        traces=traces,
        scores=scores,
        valid=error_rate is None or error_rate <= MAX_ERROR_RATE,
    )
