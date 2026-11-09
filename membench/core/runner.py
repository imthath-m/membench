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
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from membench.core.adapter import Event, MemoryAdapter, RecallResult
from membench.core.retry import retry_call
from membench.core.tokens import count_tokens
from membench.packs.schema import ForgetStep, Pack, QueryStep, WriteStep

logger = logging.getLogger("membench")

DEFAULT_STEP_TIMEOUT_S = 120.0
MAX_ERROR_RATE = 0.1


@dataclass
class StepTrace:
    step_index: int
    step_type: str  # "write" | "query" | "forget"
    session_id: str
    latency_ms: float
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    episode: int = 0


@dataclass
class RunResult:
    pack_name: str
    backend_name: str
    started_at: str
    finished_at: str
    traces: list[StepTrace] = field(default_factory=list)
    scores: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, int] | None = None  # adapter.usage(): backend LLM/embedding tokens
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
    tokenizer: Callable[[str], int] = count_tokens,
) -> RunResult:
    """Replay every step in `pack` against `adapter` in order.

    `judge` is an optional `core.scoring.judge.Judge`; when None the judge
    metrics score as None. `step_timeout` (seconds, None for no limit) caps
    each adapter call, retries included. `tokenizer` counts each recalled
    `context`. `adapter.reset()` runs between episodes; if it fails, every
    step of that episode is recorded as an error. `adapter.flush()` runs
    before a query that follows writes or forgets, timed as `ingest_latency_ms`.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    traces: list[StepTrace] = []
    episode, reset_error, pending_ingest = 0, None, False

    for index, step in enumerate(pack.steps):
        if step.episode != episode:
            episode, pending_ingest = step.episode, False
            _, _, _, reset_error = _run_step(adapter.reset, adapter, index, step_timeout)

        def call(
            fn: Callable[[], Any], index: int = index, reset_error: str | None = reset_error
        ) -> tuple[Any, float, int, str | None]:
            if reset_error is not None:  # this episode never started cleanly
                return None, 0.0, 0, f"reset before episode failed: {reset_error}"
            return _run_step(fn, adapter, index, step_timeout)

        if isinstance(step, WriteStep):
            event = Event(
                id=step.id,
                session_id=step.session_id,
                role=step.role,
                content=step.content,
                timestamp=step.timestamp,
                metadata=step.metadata,
            )
            _, latency_ms, retries, error = call(lambda event=event: adapter.write(event))
            pending_ingest = True
            detail = {
                "event_id": step.id,
                "fact_key": step.fact_key,
                "fact_type": step.fact_type,
                "retries": retries,
            }
        elif isinstance(step, ForgetStep):
            _, latency_ms, retries, error = call(
                lambda target=list(step.target): adapter.forget(target)
            )
            pending_ingest = True
            detail = {"target": step.target, "retries": retries}
        elif isinstance(step, QueryStep):
            ingest_ms, error, result, latency_ms, retries = None, None, None, 0.0, 0
            if pending_ingest:
                _, ingest_ms, _, error = call(adapter.flush)
                pending_ingest = False
                error = error and f"flush: {error}"
            if error is None:
                result, latency_ms, retries, error = call(
                    lambda step=step: adapter.recall(
                        step.query, session_id=step.session_id, at=step.timestamp
                    )
                )
            if result is None:  # failed step: nothing was recalled
                result = RecallResult(context="")
            detail = {
                "query": step.query,
                "context": result.context,
                "token_count": tokenizer(result.context),
                "items": [asdict(item) for item in result.items],
                "ingest_latency_ms": ingest_ms,
                "expect_contains": step.expect_contains,
                "expect_key": step.expect_key,
                "retries": retries,
            }
        else:  # pragma: no cover - schema guarantees this can't happen
            raise TypeError(f"unknown step type: {step!r}")
        traces.append(
            StepTrace(
                step_index=index,
                step_type=step.type,
                session_id=step.session_id,
                latency_ms=latency_ms,
                detail=detail,
                error=error,
                episode=episode,
            )
        )

    usage, _, _, _ = _run_step(adapter.usage, adapter, len(pack.steps), step_timeout)
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
        usage=usage,
        valid=error_rate is None or error_rate <= MAX_ERROR_RATE,
    )
