"""Shared wait loop for backends that ingest asynchronously (used by `flush()`).

Hosted backends hand back a job id per write (Mem0 Platform events, Zep
tasks); `drain` polls them in order until each is done.
"""

from __future__ import annotations

import time
from collections.abc import Callable

POLL_INTERVAL_S = 1.0
FLUSH_TIMEOUT_S = 600.0


def drain(
    pending: list[str],
    done: Callable[[str], bool],
    *,
    interval_s: float = POLL_INTERVAL_S,
    timeout_s: float = FLUSH_TIMEOUT_S,
) -> None:
    """Poll `done(job_id)` for each id in `pending`, removing ids as they finish.

    `done` returns True when the job has finished and raises if it failed (the
    failed id is dropped, so a later flush does not raise for it again).
    Raises TimeoutError when jobs are still running after `timeout_s`.
    """
    deadline = time.monotonic() + timeout_s
    while pending:
        try:
            finished = done(pending[0])
        except Exception:
            pending.pop(0)
            raise
        if finished:
            pending.pop(0)
        elif time.monotonic() > deadline:
            raise TimeoutError(f"{len(pending)} ingestion job(s) still running after {timeout_s}s")
        else:
            time.sleep(interval_s)


def poll_options(config: dict) -> dict[str, float]:
    """`drain` kwargs from adapter config keys `poll_interval_s` / `flush_timeout_s`."""
    return {
        "interval_s": float(config.get("poll_interval_s", POLL_INTERVAL_S)),
        "timeout_s": float(config.get("flush_timeout_s", FLUSH_TIMEOUT_S)),
    }
