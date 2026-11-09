"""Shared retry helper: exponential backoff with jitter.

The runner wraps every adapter call in `retry_call`, retrying only the
exception types the adapter declares in `MemoryAdapter.retryable` (e.g. a
vendor SDK's rate-limit or timeout errors). Anything else propagates at once.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

MAX_ATTEMPTS = 5
BASE_DELAY_S = 1.0
MAX_DELAY_S = 30.0


def backoff_delay(attempt: int, base: float = BASE_DELAY_S, cap: float = MAX_DELAY_S) -> float:
    """Delay before retry number `attempt` (1-based): half fixed, half random."""
    ceiling = min(cap, base * 2 ** (attempt - 1))
    return ceiling / 2 + random.uniform(0, ceiling / 2)


def retry_call(
    fn: Callable[[], T],
    *,
    retryable: tuple[type[BaseException], ...] = (),
    max_attempts: int = MAX_ATTEMPTS,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> T:
    """Call `fn`, retrying `retryable` exceptions up to `max_attempts` times.

    `on_retry(attempt, exc, delay)` is called before each sleep, so callers
    can log or count retries. The last failure is re-raised.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except retryable as exc:
            if attempt == max_attempts:
                raise
            delay = backoff_delay(attempt)
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover
