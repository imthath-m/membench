import logging
import time

import pytest

import membench.core.retry as retry_module
from membench.adapters.full_context import FullContextAdapter
from membench.core.runner import MAX_ERROR_RATE, run_pack
from membench.packs.schema import builtin_pack_path, load_pack


class RateLimited(Exception):
    """Stand-in for a vendor SDK's 429 error."""


class FlakyAdapter(FullContextAdapter):
    """Raises a retryable error on the first `fail_times` recalls."""

    retryable = (RateLimited,)

    def __init__(self, fail_times=1):
        super().__init__()
        self.fail_times = fail_times

    def recall(self, query, **kwargs):
        if self.fail_times:
            self.fail_times -= 1
            raise RateLimited("429 Too Many Requests")
        return super().recall(query, **kwargs)


class BrokenRecallAdapter(FullContextAdapter):
    """Raises a non-retryable error on the Nth recall."""

    def __init__(self, fail_on=2):
        super().__init__()
        self.fail_on = fail_on
        self.recalls = 0

    def recall(self, query, **kwargs):
        self.recalls += 1
        if self.recalls == self.fail_on:
            raise RuntimeError("backend exploded")
        return super().recall(query, **kwargs)


class SlowAdapter(FullContextAdapter):
    def recall(self, query, **kwargs):
        time.sleep(0.5)
        return super().recall(query, **kwargs)


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(retry_module.time, "sleep", lambda s: None)


@pytest.fixture
def mini():
    return load_pack(builtin_pack_path("mini"))


def _queries(result):
    return [t for t in result.traces if t.step_type == "query"]


def test_injected_429_is_retried_and_logged(mini, no_sleep, caplog):
    with caplog.at_level(logging.WARNING, logger="membench"):
        result = run_pack(mini, FlakyAdapter(fail_times=2), "flaky")

    first = _queries(result)[0]
    assert first.error is None
    assert first.detail["retries"] == 2
    assert result.scores["error_rate"] == 0
    assert result.valid
    assert "RateLimited" in caplog.text and "retry 1" in caplog.text


def test_adapter_exception_mid_pack_is_recorded_and_run_continues(mini):
    result = run_pack(mini, BrokenRecallAdapter(fail_on=2), "broken")

    queries = _queries(result)
    assert len(result.traces) == len(mini.steps)
    assert queries[1].error == "RuntimeError: backend exploded"
    assert queries[1].detail["context"] == ""
    assert all(q.error is None for i, q in enumerate(queries) if i != 1)
    assert result.scores["error_rate"] == 1 / len(mini.steps)
    # the failed query counts against recall precision; the rest still score
    assert result.scores["recall_precision"] == 3 / 4


def test_run_is_invalid_above_error_rate_threshold(mini, no_sleep):
    result = run_pack(mini, FlakyAdapter(fail_times=100), "flaky")

    assert result.scores["error_rate"] > MAX_ERROR_RATE
    assert result.valid is False
    assert result.scores["token_cost"] is None  # no successful query


def test_step_timeout_records_error(mini):
    result = run_pack(mini, SlowAdapter(), "slow", step_timeout=0.05)

    queries = _queries(result)
    assert all(q.error == "TimeoutError: step exceeded 0.05s timeout" for q in queries)
    assert all(t.error is None for t in result.traces if t.step_type == "write")
    assert result.scores["p95_latency_ms"] < 50  # timed-out steps are excluded
