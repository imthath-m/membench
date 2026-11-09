import pytest

import membench.core.retry as retry_module
from membench.core.retry import backoff_delay, retry_call


class Transient(Exception):
    pass


@pytest.fixture
def sleeps(monkeypatch):
    recorded = []
    monkeypatch.setattr(retry_module.time, "sleep", recorded.append)
    return recorded


def _flaky(failures, exc=Transient):
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] <= failures:
            raise exc("boom")
        return "ok"

    return fn, calls


def test_retry_call_retries_declared_exceptions(sleeps):
    fn, calls = _flaky(2)
    seen = []

    result = retry_call(fn, retryable=(Transient,), on_retry=lambda a, e, d: seen.append(a))

    assert result == "ok"
    assert calls["n"] == 3
    assert seen == [1, 2]
    assert len(sleeps) == 2


def test_retry_call_does_not_retry_other_exceptions(sleeps):
    fn, calls = _flaky(1, exc=RuntimeError)

    with pytest.raises(RuntimeError):
        retry_call(fn, retryable=(Transient,))
    assert calls["n"] == 1
    assert sleeps == []


def test_retry_call_reraises_after_max_attempts(sleeps):
    fn, calls = _flaky(10)

    with pytest.raises(Transient):
        retry_call(fn, retryable=(Transient,), max_attempts=3)
    assert calls["n"] == 3
    assert len(sleeps) == 2


def test_backoff_delay_grows_exponentially_with_bounded_jitter():
    for attempt, ceiling in [(1, 1.0), (2, 2.0), (3, 4.0), (10, 30.0)]:
        for _ in range(20):
            assert ceiling / 2 <= backoff_delay(attempt) <= ceiling
