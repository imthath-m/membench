from membench.core.runner import StepTrace
from membench.core.scoring.recall_precision import recall_precision
from membench.packs.schema import Pack, QueryStep, WriteStep


def _query(query: str, expect_contains: list[str], expect_key: str | None = None) -> QueryStep:
    return QueryStep(
        type="query",
        session_id="s1",
        query=query,
        timestamp="t",
        expect_contains=expect_contains,
        expect_key=expect_key,
    )


def _trace(index: int, context: str) -> StepTrace:
    return StepTrace(
        step_index=index,
        step_type="query",
        session_id="s1",
        latency_ms=1.0,
        detail={"context": context, "token_count": len(context.split())},
    )


def test_all_queries_pass():
    pack = Pack(
        name="p",
        description="",
        steps=[_query("q1", ["blue"]), _query("q2", ["Austin"])],
    )
    traces = [_trace(0, "my color is blue"), _trace(1, "I live in Austin")]

    assert recall_precision(pack, traces) == 1.0


def test_some_queries_fail():
    pack = Pack(
        name="p",
        description="",
        steps=[_query("q1", ["blue"]), _query("q2", ["green"])],
    )
    traces = [_trace(0, "my color is blue"), _trace(1, "my color is blue")]

    assert recall_precision(pack, traces) == 0.5


def test_multiple_expect_contains_all_required():
    pack = Pack(name="p", description="", steps=[_query("q1", ["blue", "eyes"])])
    traces = [_trace(0, "my color is blue")]

    assert recall_precision(pack, traces) == 0.0


def test_case_insensitive_match():
    pack = Pack(name="p", description="", steps=[_query("q1", ["BLUE"])])
    traces = [_trace(0, "my color is blue")]

    assert recall_precision(pack, traces) == 1.0


def test_queries_without_expect_contains_are_ignored():
    pack = Pack(name="p", description="", steps=[_query("q1", [])])
    traces = [_trace(0, "anything")]

    assert recall_precision(pack, traces) is None


def test_write_steps_do_not_count():
    write = WriteStep(
        type="write", session_id="s1", key="k", role="user", content="c", timestamp="t"
    )
    pack = Pack(name="p", description="", steps=[write, _query("q1", ["blue"])])
    traces = [
        StepTrace(step_index=0, step_type="write", session_id="s1", latency_ms=1.0, detail={}),
        _trace(1, "my color is blue"),
    ]

    assert recall_precision(pack, traces) == 1.0
