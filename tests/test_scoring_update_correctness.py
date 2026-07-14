from membench.core.runner import StepTrace
from membench.core.scoring.update_correctness import update_correctness
from membench.packs.schema import Pack, QueryStep, WriteStep


def _write(key: str, fact_type: str = "state") -> WriteStep:
    return WriteStep(
        type="write",
        session_id="s1",
        key=key,
        role="user",
        content="c",
        timestamp="t",
        fact_type=fact_type,
    )


def _query(expect_contains: list[str], expect_key: str | None) -> QueryStep:
    return QueryStep(
        type="query",
        session_id="s1",
        query="q",
        timestamp="t",
        expect_contains=expect_contains,
        expect_key=expect_key,
    )


def _write_trace(index: int) -> StepTrace:
    return StepTrace(step_index=index, step_type="write", session_id="s1", latency_ms=1.0, detail={})


def _query_trace(index: int, context: str) -> StepTrace:
    return StepTrace(
        step_index=index,
        step_type="query",
        session_id="s1",
        latency_ms=1.0,
        detail={"context": context, "token_count": len(context.split())},
    )


def test_query_after_update_reflects_new_fact():
    pack = Pack(
        name="p",
        description="",
        steps=[
            _write("fact_color", "state"),
            _write("fact_color", "update"),
            _query(["green"], "fact_color"),
        ],
    )
    traces = [_write_trace(0), _write_trace(1), _query_trace(2, "color is now green")]

    assert update_correctness(pack, traces) == 1.0


def test_query_after_update_still_stale_fails():
    pack = Pack(
        name="p",
        description="",
        steps=[
            _write("fact_color", "state"),
            _write("fact_color", "update"),
            _query(["green"], "fact_color"),
        ],
    )
    traces = [_write_trace(0), _write_trace(1), _query_trace(2, "color is still blue")]

    assert update_correctness(pack, traces) == 0.0


def test_query_before_any_update_is_not_eligible():
    pack = Pack(
        name="p",
        description="",
        steps=[
            _write("fact_color", "state"),
            _query(["blue"], "fact_color"),
            _write("fact_color", "update"),
        ],
    )
    traces = [_write_trace(0), _query_trace(1, "color is blue"), _write_trace(2)]

    assert update_correctness(pack, traces) is None


def test_contradict_fact_type_also_counts_as_update():
    pack = Pack(
        name="p",
        description="",
        steps=[
            _write("fact_color", "state"),
            _write("fact_color", "contradict"),
            _query(["green"], "fact_color"),
        ],
    )
    traces = [_write_trace(0), _write_trace(1), _query_trace(2, "green now")]

    assert update_correctness(pack, traces) == 1.0


def test_unrelated_key_not_eligible():
    pack = Pack(
        name="p",
        description="",
        steps=[
            _write("fact_color", "update"),
            _query(["Austin"], "fact_city"),
        ],
    )
    traces = [_write_trace(0), _query_trace(1, "I live in Austin")]

    assert update_correctness(pack, traces) is None


def test_mixed_eligible_queries_average():
    pack = Pack(
        name="p",
        description="",
        steps=[
            _write("fact_color", "update"),
            _query(["green"], "fact_color"),
            _write("fact_city", "update"),
            _query(["Seattle"], "fact_city"),
        ],
    )
    traces = [
        _write_trace(0),
        _query_trace(1, "green now"),
        _write_trace(2),
        _query_trace(3, "still Austin"),
    ]

    assert update_correctness(pack, traces) == 0.5
