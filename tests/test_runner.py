from membench.adapters.full_context import FullContextAdapter
from membench.core.adapter import Event, MemoryAdapter, RecalledItem, RecallResult
from membench.core.runner import RunResult, StepTrace, run_pack
from membench.packs.schema import builtin_pack_path, load_pack, parse_pack


def test_run_pack_produces_a_trace_per_step():
    pack = load_pack(builtin_pack_path("mini"))
    adapter = FullContextAdapter()

    result = run_pack(pack, adapter, backend_name="full-context")

    assert isinstance(result, RunResult)
    assert result.pack_name == "mini"
    assert result.backend_name == "full-context"
    assert len(result.traces) == len(pack.steps)
    assert all(isinstance(t, StepTrace) for t in result.traces)
    assert result.scores  # Phase 2: populated, not the Phase 1 stub


def test_query_traces_carry_latency_and_tokens():
    pack = load_pack(builtin_pack_path("mini"))
    adapter = FullContextAdapter()

    result = run_pack(pack, adapter, backend_name="full-context")
    query_traces = [t for t in result.traces if t.step_type == "query"]

    assert len(query_traces) == 4
    for trace in query_traces:
        assert trace.latency_ms >= 0
        assert "token_count" in trace.detail
        assert "context" in trace.detail


def test_full_context_recalls_updated_fact():
    pack = load_pack(builtin_pack_path("mini"))
    adapter = FullContextAdapter()

    result = run_pack(pack, adapter, backend_name="full-context")
    query_traces = [t for t in result.traces if t.step_type == "query"]
    color_update_query = next(
        t for t in query_traces if "green" in t.detail["expect_contains"]
    )

    assert "green" in color_update_query.detail["context"]


class RecordingAdapter(MemoryAdapter):
    """Records every contract call; recall returns the live event log."""

    def __init__(self, config=None, fail_reset=False):
        super().__init__(config)
        self.calls: list[tuple] = []
        self.events: dict[str, Event] = {}
        self.fail_reset = fail_reset

    def write(self, event):
        self.calls.append(("write", event.id))
        self.events[event.id] = event

    def recall(self, query, *, session_id, at):
        self.calls.append(("recall", session_id, at))
        return RecallResult(
            context=" ".join(e.content for e in self.events.values()),
            items=[RecalledItem([i], e.content) for i, e in self.events.items()],
        )

    def forget(self, event_ids):
        self.calls.append(("forget", event_ids))
        for i in event_ids:
            self.events.pop(i)

    def flush(self):
        self.calls.append(("flush",))

    def reset(self):
        self.calls.append(("reset",))
        if self.fail_reset:
            raise RuntimeError("reset broke")
        self.events.clear()

    def usage(self):
        return {"llm_tokens": 7}


def _pack(*episodes):
    return parse_pack({"name": "p", "episodes": [{"steps": list(e)} for e in episodes]})


def _w(event_id, content="x y"):
    return {"type": "write", "session_id": "s1", "id": event_id, "fact_key": "k",
            "role": "user", "content": content}


def _q(at=None):
    step = {"type": "query", "session_id": "s1", "query": "q"}
    return {**step, "timestamp": at} if at else step


def test_forget_step_reaches_adapter_forget_and_is_traced():
    adapter = RecordingAdapter()
    pack = _pack([_w("e1"), _w("e2"), {"type": "forget", "session_id": "s1",
                                       "target": ["e1", "e2"]}])

    result = run_pack(pack, adapter, backend_name="rec")

    assert ("forget", ["e1", "e2"]) in adapter.calls
    trace = result.traces[2]
    assert trace.step_type == "forget"
    assert trace.detail["target"] == ["e1", "e2"]
    assert trace.latency_ms >= 0 and trace.error is None


def test_flush_runs_before_a_query_that_follows_writes():
    adapter = RecordingAdapter()
    pack = _pack([_w("e1"), _q("2024-02-01T00:00:00+00:00"), _q()])

    result = run_pack(pack, adapter, backend_name="rec")

    assert [c[0] for c in adapter.calls] == ["write", "flush", "recall", "recall"]
    assert adapter.calls[2] == ("recall", "s1", "2024-02-01T00:00:00+00:00")
    first, second = result.traces[1:]
    assert first.detail["ingest_latency_ms"] >= 0
    assert second.detail["ingest_latency_ms"] is None  # nothing new to ingest
    assert first.detail["items"] == [{"event_ids": ["e1"], "text": "x y", "score": None}]


def test_runner_counts_tokens_with_configured_tokenizer():
    pack = _pack([_w("e1", "a b c"), _q()])

    default = run_pack(pack, RecordingAdapter(), backend_name="rec")
    custom = run_pack(pack, RecordingAdapter(), backend_name="rec", tokenizer=len)

    assert default.traces[1].detail["token_count"] == 3
    assert custom.traces[1].detail["token_count"] == 5


def test_reset_between_episodes_and_usage_recorded():
    adapter = RecordingAdapter()
    pack = _pack([_w("e1"), _q()], [_w("e2"), _q()])

    result = run_pack(pack, adapter, backend_name="rec")

    assert [c[0] for c in adapter.calls].count("reset") == 1
    assert adapter.calls.index(("reset",)) == 3  # after episode 0's steps
    assert result.traces[3].detail["context"] == "x y"  # only e2
    assert result.usage == {"llm_tokens": 7}


def test_failed_reset_marks_the_episode_errored():
    adapter = RecordingAdapter(fail_reset=True)
    pack = _pack([_w("e1"), _q()], [_w("e2"), _q()])

    result = run_pack(pack, adapter, backend_name="rec")

    assert all(t.error is None for t in result.traces[:2])
    assert all("reset before episode failed" in t.error for t in result.traces[2:])
    assert ("write", "e2") not in adapter.calls


class StubReaderClient:
    model = "claude-haiku-4-5"
    base_url = None

    def __init__(self, fail_on=None):
        self.prompts: list[str] = []
        self.fail_on = fail_on

    def respond(self, prompt):
        self.prompts.append(prompt)
        if self.fail_on and self.fail_on in prompt:
            raise RuntimeError("reader down")
        return " the answer \n", {"input_tokens": 10, "output_tokens": 2}


def test_reader_answers_queries_with_gold_or_abstain_only():
    from membench.core.reader import Reader

    client = StubReaderClient(fail_on="broken?")
    pack = _pack([
        _w("e1"),
        {**_q("2025-05-01T00:00:00+00:00"), "query": "gold?", "answer": "x"},
        {**_q(), "query": "abstain?", "abstain": True},
        {**_q(), "query": "broken?", "answer": "x"},
        _q(),  # no answer, no abstain: the reader is not asked
    ])
    reader = Reader(client)

    result = run_pack(pack, RecordingAdapter(), backend_name="rec", reader=reader)

    gold, abstain, broken, plain = result.traces[1:]
    assert gold.detail["reader_answer"] == "the answer"
    assert abstain.detail["reader_answer"] == "the answer"
    assert broken.detail["reader_answer"] is None
    assert broken.detail["reader_error"] == "RuntimeError: reader down"
    assert "reader_answer" not in plain.detail
    assert "CURRENT DATE: 2025-05-01T00:00:00+00:00" in client.prompts[0]
    assert "x y" in client.prompts[0] and "gold?" in client.prompts[0]
    assert result.manifest["reader"] == {
        "model": "claude-haiku-4-5", "base_url": None, "prompt_version": "1",
    }
    assert result.cost["reader"]["calls"] == 3
    assert result.cost["reader"]["input_tokens"] == 20  # the failed call reported no usage
    assert result.cost["reader"]["usd"] == 20 * 1.0 / 1e6 + 4 * 5.0 / 1e6


def test_ingest_tokens_exclude_tokens_spent_in_recall():
    class Metered(RecordingAdapter):
        tokens = 0

        def write(self, event):
            super().write(event)
            self.tokens += 100  # extraction at write time

        def recall(self, query, *, session_id, at):
            self.tokens += 5  # query embedding
            return super().recall(query, session_id=session_id, at=at)

        def usage(self):
            return {"llm_input_tokens": self.tokens}

    pack = _pack([_w("e1"), _w("e2"), _q(), _q()])
    result = run_pack(pack, Metered(), backend_name="rec")

    assert result.usage == {"llm_input_tokens": 210}
    assert result.cost["ingest_tokens_per_1k_writes"] == 200 / 2 * 1000
