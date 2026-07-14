from membench.adapters.full_context import FullContextAdapter
from membench.core.runner import StepTrace, run_pack
from membench.core.scoring.contradiction_rate import contradiction_rate
from membench.core.scoring.judge import CachedJudge, Verdict
from membench.core.scoring.semantic_recall import semantic_recall
from membench.packs.schema import JudgeSpec, Pack, QueryStep, builtin_pack_path, load_pack


class StubJudge:
    """Judge that passes when a keyword appears in the evidence."""

    model = "stub-model"

    def __init__(self, keyword):
        self.keyword = keyword

    def score(self, rubric, evidence):
        return Verdict(passed=self.keyword.lower() in evidence.lower(), reason="stub")


class CountingClient:
    """Client stub that counts calls (used to prove cache hits skip it)."""

    model = "stub-model"

    def __init__(self):
        self.calls = 0

    def score(self, rubric, evidence):
        self.calls += 1
        return Verdict(passed="green" in evidence.lower(), reason="stub")


def _query(judge_spec):
    return QueryStep(type="query", session_id="s1", query="q", timestamp="t", judge=judge_spec)


def _query_trace(context):
    return StepTrace(step_index=0, step_type="query", session_id="s1", latency_ms=1.0,
                     detail={"context": context})


# --- metric functions -------------------------------------------------------

def test_semantic_recall_none_without_judge():
    pack = Pack("p", "", [_query(JudgeSpec(semantic_recall="r"))])
    assert semantic_recall(pack, [_query_trace("green")], judge=None) is None


def test_semantic_recall_none_without_rubric_steps():
    pack = Pack("p", "", [_query(None)])
    assert semantic_recall(pack, [_query_trace("green")], judge=StubJudge("green")) is None


def test_semantic_recall_scores_judged_queries():
    pack = Pack("p", "", [
        _query(JudgeSpec(semantic_recall="favorite color is green")),
        _query(JudgeSpec(semantic_recall="favorite color is green")),
    ])
    traces = [_query_trace("color is green"), _query_trace("color is blue")]
    assert semantic_recall(pack, traces, judge=StubJudge("green")) == 0.5


def test_contradiction_rate_scores_judged_queries():
    pack = Pack("p", "", [_query(JudgeSpec(contradiction="contradicts"))])
    # StubJudge "passed" == contradiction present.
    assert contradiction_rate(pack, [_query_trace("blue and green")], StubJudge("green")) == 1.0
    assert contradiction_rate(pack, [_query_trace("only blue")], StubJudge("green")) == 0.0


def test_contradiction_rate_none_without_judge():
    pack = Pack("p", "", [_query(JudgeSpec(contradiction="r"))])
    assert contradiction_rate(pack, [_query_trace("x")], judge=None) is None


# --- integration through run_pack + score_run ------------------------------

def test_run_pack_populates_judge_metrics_with_stub_judge():
    pack = load_pack(builtin_pack_path("mini"))
    result = run_pack(pack, FullContextAdapter(), "full-context", judge=StubJudge("green"))

    # full-context recall contains both the stale blue and the updated green.
    assert result.scores["semantic_recall"] == 1.0
    assert result.scores["contradiction_rate"] == 1.0


def test_run_pack_judge_metrics_null_when_no_judge():
    pack = load_pack(builtin_pack_path("mini"))
    result = run_pack(pack, FullContextAdapter(), "full-context", judge=None)

    assert set(result.scores) == {
        "recall_precision", "update_correctness", "token_cost", "p95_latency_ms",
        "semantic_recall", "contradiction_rate",
    }
    assert result.scores["semantic_recall"] is None
    assert result.scores["contradiction_rate"] is None


# --- cache reproducibility --------------------------------------------------

def test_cache_makes_second_run_call_free_and_identical(tmp_path):
    pack = load_pack(builtin_pack_path("mini"))
    cache = tmp_path / "cache.json"

    client1 = CountingClient()
    r1 = run_pack(pack, FullContextAdapter(), "full-context", judge=CachedJudge(client1, cache))
    assert client1.calls > 0
    assert r1.scores["semantic_recall"] is not None
    assert r1.scores["contradiction_rate"] is not None

    client2 = CountingClient()
    r2 = run_pack(pack, FullContextAdapter(), "full-context", judge=CachedJudge(client2, cache))
    assert client2.calls == 0  # every verdict served from the cache
    assert r1.scores["semantic_recall"] == r2.scores["semantic_recall"]
    assert r1.scores["contradiction_rate"] == r2.scores["contradiction_rate"]
