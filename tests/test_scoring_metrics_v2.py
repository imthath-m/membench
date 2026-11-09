"""Fixture-trace tests for the Phase 11 metrics (no adapter involved)."""

from __future__ import annotations

from membench.core.runner import StepTrace
from membench.core.scoring import score_all
from membench.core.scoring.abstention import abstention, is_abstention
from membench.core.scoring.evidence_recall import evidence_recall_at_k
from membench.core.scoring.forget_compliance import forget_compliance
from membench.core.scoring.judge import JudgeError, Verdict
from membench.core.scoring.qa_accuracy import qa_accuracy
from membench.core.scoring.staleness import contains_any, staleness
from membench.core.scoring.temporal_accuracy import temporal_accuracy
from membench.packs.schema import parse_pack


def _w(id_, fact_key, content, fact_type="state"):
    return {"type": "write", "session_id": "s", "id": id_, "fact_key": fact_key,
            "role": "user", "content": content, "fact_type": fact_type}


def _q(query, **fields):
    return {"type": "query", "session_id": "s", "query": query, **fields}


def _traces(pack, recalls, error_at=()):
    """`recalls`: per query, in order, (context, [item event_ids, ...][, extra detail])."""
    traces, recalls = [], iter(recalls)
    for i, step in enumerate(pack.steps):
        detail = {"retries": 0}
        if step.type == "query":
            context, item_ids, extra = _unpack(next(recalls))
            detail.update(context=context, token_count=len(context.split()), items=[{"event_ids": ids, "text": "", "score": None}
                                                  for ids in item_ids], **extra)
        error = "RuntimeError: down" if i in error_at else None
        traces.append(StepTrace(i, step.type, "s", 1.0, detail, error=error))
    return traces


def _unpack(recall):
    context, item_ids, *rest = recall
    return context, item_ids, (rest[0] if rest else {})


UPDATE_PACK = parse_pack({"steps": [
    _w("a", "city", "I live in Austin."),
    _q("Where?", expect_contains=["Austin"], expect_key="city", expect_absent=["Seattle"]),
    _w("b", "city", "I moved to Seattle.", "update"),
    _q("Where now?", expect_contains=["Seattle"], expect_key="city", expect_absent=["Austin"],
       evidence=["b"]),
    _q("Where now, again?", expect_contains=["Seattle"], expect_key="city",
       expect_absent=["Austin"], evidence=["a", "b"]),
]})


def test_staleness_counts_post_update_queries_still_holding_the_stale_value():
    traces = _traces(UPDATE_PACK, [
        ("Austin", [["a"]]),  # pre-update: not scored
        ("Austin. Seattle.", [["a"], ["b"]]),  # stale
        ("Seattle", [["b"]]),  # fresh
    ])
    assert staleness(UPDATE_PACK, traces) == 0.5


def test_contains_any_matches_whole_words_and_phrases_only():
    assert not contains_any(["Pro"], "Your refund was approved.")
    assert not contains_any(["Pro"], "Products and process notes")
    assert contains_any(["Pro"], "plan: Pro.")
    assert contains_any(["pro"], "Switched to PRO yesterday")
    assert contains_any(["Pro plan"], "on the pro plan now")
    assert contains_any(["$49.99", "C++"], "costs $49.99/month")
    assert contains_any(["C++"], "wrote it in C++, then Go")
    assert contains_any(["v2.1"], "upgraded to v2.1.")
    assert not contains_any(["v2.1"], "upgraded to v2.10")
    assert not contains_any(["a.b"], "axb")  # needles are literal, not regex
    assert not contains_any([], "anything")


def test_staleness_ignores_a_stale_value_inside_another_word():
    pack = parse_pack({"steps": [
        _w("a", "plan", "I am on the Pro plan."),
        _w("b", "plan", "I switched to Basic.", "update"),
        _q("Plan?", expect_contains=["Basic"], expect_key="plan", expect_absent=["Pro"]),
        _q("Plan again?", expect_contains=["Basic"], expect_key="plan", expect_absent=["Pro"]),
    ]})
    traces = _traces(pack, [("Basic. Refund approved.", [["b"]]), ("plan: Pro.", [["a"]])])
    assert staleness(pack, traces) == 0.5


def test_staleness_skips_failed_recalls_and_is_none_without_expect_absent():
    traces = _traces(UPDATE_PACK, [("", []), ("", []), ("Seattle", [["b"]])], error_at={3})
    assert staleness(UPDATE_PACK, traces) == 0.0
    pack = parse_pack({"steps": [_w("a", "k", "x"), _w("b", "k", "y", "update"),
                                 _q("?", expect_contains=["y"], expect_key="k")]})
    assert staleness(pack, _traces(pack, [("x y", [])])) is None


def test_evidence_recall_at_k_is_the_mean_fraction_of_evidence_recalled():
    traces = _traces(UPDATE_PACK, [
        ("", []),
        ("Seattle", [["b"]]),  # 1/1
        ("Seattle", [["b", "zzz"]]),  # 1/2 (a missing)
    ])
    assert evidence_recall_at_k(UPDATE_PACK, traces) == 0.75


FORGET_PACK = parse_pack({"steps": [
    _w("pw", "secret", "The password is hunter2."),
    _w("x", "other", "I like tea."),
    {"type": "forget", "session_id": "s", "target": "pw"},
    _q("Password?", category="forget", abstain=True, expect_absent=["hunter2"]),
    _q("Password again?", category="forget", abstain=True, expect_absent=["hunter2"]),
    _q("Password once more?", category="forget", abstain=True, expect_absent=["hunter2"]),
]})


def test_forget_compliance_fails_on_forgotten_text_or_forgotten_event_id():
    traces = _traces(FORGET_PACK, [
        ("I like tea.", [["x"]]),  # compliant
        ("The password is hunter2.", []),  # text leaked
        ("something", [["pw"]]),  # an item still derives from the forgotten event
    ])
    assert forget_compliance(FORGET_PACK, traces) == 1 / 3


def test_forget_compliance_is_none_without_forget_queries():
    assert forget_compliance(UPDATE_PACK, _traces(UPDATE_PACK, [("", [])] * 3)) is None


def test_temporal_accuracy_scores_only_temporal_queries():
    pack = parse_pack({"steps": [
        _w("a", "plan", "Basic plan"), _w("b", "plan", "Pro plan", "update"),
        _q("Plan in January?", category="temporal", expect_contains=["Basic"]),
        _q("When did it change?", category="temporal", expect_contains=["2025-02-20"]),
        _q("Plan now?", category="update", expect_contains=["Pro"]),
    ]})
    traces = _traces(pack, [("Basic plan", []), ("Pro plan", []), ("Pro plan", [])])
    assert temporal_accuracy(pack, traces) == 0.5
    assert temporal_accuracy(UPDATE_PACK, _traces(UPDATE_PACK, [("", [])] * 3)) is None


def test_is_abstention():
    assert is_abstention("I don't know")
    assert is_abstention("I don’t know.")
    assert is_abstention("That is not mentioned in memory.")
    assert not is_abstention("Your phone number is 555-0100.")


ABSTAIN_PACK = parse_pack({"steps": [
    _w("a", "name", "My name is Dana."),
    _q("Phone?", abstain=True),
    _q("Phone??", abstain=True),
    _q("Phone???", abstain=True),
    _q("Phone????", abstain=True),
    _q("Name?", answer="Dana"),
]})


def test_abstention_from_reader_answers():
    traces = _traces(ABSTAIN_PACK, [
        ("Dana", [], {"reader_answer": "I don't know"}),
        ("Dana", [], {"reader_answer": "555-0100"}),
        ("Dana", [], {"reader_answer": None, "reader_error": "HTTPError: 500"}),  # skipped
        ("", [], {"reader_answer": None}),  # failed recall: a miss
        ("Dana", [], {"reader_answer": "Dana"}),
    ], error_at={4})
    assert abstention(ABSTAIN_PACK, traces) == 1 / 3


def test_abstention_is_none_without_reader():
    assert abstention(ABSTAIN_PACK, _traces(ABSTAIN_PACK, [("Dana", [])] * 5)) is None


class AnswerJudge:
    """Passes when the reference answer appears in the evidence; errors on 'boom'."""

    model = "stub"

    def __init__(self) -> None:
        self.calls = 0

    def score(self, rubric: str, evidence: str) -> Verdict:
        self.calls += 1
        if evidence == "boom":
            raise JudgeError("unparseable")
        gold = rubric.split("reference answer ")[1].split(" (wording")[0].strip("'\"")
        return Verdict(passed=gold.lower() in evidence.lower())


QA_PACK = parse_pack({"steps": [
    _w("a", "name", "My name is Dana."),
    _q("Name?", answer="Dana", category="single_hop"),
    _q("Name again?", answer="Dana", category="single_hop"),
    _q("Name once more?", answer="Dana", category="multi_hop"),
    _q("And again?", answer="Dana", category="multi_hop"),
    _q("No reader here?", answer="Dana"),
]})


def test_qa_accuracy_judges_reader_answers_against_gold():
    traces = _traces(QA_PACK, [
        ("Dana", [], {"reader_answer": "Her name is Dana."}),
        ("Dana", [], {"reader_answer": "I don't know"}),
        ("Dana", [], {"reader_answer": "boom"}),  # judge error: skipped, recorded
        ("", [], {"reader_answer": None}),  # failed recall: wrong
        ("Dana", []),  # reader not run: skipped
    ], error_at={4})
    assert qa_accuracy(QA_PACK, traces, AnswerJudge()) == 1 / 3
    assert traces[3].detail["judge_error"] == {"qa_accuracy": "unparseable"}
    assert qa_accuracy(QA_PACK, traces, None) is None


def test_score_all_breaks_down_by_category_and_asks_the_judge_once_per_prompt():
    traces = _traces(QA_PACK, [
        ("Dana", [], {"reader_answer": "Dana"}),
        ("Dana", [], {"reader_answer": "Dana"}),
        ("x", [], {"reader_answer": "Dana"}),
        ("x", [], {"reader_answer": "nope"}),
        ("Dana", [], {"reader_answer": "Dana"}),
    ])
    judge = AnswerJudge()
    scores, by_category = score_all(QA_PACK, traces, judge)
    assert scores["qa_accuracy"] == 0.8
    assert judge.calls == 5  # the breakdown reuses the overall pass's verdicts
    assert set(by_category) == {"single_hop", "multi_hop", "uncategorized"}
    assert by_category["single_hop"]["n"] == 2
    assert by_category["single_hop"]["qa_accuracy"] == 1.0
    assert by_category["multi_hop"]["qa_accuracy"] == 0.5
    assert by_category["uncategorized"]["n"] == 1


def test_score_all_breakdown_keeps_updates_in_scope():
    traces = _traces(UPDATE_PACK, [("Austin", [["a"]]), ("Seattle", [["b"]]),
                                   ("Seattle", [["b"]])])
    scores, by_category = score_all(UPDATE_PACK, traces)
    assert by_category["uncategorized"]["recall_precision"] == 1.0
    assert by_category["uncategorized"]["evidence_recall_at_k"] == 0.75
    assert scores["staleness"] == 0.0


class YesNoJudge:
    """Records official-style prompts; says yes when the response says 'Dana'
    or (abstention prompt) 'not know'."""

    model = "stub"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def ask(self, prompt, parse):
        self.prompts.append(prompt)
        response = prompt.split("Model Response: ")[1].split("\n\n")[0]
        return parse("yes" if "Dana" in response or "not know" in response else "no")


LME_PACK = parse_pack({"steps": [
    _w("a", "name", "My name is Dana."),
    _q("Name?", answer="Dana", category="knowledge-update"),
    _q("Cat?", answer="Only a name was given.", abstain=True, category="single-session-user"),
]})


def test_qa_accuracy_longmemeval_style_uses_official_prompts_and_scores_abstain():
    LME_PACK.qa_style = "longmemeval"
    traces = _traces(LME_PACK, [
        ("Dana", [], {"reader_answer": "Dana"}),
        ("Dana", [], {"reader_answer": "I do not know"}),
    ])
    judge = YesNoJudge()
    assert qa_accuracy(LME_PACK, traces, judge) == 1.0
    assert "updated answer is the required answer" in judge.prompts[0]
    assert "Explanation: Only a name was given." in judge.prompts[1]


def test_qa_accuracy_default_style_leaves_abstain_queries_to_abstention():
    pack = parse_pack({"steps": [
        _w("a", "name", "My name is Dana."),
        _q("Cat?", answer="Only a name was given.", abstain=True),
    ]})
    traces = _traces(pack, [("Dana", [], {"reader_answer": "I don't know"})])
    assert qa_accuracy(pack, traces, AnswerJudge()) is None
    assert abstention(pack, traces) == 1.0
