"""LongMemEval (Wu et al., ICLR 2025): 500 questions, each with its own haystack.

Source: the `xiaowu0162/longmemeval-cleaned` Hugging Face dataset (MIT per its
card and the upstream repo), pinned to one revision and sha256-checked.
Variants: `s` (~115k-token haystacks), `m` (~500 sessions per question; the
file is 2.7 GB, so loading it needs a machine with lots of RAM), `oracle`
(only the evidence sessions; for reader calibration).

Each question becomes one episode: every haystack turn is a `write` (sessions
in date order, turn `i` at the session date + `i` seconds), then the question
is a `query` at `question_date`. Event and session ids are opaque indexes:
upstream session ids start with `answer_` on evidence sessions, and the
`has_answer` turn flags are ground truth, so neither reaches an adapter.
`evidence` = the `has_answer` turns of the answer sessions; `question_type`
→ `category`; `_abs` questions → `abstain` (their `answer` is the upstream
explanation of why it is unanswerable, used by the official judge prompt).

`reader_prompt` / `judge_prompt` are the official templates
(`src/generation/run_generation.py`, full-history, no chain-of-thought;
`src/evaluation/evaluate_qa.py`) from github.com/xiaowu0162/LongMemEval at
commit 9e0b455f4ef0e2ab8f2e582289761153549043fc; the pack sets
`qa_style: longmemeval` so the reader and QA judge use them.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from membench.datasets.fetch import Source

_BASE = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve"
_REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
SOURCES = {
    "s": Source(
        "longmemeval-s", f"{_BASE}/{_REVISION}/longmemeval_s_cleaned.json", _REVISION,
        "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442", "MIT",
    ),
    "m": Source(
        "longmemeval-m", f"{_BASE}/{_REVISION}/longmemeval_m_cleaned.json", _REVISION,
        "9d79e5524794a2e6900a3aa9cb7d9152c5a3e8319c9a87c25494ba1eacee495f", "MIT",
    ),
    "oracle": Source(
        "longmemeval-oracle", f"{_BASE}/{_REVISION}/longmemeval_oracle.json", _REVISION,
        "821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c", "MIT",
    ),
}
_DATE_FORMAT = "%Y/%m/%d (%a) %H:%M"


def _parse_date(text: str) -> datetime:
    """'2023/04/10 (Mon) 17:50' -> aware UTC datetime (the weekday is dropped)."""
    return datetime.strptime(re.sub(r" \(\w+\)", "", text), "%Y/%m/%d %H:%M").replace(
        tzinfo=timezone.utc
    )


def build(records: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    """Raw pack mapping (as `parse_pack` takes) from the upstream JSON records."""
    episodes = []
    for q, record in enumerate(records):
        sessions = sorted(
            zip(record["haystack_dates"], record["haystack_session_ids"],
                record["haystack_sessions"]),
            key=lambda session: _parse_date(session[0]),
        )
        answer_sessions = set(record["answer_session_ids"])
        steps: list[dict[str, Any]] = []
        evidence: list[str] = []
        for s, (date, session_id, turns) in enumerate(sessions):
            start = _parse_date(date)
            for t, turn in enumerate(turns):
                event_id = f"q{q}-s{s}-t{t}"
                steps.append({
                    "type": "write", "session_id": f"q{q}-s{s}", "id": event_id,
                    "fact_key": event_id, "role": turn["role"], "content": turn["content"],
                    "timestamp": (start + timedelta(seconds=t)).isoformat(),
                })
                if session_id in answer_sessions and turn.get("has_answer"):
                    evidence.append(event_id)
        abstain = record["question_id"].endswith("_abs")
        steps.append({
            "type": "query", "session_id": f"q{q}-question", "query": record["question"],
            "timestamp": _parse_date(record["question_date"]).isoformat(),
            "category": record["question_type"], "answer": str(record["answer"]),
            "evidence": [] if abstain else evidence, "abstain": abstain,
        })
        episodes.append({"steps": steps})
    return {
        "name": f"longmemeval_{variant}",
        "description": f"LongMemEval ({variant}), one episode per question.",
        "episodes": episodes,
    }


_READER = (
    "I will give you several history chats between you and a user. Please answer the "
    "question based on the relevant chat history.\n\n\nHistory Chats:\n\n{}\n\n"
    "Current Date: {}\nQuestion: {}\nAnswer:"
)
_JUDGE_PREFIX = (
    "I will give you a question, a correct answer, and a response from a model. Please "
    "answer yes if the response contains the correct answer. Otherwise, answer no. "
)
_JUDGE_STEPS = (
    "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no. "
)
_JUDGE_SUFFIX = (
    "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)
_JUDGE = {
    "default": _JUDGE_PREFIX + _JUDGE_STEPS + _JUDGE_SUFFIX,
    "temporal-reasoning": _JUDGE_PREFIX + _JUDGE_STEPS + (
        "In addition, do not penalize off-by-one errors for the number of days. If the "
        "question asks for the number of days/weeks/months, etc., and the model makes "
        "off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's "
        "response is still correct. "
    ) + _JUDGE_SUFFIX,
    "knowledge-update": _JUDGE_PREFIX + (
        "If the response contains some previous information along with an updated answer, "
        "the response should be considered as correct as long as the updated answer is the "
        "required answer."
    ) + _JUDGE_SUFFIX,
    "single-session-preference": (
        "I will give you a question, a rubric for desired personalized response, and a "
        "response from a model. Please answer yes if the response satisfies the desired "
        "response. Otherwise, answer no. The model does not need to reflect all the points "
        "in the rubric. The response is correct as long as it recalls and utilizes the "
        "user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\n"
        "Model Response: {}\n\nIs the model response correct? Answer yes or no only."
    ),
    "abstention": (
        "I will give you an unanswerable question, an explanation, and a response from a "
        "model. Please answer yes if the model correctly identifies the question as "
        "unanswerable. The model could say that the information is incomplete, or some "
        "other information is given but the asked information is not.\n\nQuestion: {}\n\n"
        "Explanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the "
        "question as unanswerable? Answer yes or no only."
    ),
}


def reader_prompt(question: str, context: str, at: str) -> str:
    """Official reader prompt; `at` (ISO) is shown in LongMemEval's date format."""
    return _READER.format(context, datetime.fromisoformat(at).strftime(_DATE_FORMAT), question)


def judge_prompt(
    category: str | None, question: str, answer: str | None, response: str, abstain: bool
) -> str:
    """Official answer-check prompt for this question type (yes/no reply)."""
    template = _JUDGE["abstention" if abstain else category or "default"] if (
        abstain or category in _JUDGE
    ) else _JUDGE["default"]
    return template.format(question, answer, response)
