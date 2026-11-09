"""QA accuracy: the reader's answer, judged against the gold `answer`.

For each query with a gold `answer` that the reader stage answered
(`detail.reader_answer`), the judge decides whether the reader's reply gives
the same answer. The fraction judged correct; a failed recall scores as
wrong, a reader or judge error is skipped. None without a reader or judge.
This is the number comparable to published LongMemEval / LoCoMo results.

With `pack.qa_style == "longmemeval"` the judge gets LongMemEval's official
per-question-type prompt (yes/no), and `abstain` queries are scored too, with
the official abstention prompt, as the paper's overall accuracy does.
Otherwise `abstain` queries are left to the `abstention` metric.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from membench.core.runner import StepTrace
from membench.core.scoring.judge import JudgeError, parse_yes_no, record_judge_error
from membench.packs.schema import Pack, QueryStep

if TYPE_CHECKING:
    from membench.core.scoring.judge import Judge, Verdict


def qa_rubric(question: str, gold: str) -> str:
    return (
        f"The EVIDENCE is an answer to the question {question!r}. It gives the same "
        f"answer as the reference answer {gold!r} (wording may differ)."
    )


def _verdict(pack: Pack, step: QueryStep, answer: str, judge: Judge) -> Verdict:
    if pack.qa_style == "longmemeval":
        from membench.datasets.longmemeval import judge_prompt

        prompt = judge_prompt(step.category, step.query, step.answer, answer, step.abstain)
        return judge.ask(prompt, parse_yes_no)
    return judge.score(qa_rubric(step.query, step.answer), answer)


def qa_accuracy(pack: Pack, traces: list[StepTrace], judge: Judge | None) -> float | None:
    if judge is None:
        return None
    results: list[bool] = []
    for step, trace in zip(pack.steps, traces):
        if not (isinstance(step, QueryStep) and step.answer is not None):
            continue
        if step.abstain and pack.qa_style != "longmemeval":
            continue
        if "reader_answer" not in trace.detail or "reader_error" in trace.detail:
            continue
        answer = trace.detail["reader_answer"]
        if trace.error is not None or answer is None:
            results.append(False)
            continue
        try:
            verdict = _verdict(pack, step, answer, judge)
        except JudgeError as exc:
            record_judge_error(trace.detail, "qa_accuracy", exc)
            continue
        results.append(verdict.passed)
    if not results:
        return None
    return sum(results) / len(results)
