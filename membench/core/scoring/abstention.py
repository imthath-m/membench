"""Abstention: the reader says "I don't know" when no answer exists.

Scored on `abstain: true` queries that the reader stage answered (`detail.
reader_answer`): the fraction whose answer abstains. Deterministic on the
stored answer, so `rescore` needs no network. A failed recall scores as a
miss; a reader error is skipped. None without a reader.
"""

from __future__ import annotations

import re

from membench.core.runner import StepTrace
from membench.packs.schema import Pack, QueryStep

_ABSTAINS = re.compile(
    r"\b(i (don'?t|do not) know|unknown|not (known|mentioned))\b", re.IGNORECASE
)


def is_abstention(answer: str) -> bool:
    return bool(_ABSTAINS.search(answer.replace("’", "'")))


def abstention(pack: Pack, traces: list[StepTrace]) -> float | None:
    results: list[bool] = []
    for step, trace in zip(pack.steps, traces):
        if not (isinstance(step, QueryStep) and step.abstain):
            continue
        if "reader_answer" not in trace.detail or "reader_error" in trace.detail:
            continue
        answer = trace.detail["reader_answer"]
        results.append(trace.error is None and answer is not None and is_abstention(answer))
    if not results:
        return None
    return sum(results) / len(results)
