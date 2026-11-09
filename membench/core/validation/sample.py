"""`judge-sample`: the (rubric, evidence) items the judge scores, and a
stratified sample of them for human labelling.

Items are collected by running each judge metric's own outcome function over
a report's traces with a recording judge, so an item is exactly what the
judge would see. Each item is stratified by (backend, verdict, context-length
bucket) within its metric; the verdict comes from the judge cache when the
run was judged, else it is null ("unjudged").
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from membench.core.runner import StepTrace
from membench.core.scoring import OUTCOMES
from membench.core.scoring.judge import (
    JUDGE_PROMPT_VERSION,
    JudgeCache,
    JudgeError,
    Verdict,
    build_prompt,
    parse_verdict,
    parse_yes_no,
)
from membench.core.scoring.qa_accuracy import qa_rubric
from membench.core.tokens import count_tokens
from membench.core.validation import JUDGE_METRICS
from membench.packs.schema import Pack

# Upper bounds (whitespace tokens of the judged evidence) and bucket names.
LENGTH_BUCKETS = ((128, "<128"), (512, "128-511"), (2048, "512-2047"), (None, "2048+"))
DEFAULT_N_PER_METRIC = 200


def length_bucket(tokens: int) -> str:
    return next(name for bound, name in LENGTH_BUCKETS if bound is None or tokens < bound)


def item_prompt(item: dict) -> tuple[str, Callable[[str], Verdict]]:
    """The exact prompt a judge gets for an item, and its reply parser."""
    if item.get("prompt"):  # an official template (LongMemEval), yes/no reply
        return item["prompt"], parse_yes_no
    return build_prompt(item["rubric"], item["evidence"]), parse_verdict


class _Recorder:
    """A judge that records each call and declines it (JudgeError)."""

    model = "recorder"

    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def score(self, rubric: str, evidence: str) -> Verdict:
        self.calls.append({"rubric": rubric, "evidence": evidence})
        raise JudgeError("sampled")

    def ask(self, prompt: str, parse: Callable[[str], Verdict]) -> Verdict:
        self.calls.append({"prompt": prompt})
        raise JudgeError("sampled")


def judge_items(report: dict, pack: Pack, cache: JudgeCache | None = None,
                judge_model: str | None = None) -> list[dict[str, Any]]:
    """Every judge item of a report's first run, with its cached verdict if any."""
    traces = [StepTrace(**copy.deepcopy(t)) for t in report["traces"]]
    steps = {t.step_index: s for s, t in zip(pack.steps, traces)}
    backend = report["backend_name"]
    items = []
    for metric in JUDGE_METRICS:
        recorder = _Recorder()
        OUTCOMES[metric](pack, traces, recorder)
        asked = [t for t in traces if metric in t.detail.get("judge_error", {})]
        for trace, call in zip(asked, recorder.calls, strict=True):
            step = steps[trace.step_index]
            name, item = metric, {"rubric": call.get("rubric"), "evidence": call.get("evidence")}
            if "prompt" in call:  # LongMemEval-style QA: show humans the plain rubric
                name = f"{metric}:longmemeval"
                rubric = (f"The EVIDENCE says the question {step.query!r} cannot be answered."
                          if step.abstain else qa_rubric(step.query, step.answer))
                item = {"rubric": rubric, "evidence": trace.detail["reader_answer"],
                        "prompt": call["prompt"]}
            tokens = count_tokens(item["evidence"])
            item = {
                "id": _item_id(name, item), "metric": name, "backend": backend,
                "pack": report["pack_name"], "step_index": trace.step_index,
                "category": step.category, **item, "tokens": tokens,
                "length_bucket": length_bucket(tokens), "prompt_version": JUDGE_PROMPT_VERSION,
                "judge_model": judge_model, "verdict": None,
            }
            if cache is not None and judge_model:
                entry = cache.get(JudgeCache.key(judge_model, item_prompt(item)[0]))
                if entry is not None and "passed" in entry:
                    item["verdict"] = entry["passed"]
            items.append({k: v for k, v in item.items() if v is not None or k == "verdict"})
    return items


def _item_id(metric: str, item: dict) -> str:
    """Stable across prompt versions: humans label the statement, not the prompt."""
    text = json.dumps([metric, item["rubric"], item["evidence"], item.get("prompt", "")])
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def stratum(item: dict) -> tuple[str, str, str]:
    verdict = {True: "pass", False: "fail", None: "unjudged"}[item.get("verdict")]
    return item["backend"], verdict, item["length_bucket"]


def stratified_sample(items: list[dict], n_per_metric: int = DEFAULT_N_PER_METRIC,
                      seed: int = 0) -> list[dict]:
    """Up to `n_per_metric` distinct items per metric, drawn round-robin over the
    (backend, verdict, length bucket) strata so small strata are fully covered."""
    rng = random.Random(f"judge-sample:{seed}")
    unique = {}
    for item in items:
        unique.setdefault(item["id"], item)
    by_metric: dict[str, dict[tuple, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for item in unique.values():
        by_metric[item["metric"]][stratum(item)].append(item)
    sample = []
    for metric in sorted(by_metric):
        strata = [by_metric[metric][key] for key in sorted(by_metric[metric])]
        for group in strata:
            rng.shuffle(group)
        picked: list[dict] = []
        while len(picked) < n_per_metric and any(strata):
            for group in strata:
                if group and len(picked) < n_per_metric:
                    picked.append(group.pop())
        sample += sorted(picked, key=lambda i: i["id"])  # hash order: hides the strata
    return sample


def sample_summary(sample: list[dict]) -> dict[str, dict[str, int]]:
    """{metric: {"backend/verdict/bucket": count}} for the CLI and docs."""
    summary: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item in sample:
        summary[item["metric"]]["/".join(stratum(item))] += 1
    return {m: dict(sorted(c.items())) for m, c in sorted(summary.items())}
