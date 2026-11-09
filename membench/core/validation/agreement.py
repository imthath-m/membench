"""`judge-agreement`: does the judge agree with humans, and with other judges?

Per metric:
  - human–human: Cohen's κ between the two annotators (Light's κ, the mean
    pairwise Cohen's κ, with more than two) and raw agreement;
  - each judge vs the majority-human (adjudicated) label: κ, accuracy, and
    false-pass / false-fail rates, overall and per context-length bucket;
  - judge vs judge: pairwise κ and the items they disagree on.

A judge is `validated` on a metric when human–human κ ≥ 0.7, its own κ ≥ 0.6
and no more than 0.1 below human–human, on ≥ 200 settled items with no
disagreement left unadjudicated. Anything else is `unvalidated`.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any

from membench.core.scoring.judge import JUDGE_PROMPT_VERSION
from membench.core.validation.labels import gold
from membench.core.validation.sample import LENGTH_BUCKETS

MIN_HUMAN_KAPPA = 0.7
MIN_JUDGE_KAPPA = 0.6
MAX_KAPPA_GAP = 0.1
MIN_VALIDATION_N = 200
THRESHOLDS = {"min_human_kappa": MIN_HUMAN_KAPPA, "min_judge_kappa": MIN_JUDGE_KAPPA,
              "max_kappa_gap": MAX_KAPPA_GAP, "min_n": MIN_VALIDATION_N}


def cohen_kappa(a: list[bool], b: list[bool]) -> float | None:
    """Cohen's κ of two raters' binary labels; None when undefined (no items,
    or both raters gave one and the same label to every item)."""
    n = len(a)
    if n == 0 or n != len(b):
        return None
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected == 1:
        return None
    return (observed - expected) / (1 - expected)


def binary_agreement(judge: list[bool], human: list[bool]) -> dict[str, Any]:
    """κ, accuracy and false-pass / false-fail rates of a judge vs human labels.

    false_pass_rate = judge passed / human said fail; false_fail_rate = judge
    failed / human said pass (None when the human class is empty).
    """
    n = len(human)
    neg, pos = human.count(False), human.count(True)
    false_pass = sum(j and not h for j, h in zip(judge, human))
    false_fail = sum(h and not j for j, h in zip(judge, human))
    return {
        "n": n,
        "kappa": _round(cohen_kappa(judge, human)),
        "accuracy": _round((n - false_pass - false_fail) / n) if n else None,
        "false_pass": false_pass,
        "false_pass_rate": _round(false_pass / neg) if neg else None,
        "false_fail": false_fail,
        "false_fail_rate": _round(false_fail / pos) if pos else None,
    }


def _round(x: float | None) -> float | None:
    return None if x is None else round(x, 4)


def human_agreement(lines: list[dict]) -> dict[str, Any]:
    """Human–human agreement of one metric's labels lines."""
    names = sorted(lines[0]["labels"]) if lines else []
    kappas, agreements = [], []
    for x, y in combinations(names, 2):
        a = [line["labels"][x] for line in lines]
        b = [line["labels"][y] for line in lines]
        kappas.append(cohen_kappa(a, b))
        agreements.append(sum(p == q for p, q in zip(a, b)) / len(lines))
    kappa = None if not kappas or None in kappas else sum(kappas) / len(kappas)
    return {
        "n": len(lines), "annotators": names, "kappa": _round(kappa),
        "agreement": _round(sum(agreements) / len(agreements)) if agreements else None,
        "unresolved": sum(gold(line) is None for line in lines),
    }


def _status(human: dict, judge: dict) -> tuple[str, list[str]]:
    reasons = []
    hk, jk = human.get("kappa"), judge["kappa"]
    if hk is None or hk < MIN_HUMAN_KAPPA:
        reasons.append(f"human-human kappa {hk} < {MIN_HUMAN_KAPPA}")
    if jk is None or jk < MIN_JUDGE_KAPPA:
        reasons.append(f"judge-human kappa {jk} < {MIN_JUDGE_KAPPA}")
    elif hk is not None and jk < hk - MAX_KAPPA_GAP:
        reasons.append(f"judge-human kappa {jk} is more than {MAX_KAPPA_GAP} below "
                       f"human-human {hk}")
    if judge["n"] < MIN_VALIDATION_N:
        reasons.append(f"n {judge['n']} < {MIN_VALIDATION_N}")
    if human.get("unresolved"):
        reasons.append(f"{human['unresolved']} disagreements not adjudicated")
    return ("unvalidated" if reasons else "validated"), reasons


def agreement_report(labels: list[dict] | None = None,
                     verdicts: list[dict] | None = None) -> dict[str, Any]:
    """The agreement report (JSON-ready) from labels lines and/or judge verdict rows."""
    labels, verdicts = labels or [], verdicts or []
    versions = {v["prompt_version"] for v in verdicts}
    if len(versions) > 1:
        raise ValueError(f"verdicts mix judge prompt versions {sorted(versions)}")
    prompt_version = versions.pop() if versions else JUDGE_PROMPT_VERSION

    by_metric: dict[str, list[dict]] = defaultdict(list)
    for line in labels:
        by_metric[line["metric"]].append(line)
    human = {m: human_agreement(lines) for m, lines in sorted(by_metric.items())}
    settled = {line["item_id"]: (gold(line), line) for line in labels
               if gold(line) is not None}

    # judge name -> {item_id: passed}; verdict rows with an error are counted, not scored
    passed: dict[str, dict[str, bool]] = defaultdict(dict)
    judges: dict[str, dict[str, Any]] = {}
    metric_of: dict[str, str] = {}
    errors: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for v in verdicts:
        judges.setdefault(v["judge"], {"model": v["model"], "base_url": v.get("base_url"),
                                       "metrics": {}})
        metric_of[v["item_id"]] = v["metric"]
        if v.get("passed") is None:
            errors[v["judge"]][v["metric"]] += 1
        else:
            passed[v["judge"]][v["item_id"]] = v["passed"]

    for name, info in judges.items():
        metrics = sorted({metric_of[i] for i in passed[name]} | set(errors[name]))
        for metric in metrics:
            ids = [i for i in passed[name] if i in settled and metric_of[i] == metric]
            entry = binary_agreement([passed[name][i] for i in ids], [settled[i][0] for i in ids])
            entry["errors"] = errors[name][metric]
            entry["by_length"] = {}
            for _, bucket in LENGTH_BUCKETS:
                in_bucket = [i for i in ids if settled[i][1]["length_bucket"] == bucket]
                if in_bucket:
                    entry["by_length"][bucket] = binary_agreement(
                        [passed[name][i] for i in in_bucket], [settled[i][0] for i in in_bucket])
            entry["status"], entry["reasons"] = _status(human.get(metric, {}), entry)
            info["metrics"][metric] = entry

    pairwise: dict[str, list[dict]] = defaultdict(list)
    for x, y in combinations(sorted(judges), 2):
        shared = sorted(set(passed[x]) & set(passed[y]))
        for metric in sorted({metric_of[i] for i in shared}):
            ids = [i for i in shared if metric_of[i] == metric]
            a, b = [passed[x][i] for i in ids], [passed[y][i] for i in ids]
            pairwise[metric].append({
                "a": x, "b": y, "n": len(ids), "kappa": _round(cohen_kappa(a, b)),
                "agreement": _round(sum(p == q for p, q in zip(a, b)) / len(ids)),
                "disagreements": [i for i, p, q in zip(ids, a, b) if p != q],
            })
    return {"prompt_version": prompt_version, "thresholds": THRESHOLDS, "human": human,
            "judges": judges, "pairwise": dict(pairwise)}


def _fmt(x: object) -> str:
    return "n/a" if x is None else f"{x:.2f}" if isinstance(x, float) else str(x)


def render_agreement_markdown(report: dict[str, Any]) -> str:
    lines = [f"# Judge agreement — prompt v{report['prompt_version']}", "",
             "## Human–human", "", "| Metric | n | Annotators | κ | Agreement | Unadjudicated |",
             "| --- | --- | --- | --- | --- | --- |"]
    for metric, h in report["human"].items():
        lines.append(f"| {metric} | {h['n']} | {', '.join(h['annotators'])} | "
                     f"{_fmt(h['kappa'])} | {_fmt(h['agreement'])} | {h['unresolved']} |")
    lines += ["", "## Judge vs majority human", "",
              ("| Judge | Model | Metric | n | κ | Accuracy | False pass | False fail | Errors "
               "| Status |"), "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    buckets = []
    for name, info in sorted(report["judges"].items()):
        for metric, m in info["metrics"].items():
            lines.append(
                f"| {name} | {info['model']} | {metric} | {m['n']} | {_fmt(m['kappa'])} | "
                f"{_fmt(m['accuracy'])} | {_fmt(m['false_pass_rate'])} | "
                f"{_fmt(m['false_fail_rate'])} | {m['errors']} | {m['status']} |")
            for bucket, b in m["by_length"].items():
                buckets.append(f"| {name} | {metric} | {bucket} | {b['n']} | {_fmt(b['kappa'])} "
                               f"| {_fmt(b['accuracy'])} | {_fmt(b['false_pass_rate'])} | "
                               f"{_fmt(b['false_fail_rate'])} |")
    lines += ["", "## By context length (whitespace tokens of the judged evidence)", "",
              "| Judge | Metric | Bucket | n | κ | Accuracy | False pass | False fail |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |", *buckets,
              "", "## Judge vs judge", "",
              "| Metric | A | B | n | κ | Agreement | Disagreeing items |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for metric, pairs in report["pairwise"].items():
        for p in pairs:
            ids = ", ".join(p["disagreements"][:10]) + (" …" if len(p["disagreements"]) > 10
                                                        else "")
            lines.append(f"| {metric} | {p['a']} | {p['b']} | {p['n']} | {_fmt(p['kappa'])} | "
                         f"{_fmt(p['agreement'])} | {ids} |")
    t = report["thresholds"]
    lines += ["", (f"Validated: human–human κ ≥ {t['min_human_kappa']}, judge κ ≥ "
                   f"{t['min_judge_kappa']} and within {t['max_kappa_gap']} of human–human, "
                   f"n ≥ {t['min_n']}, every disagreement adjudicated."), ""]
    return "\n".join(lines)
