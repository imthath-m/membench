"""`judge-tune`: measure judge prompt versions against human labels.

Each prompt version judges every labelled item (one shared, cached judge per
version); items are split by metric into a dev and a held-out set, and the
agreement with the human gold is reported per version, split, metric and
context-length bucket. Tune on dev; judge the chosen candidate on held-out once.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from membench.core.scoring.judge import (
    DEFAULT_JUDGE_MODEL,
    CachedJudge,
    CacheMissError,
    JudgeError,
    build_prompt,
    resolve_client,
)
from membench.core.validation.agreement import binary_agreement
from membench.core.validation.cross import share_cache
from membench.core.validation.labels import gold
from membench.core.validation.sample import LENGTH_BUCKETS, item_prompt

SPLITS = ("dev", "held_out", "all")
MIN_SPLIT_N = 50
_DELTA_KEYS = ("kappa", "accuracy", "false_pass_rate", "false_fail_rate")
_REMINDER = ("Choose the candidate on dev only; judge it on held-out once — re-tuning after "
             "looking at held-out numbers turns held-out into dev.")


def resolved(labels: Iterable[dict]) -> list[dict]:
    """Labels lines with a settled gold (adjudicated or a clear majority)."""
    return [line for line in labels if gold(line) is not None]


def split_items(labels: list[dict], *, seed: int = 0,
                held_out_fraction: float = 0.5) -> dict[str, str]:
    """{item_id: "dev" | "held_out"}, stratified by metric and deterministic: it
    depends only on the seed, the fraction and each metric's set of resolved ids."""
    if not 0 <= held_out_fraction <= 1:
        raise ValueError(f"held-out fraction must be in [0, 1], got {held_out_fraction}")
    by_metric: dict[str, set[str]] = {}
    for line in resolved(labels):
        by_metric.setdefault(line["metric"], set()).add(line["item_id"])
    split = {}
    for metric, ids in by_metric.items():
        order = sorted(ids)
        random.Random(f"{seed}:{metric}").shuffle(order)
        n_held = round(len(order) * held_out_fraction)
        for i, item_id in enumerate(order):
            split[item_id] = "held_out" if i < n_held else "dev"
    return split


def join_sample(labels: list[dict], sample: list[dict]) -> dict[str, dict]:
    """{item_id: sample item} for every labelled item; labels carry no rubric/evidence."""
    items = {item["id"]: item for item in sample}
    missing = sorted(line["item_id"] for line in labels if line["item_id"] not in items)
    if missing:
        raise ValueError(f"{len(missing)} labelled items are not in the sample (first: "
                         f"{', '.join(missing[:5])})")
    return {line["item_id"]: items[line["item_id"]] for line in labels}


def build_version_judges(versions: list[str], cache_path: str | Path, *, strict: bool,
                         model: str | None = None, base_url: str | None = None,
                         api_key: str | None = None) -> dict[str, CachedJudge]:
    """One cached judge per prompt version, all sharing one in-memory cache (the cache
    file is rewritten whole on every write, so separate caches would clobber it)."""
    model = model or DEFAULT_JUDGE_MODEL
    client = None if strict else resolve_client(model=model, base_url=base_url,
                                                api_key=api_key)
    if client is None:
        strict = True
    judges = {v: CachedJudge(client, cache_path, strict=strict, model=model, base_url=base_url,
                             prompt_version=v) for v in versions}
    share_cache(judges, cache_path)
    return judges


def run_versions(items: dict[str, dict], judges: dict[str, CachedJudge]
                 ) -> tuple[dict[str, dict[str, dict]], dict[str, list[str]]]:
    """Judge every item with every version: ({version: {item_id: {"passed"} |
    {"error"}}}, {version: [item ids with no cached verdict]}). Never stops midway."""
    verdicts: dict[str, dict[str, dict]] = {}
    misses: dict[str, list[str]] = {}
    for version, judge in judges.items():
        verdicts[version], misses[version] = {}, []
        for item_id in sorted(items):
            try:
                passed = judge.ask(*item_prompt(items[item_id], version)).passed
                verdicts[version][item_id] = {"passed": passed}
            except CacheMissError:
                misses[version].append(item_id)
                verdicts[version][item_id] = {"error": "no cached verdict"}
            except JudgeError as exc:
                verdicts[version][item_id] = {"error": str(exc)}
    return verdicts, {v: ids for v, ids in misses.items() if ids}


def _stats(rows: list[tuple[dict, bool, str]]) -> dict[str, Any]:
    """Agreement of (verdict, gold, bucket) rows; errors are counted, not scored."""
    out = _flat_stats(rows)
    out["by_length"] = {}
    for _, bucket in LENGTH_BUCKETS:
        stats = _flat_stats([r for r in rows if r[2] == bucket])
        if stats["n"]:  # empty buckets are omitted
            out["by_length"][bucket] = stats
    return out


def _flat_stats(rows: list[tuple[dict, bool, str]]) -> dict[str, Any]:
    ok = [(v["passed"], g) for v, g, _ in rows if "passed" in v]
    return {**binary_agreement([j for j, _ in ok], [g for _, g in ok]),
            "errors": len(rows) - len(ok)}


def _delta(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else round(b - a, 4)


def prompt_sha256(version: str) -> str:
    return hashlib.sha256(build_prompt("{rubric}", "{evidence}", version).encode()).hexdigest()


def tuning_report(labels: list[dict], items: dict[str, dict],
                  verdicts: dict[str, dict[str, dict]], *, seed: int = 0,
                  held_out_fraction: float = 0.5, judge_model: str | None = None,
                  base_url: str | None = None,
                  usage: dict[str, dict] | None = None) -> dict[str, Any]:
    """Agreement with the human gold per version × split × metric (and length bucket)."""
    versions = list(verdicts)
    lines = resolved(labels)
    split = split_items(labels, seed=seed, held_out_fraction=held_out_fraction)
    metrics = sorted({line["metric"] for line in lines})
    bucket = {line["item_id"]: items.get(line["item_id"], {}).get("length_bucket",
                                                                    line["length_bucket"])
              for line in lines}
    split_sizes = {m: {s: sum(1 for x in lines if x["metric"] == m and split[x["item_id"]] == s)
                       for s in ("dev", "held_out")} for m in metrics}
    prompt_varies = {m: any(not items.get(x["item_id"], {}).get("prompt")
                            for x in lines if x["metric"] == m) for m in metrics}
    results: dict[str, dict] = {}
    disagreements: dict[str, list[str]] = {}
    for version in versions:
        results[version] = {}
        for s in SPLITS:
            chosen = [x for x in lines if s == "all" or split[x["item_id"]] == s]
            rows = {m: [(verdicts[version].get(x["item_id"], {"error": "not judged"}),
                         gold(x), bucket[x["item_id"]]) for x in chosen if x["metric"] == m]
                    for m in metrics}
            block = {m: _stats(r) for m, r in rows.items()}
            block["overall"] = _stats([r for m in metrics for r in rows[m]])
            results[version][s] = block
        disagreements[version] = sorted(
            x["item_id"] for x in lines if split[x["item_id"]] == "dev"
            and "passed" in (v := verdicts[version].get(x["item_id"], {}))
            and v["passed"] != gold(x))
    baseline, candidate = versions[0], versions[-1]
    deltas = {s: {m: {k: _delta(results[baseline][s][m][k], results[candidate][s][m][k])
                      for k in _DELTA_KEYS} for m in [*metrics, "overall"]} for s in SPLITS}
    warnings = []
    for m in metrics:
        for s, n in split_sizes[m].items():
            if n < MIN_SPLIT_N:
                warnings.append(f"{m}: only {n} {s} items (< {MIN_SPLIT_N}); differences "
                                "between versions are likely noise")
    present = set(bucket.values())
    empty = [name for _, name in LENGTH_BUCKETS if name not in present]
    if empty:
        warnings.append(f"no labelled items in length buckets {', '.join(empty)}: "
                        "no evidence about long contexts there")
    for m in metrics:
        if not prompt_varies[m]:
            warnings.append(f"{m}: prompt_varies is false (fixed official template); "
                            "identical numbers across versions are expected")
    for version in versions:
        errors = results[version]["all"]["overall"]["errors"]
        if errors:
            warnings.append(f"v{version}: {errors} judge errors excluded from agreement")
    warnings.append(_REMINDER)
    return {
        "schema": 1, "seed": seed, "held_out_fraction": held_out_fraction,
        "judge_model": judge_model, "base_url": base_url, "versions": versions,
        "baseline": baseline, "candidate": candidate, "n_items": len(lines),
        "n_unresolved": len(labels) - len(lines), "split_sizes": split_sizes,
        "prompt_varies": prompt_varies, "results": results, "deltas": deltas,
        "disagreements": disagreements,
        "prompt_sha256": {v: prompt_sha256(v) for v in versions},
        "usage": usage or {}, "warnings": warnings,
        "notes": [
            ("Split is stratified by metric and seeded; it does not depend on which "
             "versions are compared."),
            "Errors are excluded from agreement (never counted as pass or fail).",
            "Disagreements list dev items only; held-out items stay unseen.",
            f"{len(labels) - len(lines)} labelled items without a settled gold are excluded.",
        ],
    }


def _fmt(x: object) -> str:
    return "n/a" if x is None else f"{x:.2f}" if isinstance(x, float) else str(x)


def _row(*cells: object) -> str:
    return "| " + " | ".join(_fmt(c) for c in cells) + " |"


def render_tuning_markdown(report: dict[str, Any]) -> str:
    stat_cols = "n | κ | Accuracy | False pass | False fail | Errors |"
    lines = [f"# Judge prompt tuning — v{report['baseline']} vs v{report['candidate']}", "",
             (f"Judge `{report['judge_model']}`; versions {', '.join(report['versions'])}; "
              f"seed {report['seed']}; held-out fraction {report['held_out_fraction']}; "
              f"{report['n_items']} items ({report['n_unresolved']} unresolved excluded)."),
             ""]
    by_length = []
    for s in SPLITS:
        lines += [f"## {s}", "", f"| Version | Metric | {stat_cols}",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for version in report["versions"]:
            for metric, m in report["results"][version][s].items():
                lines.append(_row(version, metric, m["n"], m["kappa"], m["accuracy"],
                                  m["false_pass_rate"], m["false_fail_rate"], m["errors"]))
                for b, st in m["by_length"].items():
                    by_length.append(_row(s, version, metric, b, st["n"], st["kappa"],
                                          st["accuracy"], st["false_pass_rate"],
                                          st["false_fail_rate"], st["errors"]))
        lines.append("")
    lines += ["## By context length", "", f"| Split | Version | Metric | Bucket | {stat_cols}",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |", *by_length, "",
              f"## Held-out deltas (v{report['candidate']} − v{report['baseline']})", "",
              "| Metric | Δκ | ΔAccuracy | ΔFalse pass | ΔFalse fail |",
              "| --- | --- | --- | --- | --- |"]
    for metric, d in report["deltas"]["held_out"].items():
        lines.append(_row(metric, *(d[k] for k in _DELTA_KEYS)))
    lines += ["", "## Warnings", "", *(f"- {w}" for w in report["warnings"]), "",
              "## How to read", "",
              ("- Agreement is with the human gold (adjudicated, else majority); n/a = "
               "undefined (no items, or a constant rater)."),
              "- Compare versions on dev; the held-out deltas are the one-shot confirmation.",
              "- False pass = judge passed where humans failed; false fail = the reverse.", ""]
    return "\n".join(lines)
