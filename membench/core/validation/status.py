"""Which judge metrics a committed validation result has validated.

`judge-agreement` stores its result as `validation/agreement-v<prompt
version>.json`. A judge metric in a report is `validated` only if that file
exists for the report's judge prompt version and names the report's judge
model as validated on the metric; everything else is `unvalidated` (the safe
default, e.g. when no validation result is committed, or for an installed
wheel without the repo's `validation/` dir; set `MEMBENCH_VALIDATION_DIR`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_DIR = Path(__file__).resolve().parents[3] / "validation"


def validation_dir() -> Path:
    return Path(os.environ.get("MEMBENCH_VALIDATION_DIR") or DEFAULT_DIR)


def agreement_path(prompt_version: str, directory: Path | None = None) -> Path:
    return (directory or validation_dir()) / f"agreement-v{prompt_version}.json"


def metric_status(metric: str, judge_model: str | None, prompt_version: str | None,
                  directory: Path | None = None) -> str:
    """`validated` or `unvalidated` for one judge metric of one (model, prompt version)."""
    if not judge_model or not prompt_version:
        return "unvalidated"
    try:
        report = json.loads(agreement_path(prompt_version, directory).read_text())
    except (OSError, json.JSONDecodeError):
        return "unvalidated"
    if report.get("prompt_version") != prompt_version:
        return "unvalidated"
    for judge in report.get("judges", {}).values():
        entry = judge.get("metrics", {}).get(metric) or {}
        if judge.get("model") == judge_model and entry.get("status") == "validated":
            return "validated"
    return "unvalidated"


def report_metric_status(report: dict, metric: str) -> str:
    """Status of a judge metric in a run report (QA in LongMemEval style has its own prompt)."""
    manifest = report.get("manifest") or {}
    judge = manifest.get("judge") or {}
    if metric == "qa_accuracy" and (manifest.get("pack") or {}).get("qa_style") == "longmemeval":
        metric = "qa_accuracy:longmemeval"
    return metric_status(metric, judge.get("model"), judge.get("prompt_version"))
