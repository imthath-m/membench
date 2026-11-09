"""`validation/labels.jsonl`: human labels for judge items.

One JSON object per line, one line per labelled item:

    {"item_id": "3f2a…", "metric": "semantic_recall", "backend": "chroma",
     "length_bucket": "<128", "prompt_version": "1", "judge_model": "claude-sonnet-5",
     "labels": {"ann1": true, "ann2": false}, "adjudicated": true, "notes": "…"}

A label is the human's answer to the judge's own question: `true` = the
evidence clearly supports the rubric statement (for `contradiction_rate`:
the contradiction is present), exactly like the judge's `passed`. Every line
of a metric has the same two or more annotators; `adjudicated` settles a
disagreement (null while pending). Text lives in the sample, not here.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from membench.core.validation import JUDGE_METRICS

_REQUIRED = {"item_id": str, "metric": str, "backend": str, "length_bucket": str,
             "prompt_version": str, "labels": dict}
_METRICS = {*JUDGE_METRICS, "qa_accuracy:longmemeval"}


class LabelsError(ValueError):
    """A labels file (or annotator file) is malformed."""


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LabelsError(f"{path}:{n}: not JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise LabelsError(f"{path}:{n}: not a JSON object")
        rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def gold(line: dict) -> bool | None:
    """Adjudicated label, else the majority of annotators; None on an unsettled tie."""
    if line.get("adjudicated") is not None:
        return line["adjudicated"]
    votes = Counter(line["labels"].values())
    if votes[True] == votes[False]:
        return None
    return votes[True] > votes[False]


def validate_labels(lines: list[dict], source: str = "labels") -> list[dict]:
    """Check the format; raise `LabelsError` naming the first bad line."""
    seen: set[str] = set()
    annotators: dict[str, set[str]] = {}
    for n, line in enumerate(lines, 1):
        where = f"{source}:{n}"
        for key, kind in _REQUIRED.items():
            if not isinstance(line.get(key), kind):
                raise LabelsError(f"{where}: `{key}` must be a {kind.__name__}")
        if line["metric"] not in _METRICS:
            raise LabelsError(f"{where}: unknown metric {line['metric']!r}")
        if line["item_id"] in seen:
            raise LabelsError(f"{where}: duplicate item_id {line['item_id']!r}")
        seen.add(line["item_id"])
        labels = line["labels"]
        if len(labels) < 2 or not all(isinstance(v, bool) for v in labels.values()):
            raise LabelsError(f"{where}: `labels` needs >= 2 annotators with true/false")
        if line.get("adjudicated") not in (None, True, False):
            raise LabelsError(f"{where}: `adjudicated` must be true, false or null")
        expected = annotators.setdefault(line["metric"], set(labels))
        if set(labels) != expected:
            raise LabelsError(f"{where}: annotators {sorted(labels)} differ from this "
                              f"metric's {sorted(expected)}")
    return lines


def load_labels(path: str | Path) -> list[dict]:
    return validate_labels(read_jsonl(path), str(path))


def merge_annotations(sample: list[dict], annotations: dict[str, list[dict]],
                      adjudication: list[dict] | None = None) -> list[dict]:
    """Build labels lines from per-annotator files of `{"item_id", "label", "note"?}`
    (and an optional adjudication file of the same shape). Only items every
    annotator labelled are kept."""
    items = {item["id"]: item for item in sample}
    by_annotator: dict[str, dict[str, dict]] = {}
    for name, rows in annotations.items():
        by_annotator[name] = {}
        for row in rows:
            if row.get("item_id") not in items or not isinstance(row.get("label"), bool):
                raise LabelsError(f"annotator {name!r}: bad row {row!r} (unknown item_id or "
                                  "non-boolean label)")
            by_annotator[name][row["item_id"]] = row
    settled = {}
    for row in adjudication or []:
        if row.get("item_id") not in items or not isinstance(row.get("label"), bool):
            raise LabelsError(f"adjudication: bad row {row!r}")
        settled[row["item_id"]] = row["label"]
    lines = []
    for item_id, item in items.items():
        if not all(item_id in rows for rows in by_annotator.values()):
            continue
        notes = [f"{n}: {rows[item_id]['note']}" for n, rows in by_annotator.items()
                 if rows[item_id].get("note")]
        line = {
            "item_id": item_id, "metric": item["metric"], "backend": item["backend"],
            "length_bucket": item["length_bucket"], "prompt_version": item["prompt_version"],
            "judge_model": item.get("judge_model"),
            "labels": {n: rows[item_id]["label"] for n, rows in by_annotator.items()},
            "adjudicated": settled.get(item_id),
        }
        if notes:
            line["notes"] = "; ".join(notes)
        lines.append(line)
    return validate_labels(lines, "merged labels")
