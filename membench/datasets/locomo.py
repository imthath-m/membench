"""LoCoMo (Maharana et al., ACL 2024): 10 long two-person conversations.

Source: `data/locomo10.json` from github.com/snap-research/locomo, pinned to
one commit and sha256-checked. License: CC BY-NC 4.0 (non-commercial): never
redistribute the raw file or a converted pack (`membench convert` output is
for local inspection only).

Each conversation becomes one episode: every dialogue turn is a `write` with
`Event.id` = `<sample_id>/<dia_id>` (e.g. `conv-26/D1:3`), role = speaker, at
the session `date_time` + turn index seconds; a shared image adds its caption
to the content. All QA items are `query` steps after the last session (+1
day). QA `evidence` → `evidence` (malformed ids like `D:11:26` or `D30:05` are
normalized; ids not in the conversation are dropped); `category` 1-5 →
`multi_hop`, `temporal`, `open_domain`, `single_hop`, `adversarial`;
adversarial questions → `abstain` (no answer exists in the conversation).

`locomo_exclusions.yaml` lists QA items whose gold annotation is broken, with
reasons; they are `Pack.flagged`, so a run can be scored with and without
them (`--exclude-flagged`).
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from membench.datasets.download import Source

_REVISION = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
SOURCE = Source(
    "locomo",
    f"https://raw.githubusercontent.com/snap-research/locomo/{_REVISION}/data/locomo10.json",
    _REVISION, "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4",
    "CC BY-NC 4.0",
)
CATEGORIES = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop", 5: "adversarial"}
EXCLUSIONS_PATH = Path(__file__).with_name("locomo_exclusions.yaml")
_SESSION = re.compile(r"session_(\d+)")
_DIA_ID = re.compile(r"D:?(\d+):0*(\d+)")


def exclusions() -> dict[str, str]:
    """QA item id (`<sample_id>/q<index>`) -> reason it is excluded."""
    return {e["id"]: e["reason"] for e in yaml.safe_load(EXCLUSIONS_PATH.read_text())}


def _parse_date(text: str) -> datetime:
    """'1:56 pm on 8 May, 2023' -> aware UTC datetime."""
    return datetime.strptime(text, "%I:%M %p on %d %B, %Y").replace(tzinfo=timezone.utc)


def _evidence(raw: list[str]) -> list[str]:
    """Split 'D8:6; D9:17'-style entries and repair 'D:11:26' / 'D30:05'."""
    ids = []
    for entry in raw:
        for part in re.split(r"[;,\s]+", entry.strip()):
            match = _DIA_ID.fullmatch(part)
            ids.append(f"D{match[1]}:{match[2]}" if match else part)
    return [i for i in ids if i]


def build(records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    """(raw pack mapping, QA item id of each query step in order)."""
    episodes, qa_ids = [], []
    for record in records:
        sample, conversation = record["sample_id"], record["conversation"]
        steps: list[dict[str, Any]] = []
        sessions = sorted(
            (int(m[1]), key) for key in conversation if (m := _SESSION.fullmatch(key))
        )
        last = None
        for number, key in sessions:
            start = _parse_date(conversation[f"{key}_date_time"])
            for t, turn in enumerate(conversation[key]):
                content = turn["text"]
                if turn.get("blip_caption"):
                    content += f" [shares an image: {turn['blip_caption']}]"
                last = start + timedelta(seconds=t)
                steps.append({
                    "type": "write", "session_id": f"{sample}/session_{number}",
                    "id": f"{sample}/{turn['dia_id']}", "fact_key": f"{sample}/{turn['dia_id']}",
                    "role": turn["speaker"], "content": content, "timestamp": last.isoformat(),
                })
        known = {step["id"] for step in steps}
        asked_at = (last + timedelta(days=1)).isoformat()
        for q, item in enumerate(record["qa"]):
            adversarial = item["category"] == 5
            evidence = [f"{sample}/{e}" for e in _evidence(item.get("evidence", []))]
            steps.append({
                "type": "query", "session_id": f"{sample}/qa", "query": item["question"],
                "timestamp": asked_at, "category": CATEGORIES[item["category"]],
                "answer": None if adversarial else str(item["answer"]),
                "evidence": [e for e in evidence if e in known], "abstain": adversarial,
            })
            qa_ids.append(f"{sample}/q{q}")
        episodes.append({"steps": steps})
    raw = {
        "name": "locomo",
        "description": "LoCoMo: 10 long conversations, one episode each (CC BY-NC 4.0).",
        "episodes": episodes,
    }
    return raw, qa_ids
