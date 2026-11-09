"""Committed results layout: `results/<YYYY-MM-DD>/<dataset>/<backend>/`.

Each result dir holds `report.json` (manifest, traces, scores), `judge_cache.json`
(only the cache entries the report's judge scores replay; `{}` without a judge)
and `meta.json` (`uv.lock` sha256, config label, what was trimmed). Reports are
written as compact JSON with recalled item texts dropped: scoring reads each
query's `context` and item `event_ids`, never item text, so `rescore` still
reproduces every score. A vendor-tuned config gets its own
`<backend>-vendor-tuned/` dir and leaderboard row.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from membench.core.scoring.judge import JudgeCache

CONFIG_LABELS = ("default", "vendor-tuned")
TRIMMED_ITEM_TEXT = "traces[].detail.items[].text"


class RecordingCache(JudgeCache):
    """A judge cache that remembers every entry read from it."""

    def __init__(self, path: str | Path) -> None:
        super().__init__(path)
        self.used: dict[str, dict] = {}

    def get(self, key: str) -> dict | None:
        entry = super().get(key)
        if entry is not None:
            self.used[key] = entry
        return entry


def result_dir(report: dict, results: Path, config: str = "default") -> Path:
    """`results/<date>/<dataset>/<backend>[-vendor-tuned]` for a run report."""
    if config not in CONFIG_LABELS:
        raise ValueError(f"config label must be one of {CONFIG_LABELS}, not {config!r}")
    manifest = report["manifest"]
    dataset = (manifest.get("dataset") or {}).get("id") or manifest["pack"]["name"]
    subset = manifest["pack"].get("subset") or {}
    dataset += "".join(f"-{k}{v}" for k, v in sorted(subset.items()))
    backend = manifest["backend"]["name"] + ("" if config == "default" else f"-{config}")
    return results / manifest["started_at"][:10] / dataset / backend


def trim(report: dict) -> list[str]:
    """Drop recalled item texts (in place, repeats included); returns what was trimmed."""
    for run in [report, *report.get("repeats", [])]:
        for trace in run["traces"]:
            for item in trace["detail"].get("items", []):
                item.pop("text", None)
    return [TRIMMED_ITEM_TEXT]


def sha256_file(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def write_result(directory: Path, report: dict, judge_cache: dict, meta: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text(json.dumps(report, separators=(",", ":")) + "\n")
    (directory / "judge_cache.json").write_text(
        json.dumps(judge_cache, indent=2, sort_keys=True) + "\n"
    )
    (directory / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
