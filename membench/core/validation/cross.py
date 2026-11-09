"""Cross-judge runner: several judges score the same sampled items.

Judges come from a YAML panel (`configs/judges.yaml`): each entry has a
`name`, `model`, optional `base_url` (any OpenAI-compatible endpoint, e.g. a
hosted open-weight model) and the `api_key_env` its key is read from. Every
call goes through `CachedJudge`, so a re-run (or `strict` replay) is free.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from membench.core.scoring.judge import (
    API_KEY_ENV,
    JUDGE_PROMPT_VERSION,
    CachedJudge,
    JudgeCache,
    JudgeError,
    resolve_client,
)
from membench.core.validation.sample import item_prompt


@dataclass(frozen=True)
class JudgeConfig:
    name: str
    model: str
    base_url: str | None = None
    api_key_env: str = API_KEY_ENV


def load_panel(path: str | Path) -> list[JudgeConfig]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    panel = [JudgeConfig(**entry) for entry in data.get("judges", [])]
    if len({j.name for j in panel}) != len(panel):
        raise ValueError(f"{path}: judge names must be unique")
    return panel


def build_panel_judge(config: JudgeConfig, cache_path: str | Path, strict: bool) -> CachedJudge:
    if strict:
        return CachedJudge(None, cache_path, strict=True, model=config.model,
                           base_url=config.base_url)
    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise ValueError(f"judge {config.name!r} needs an API key in ${config.api_key_env}")
    client = resolve_client(model=config.model, base_url=config.base_url, api_key=api_key)
    return CachedJudge(client, cache_path)


def share_cache(judges: dict[str, CachedJudge], cache_path: str | Path) -> None:
    """Point every judge at one in-memory cache, so their writes don't overwrite each other."""
    cache = JudgeCache(cache_path)
    for judge in judges.values():
        judge.cache = cache


def run_judges(items: list[dict], judges: dict[str, CachedJudge]) -> list[dict[str, Any]]:
    """One verdict row per (judge, item); a judge failure is a row with `error`."""
    rows = []
    for name, judge in judges.items():
        for item in items:
            row = {"item_id": item["id"], "metric": item["metric"], "judge": name,
                   "model": judge.model, "base_url": judge.base_url,
                   "prompt_version": JUDGE_PROMPT_VERSION, "passed": None}
            try:
                row["passed"] = judge.ask(*item_prompt(item)).passed
            except JudgeError as exc:
                row["error"] = str(exc)
            rows.append(row)
    return rows
