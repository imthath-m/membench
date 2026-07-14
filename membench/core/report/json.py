"""Emit a RunResult as JSON: raw run + per-step trace + score stubs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from membench.core.runner import RunResult


def to_dict(result: RunResult) -> dict:
    return asdict(result)


def write_json(result: RunResult, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.write_text(json.dumps(to_dict(result), indent=2))
