"""YAML scenario pack loader + validation.

A pack is an ordered list of steps. Each step is either:
  - a `write`: state / update / contradict a fact. `fact_type` of
    "update" or "contradict" marks that this write supersedes a prior
    fact under the same `key` — `core.scoring.update_correctness` uses
    this to find queries that should reflect the new value.
  - a `query`: ask a question, with expected-fact assertions
    (`expect_contains`, `expect_key`) scored by `core.scoring`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

_EPOCH = datetime(2024, 1, 1, tzinfo=timezone.utc)


class PackValidationError(ValueError):
    """Raised when a pack YAML file fails schema validation."""


@dataclass
class WriteStep:
    type: str  # "write"
    session_id: str
    key: str
    role: str
    content: str
    timestamp: str
    fact_type: str = "state"  # state | update | contradict
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryStep:
    type: str  # "query"
    session_id: str
    query: str
    timestamp: str
    expect_contains: list[str] = field(default_factory=list)
    expect_key: str | None = None
    judge: str | None = None


Step = WriteStep | QueryStep


@dataclass
class Pack:
    name: str
    description: str
    steps: list[Step]


def _require(d: dict, field_name: str, step_index: int) -> Any:
    if field_name not in d:
        raise PackValidationError(
            f"step {step_index}: missing required field {field_name!r}"
        )
    return d[field_name]


def _parse_step(raw: dict, index: int) -> Step:
    if not isinstance(raw, dict) or "type" not in raw:
        raise PackValidationError(f"step {index}: must be a mapping with a 'type' field")

    step_type = raw["type"]
    timestamp = raw.get("timestamp") or (_EPOCH + timedelta(seconds=index)).isoformat()

    if step_type == "write":
        return WriteStep(
            type="write",
            session_id=_require(raw, "session_id", index),
            key=_require(raw, "key", index),
            role=_require(raw, "role", index),
            content=_require(raw, "content", index),
            timestamp=timestamp,
            fact_type=raw.get("fact_type", "state"),
            metadata=raw.get("metadata", {}) or {},
        )
    if step_type == "query":
        return QueryStep(
            type="query",
            session_id=_require(raw, "session_id", index),
            query=_require(raw, "query", index),
            timestamp=timestamp,
            expect_contains=raw.get("expect_contains", []) or [],
            expect_key=raw.get("expect_key"),
            judge=raw.get("judge"),
        )
    raise PackValidationError(
        f"step {index}: unknown step type {step_type!r} (expected 'write' or 'query')"
    )


def load_pack(path: str | Path) -> Pack:
    """Load and validate a scenario pack from a YAML file."""
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise PackValidationError(f"{path}: pack must be a YAML mapping")

    name = raw.get("name", path.stem)
    description = raw.get("description", "")
    raw_steps = raw.get("steps")
    if not raw_steps or not isinstance(raw_steps, list):
        raise PackValidationError(f"{path}: pack must define a non-empty 'steps' list")

    steps = [_parse_step(step, i) for i, step in enumerate(raw_steps)]
    return Pack(name=name, description=description, steps=steps)


def builtin_pack_path(pack_name: str) -> Path:
    """Resolve a builtin pack name (e.g. 'mini') to its YAML file."""
    path = Path(__file__).parent / f"{pack_name}.yaml"
    if not path.exists():
        raise PackValidationError(f"unknown builtin pack {pack_name!r} (no {path})")
    return path
