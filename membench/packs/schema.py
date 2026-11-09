"""YAML scenario pack loader + validation.

A pack is an ordered list of steps, either at the top level (`steps:`) or
split into independent `episodes:` (each a mapping with its own `steps:`);
the runner calls `reset()` between episodes. Each step is one of:
  - a `write`: state / update / contradict a fact. `id` is the opaque
    `Event.id` the adapter sees (auto-generated when omitted; an update
    gets its own id). `fact_key` names the fact for scoring only and never
    reaches an adapter; `fact_type` of "update" or "contradict" marks that
    this write supersedes a prior fact under the same `fact_key`.
  - a `query`: ask a question, with expected-fact assertions
    (`expect_contains`, `expect_key` — a `fact_key`) scored by `core.scoring`.
  - a `forget`: remove earlier writes by id (`target: <id or list of ids>`).
"""

from __future__ import annotations

import hashlib
import json
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
    id: str
    fact_key: str
    role: str
    content: str
    timestamp: str
    fact_type: str = "state"  # state | update | contradict
    metadata: dict[str, Any] = field(default_factory=dict)
    episode: int = 0


@dataclass
class JudgeSpec:
    """LLM-judge expectations on a query step (both fields optional).

    `semantic_recall` is a rubric describing what a correct recall should
    convey; `contradiction` is a rubric describing what would count as a
    contradiction in the recalled context. Each drives the like-named metric.
    """

    semantic_recall: str | None = None
    contradiction: str | None = None


@dataclass
class QueryStep:
    type: str  # "query"
    session_id: str
    query: str
    timestamp: str
    expect_contains: list[str] = field(default_factory=list)
    expect_key: str | None = None
    judge: JudgeSpec | None = None
    episode: int = 0


@dataclass
class ForgetStep:
    type: str  # "forget"
    session_id: str
    target: list[str]
    timestamp: str
    episode: int = 0


Step = WriteStep | QueryStep | ForgetStep


@dataclass
class Pack:
    """`steps` is flat across episodes; each step's `episode` is its index.

    `sha256` hashes the parsed pack content (canonical JSON), so whitespace
    and comments don't change it; the run manifest records it.
    """

    name: str
    description: str
    steps: list[Step]
    sha256: str = ""


def _require(d: dict, field_name: str, step_index: int) -> Any:
    if field_name not in d:
        raise PackValidationError(
            f"step {step_index}: missing required field {field_name!r}"
        )
    return d[field_name]


def _parse_judge(raw: Any, index: int) -> JudgeSpec | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise PackValidationError(f"step {index}: 'judge' must be a mapping")
    return JudgeSpec(
        semantic_recall=raw.get("semantic_recall"),
        contradiction=raw.get("contradiction"),
    )


def _parse_step(raw: dict, index: int, episode: int) -> Step:
    if not isinstance(raw, dict) or "type" not in raw:
        raise PackValidationError(f"step {index}: must be a mapping with a 'type' field")

    step_type = raw["type"]
    timestamp = raw.get("timestamp") or (_EPOCH + timedelta(seconds=index)).isoformat()

    if step_type == "write":
        if "key" in raw and "fact_key" not in raw:
            raise PackValidationError(f"step {index}: 'key' was renamed to 'fact_key' in 0.2")
        return WriteStep(
            type="write",
            session_id=_require(raw, "session_id", index),
            id=str(raw.get("id") or f"evt-{index}"),
            fact_key=_require(raw, "fact_key", index),
            role=_require(raw, "role", index),
            content=_require(raw, "content", index),
            timestamp=timestamp,
            fact_type=raw.get("fact_type", "state"),
            metadata=raw.get("metadata", {}) or {},
            episode=episode,
        )
    if step_type == "query":
        return QueryStep(
            type="query",
            session_id=_require(raw, "session_id", index),
            query=_require(raw, "query", index),
            timestamp=timestamp,
            expect_contains=raw.get("expect_contains", []) or [],
            expect_key=raw.get("expect_key"),
            judge=_parse_judge(raw.get("judge"), index),
            episode=episode,
        )
    if step_type == "forget":
        target = _require(raw, "target", index)
        return ForgetStep(
            type="forget",
            session_id=_require(raw, "session_id", index),
            target=[str(t) for t in target] if isinstance(target, list) else [str(target)],
            timestamp=timestamp,
            episode=episode,
        )
    raise PackValidationError(
        f"step {index}: unknown step type {step_type!r} (expected 'write', 'query' or 'forget')"
    )


def _check_ids(steps: list[Step]) -> None:
    """Event ids are unique; a forget targets an earlier write in its episode."""
    seen: dict[str, int] = {}  # event id -> episode
    for index, step in enumerate(steps):
        if isinstance(step, WriteStep):
            if step.id in seen:
                raise PackValidationError(f"step {index}: duplicate event id {step.id!r}")
            seen[step.id] = step.episode
        elif isinstance(step, ForgetStep):
            for target in step.target:
                if seen.get(target) != step.episode:
                    raise PackValidationError(
                        f"step {index}: forget target {target!r} is not an earlier write "
                        "in this episode"
                    )


def parse_pack(raw: Any, default_name: str = "pack") -> Pack:
    """Validate an already-parsed pack mapping (e.g. from `yaml.safe_load`).

    Split out of `load_pack` so a generated pack (Phase 6) can be validated
    against the exact same rules before it's ever written to disk.
    """
    if not isinstance(raw, dict):
        raise PackValidationError(f"{default_name}: pack must be a YAML mapping")

    name = raw.get("name", default_name)
    description = raw.get("description", "")
    raw_episodes = raw["episodes"] if "episodes" in raw else [{"steps": raw.get("steps")}]
    if not raw_episodes or not isinstance(raw_episodes, list):
        raise PackValidationError(f"{default_name}: 'episodes' must be a non-empty list")

    steps: list[Step] = []
    for episode, raw_episode in enumerate(raw_episodes):
        raw_steps = raw_episode.get("steps") if isinstance(raw_episode, dict) else None
        if not raw_steps or not isinstance(raw_steps, list):
            raise PackValidationError(
                f"{default_name}: pack (and each episode) must define a non-empty 'steps' list"
            )
        steps += [_parse_step(s, len(steps) + i, episode) for i, s in enumerate(raw_steps)]
    _check_ids(steps)
    digest = hashlib.sha256(json.dumps(raw, sort_keys=True, default=str).encode()).hexdigest()
    return Pack(name=name, description=description, steps=steps, sha256=digest)


def load_pack(path: str | Path) -> Pack:
    """Load and validate a scenario pack from a YAML file."""
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)
    return parse_pack(raw, default_name=path.stem)


def builtin_pack_path(pack_name: str) -> Path:
    """Resolve a builtin pack name (e.g. 'mini' or 'customer-support') to its YAML file.

    Hyphens are normalized to underscores so CLI-style names (as used in the
    docs and README, e.g. `--pack customer-support`) resolve to the on-disk
    `customer_support.yaml`.
    """
    path = Path(__file__).parent / f"{pack_name.replace('-', '_')}.yaml"
    if not path.exists():
        raise PackValidationError(f"unknown builtin pack {pack_name!r} (no {path})")
    return path
