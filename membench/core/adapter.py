"""Adapter contract v2: MemoryAdapter ABC, Event, RecalledItem, RecallResult.

Adapters are thin, stateless config wrappers over a backend client. They
only move data (write/flush/recall/forget) — the runner owns timing, token
accounting, and scoring. Adapters never see scoring ground truth: `Event.id`
is an opaque per-write id, and the pack's fact keys stay with the scorer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Event:
    """One fact/turn/tool-result to be ingested by an adapter.

    `id` is unique per write (an update gets a new id) and is the only id an
    adapter sees; `forget()` targets it and `RecalledItem.event_ids` maps back
    to it.
    """

    id: str
    session_id: str
    role: str
    content: str
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecalledItem:
    """One recalled unit, traced back to the `Event.id`s it came from.

    `score` is the backend's relevance score (higher is better), or None.
    """

    event_ids: list[str]
    text: str
    score: float | None = None


@dataclass
class RecallResult:
    """What a `recall()` call returns.

    `context` is exactly what would be injected into a prompt — the runner
    counts its tokens for the token-cost metric.
    """

    context: str
    items: list[RecalledItem] = field(default_factory=list)


class MemoryAdapter(ABC):
    """Thin interface every memory backend implements.

    Construction takes a plain dict of config (from pack/CLI config).
    `retryable` lists transient exception types (rate limits, timeouts) the
    runner retries with backoff; anything else fails the step. `packages` and
    `deployment` (plus `describe()`) feed the run manifest; `uses_llm` sets the
    default number of repeats.
    """

    retryable: tuple[type[BaseException], ...] = ()
    packages: tuple[str, ...] = ()  # backend SDK distributions whose versions the manifest records
    deployment: str = "local"  # "local" | "hosted"; a hosted region comes from config["region"]
    uses_llm: bool = False  # backend calls an LLM (non-deterministic): default --repeats 3
    # Oracle baseline only: the runner sets `adapter.evidence` (the query's gold
    # `evidence` event ids) before each recall. Every real backend leaves it False.
    sees_evidence: bool = False

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @property
    def namespace(self) -> str:
        """Where this run's state lives (`config["namespace"]`, default "membench").

        Backends with persistent state default their collection / user id to it;
        the CLI gives each repeat a new one.
        """
        return self.config.get("namespace", "membench")

    @abstractmethod
    def write(self, event: Event) -> None:
        """Ingest one fact/turn/tool-result."""

    @abstractmethod
    def recall(self, query: str, *, session_id: str, at: str) -> RecallResult:
        """Return injected context + items for a query asked in `session_id` at time `at`."""

    @abstractmethod
    def forget(self, event_ids: list[str]) -> None:
        """Remove previously written events by their `Event.id`s."""

    def flush(self) -> None:
        """Block until async ingestion is done. Default: no-op (ingest is synchronous)."""

    def reset(self) -> None:
        """Clear any state persisted by earlier runs or episodes. Default: no-op."""

    def usage(self) -> dict[str, int] | None:
        """Cumulative backend token counts, or None if not tracked.

        Keys: `llm_input_tokens`, `llm_output_tokens`, `embedding_tokens`; priced
        against the `llm_model` / `embedding_model` from `describe()`.
        """
        return None

    def describe(self) -> dict[str, Any] | None:
        """Models and settings that shape results (e.g. `llm_model`, `llm_base_url`,
        `temperature`, `embedding_model`), or None. Recorded in the run manifest."""
        return None
