"""Adapter contract: MemoryAdapter ABC, Event, and RecallResult.

Adapters are thin, stateless config wrappers over a backend client. They
only move data (write/recall/forget) — the runner owns timing, token
accounting, and scoring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Event:
    """One fact/turn/tool-result to be ingested by an adapter."""

    session_id: str
    key: str
    role: str
    content: str
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecallResult:
    """What a `recall()` call returns.

    `context` is exactly what would be injected into a prompt — this is
    what the token-cost metric measures.
    """

    context: str
    items: list[Any] = field(default_factory=list)
    token_count: int = 0


class MemoryAdapter(ABC):
    """Thin interface every memory backend implements.

    Construction takes a plain dict of config (from pack/CLI config).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def write(self, event: Event) -> None:
        """Ingest one fact/turn/tool-result."""

    @abstractmethod
    def recall(self, query: str) -> RecallResult:
        """Return injected context + items for a query."""

    @abstractmethod
    def forget(self, key: str) -> None:
        """Remove a previously written event by its stable key."""
