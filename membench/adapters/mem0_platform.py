"""Mem0 Platform adapter: the hosted `MemoryClient` from `mem0ai`.

Adds are processed asynchronously, so `flush()` polls `GET /v1/event/<id>/` until
every add has finished. Each episode gets its own user_id (`<namespace>-e<n>`);
memories carry the source `Event.id` in their metadata.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from membench.core.adapter import Event, MemoryAdapter, RecalledItem, RecallResult
from membench.core.poll import drain, poll_options

try:
    from mem0 import MemoryClient
    from mem0.exceptions import NetworkError, RateLimitError

    _RETRYABLE: tuple[type[BaseException], ...] = (RateLimitError, NetworkError)
except ImportError:  # pragma: no cover - exercised only when mem0 missing
    MemoryClient, _RETRYABLE = None, ()

_KEY = "membench_event_id"


class Mem0PlatformAdapter(MemoryAdapter):
    """Hosted Mem0 Platform (API key from `api_key` or `MEM0_API_KEY`)."""

    retryable = _RETRYABLE
    packages = ("mem0ai",)
    deployment = "hosted"
    uses_llm = True

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if MemoryClient is None:
            raise ImportError("mem0ai is not installed. Install it with "
                              "`pip install membench[mem0-platform]` to use mem0-platform.")
        super().__init__(config)
        self.top_k = int(self.config.get("top_k", 3))
        self.poll = poll_options(self.config)
        self._client = MemoryClient(api_key=self.config.get("api_key"), host=self.config.get("host"))
        self._episode, self.user_id = 0, f"{self.namespace}-e0"
        self._pending: list[str] = []  # add event ids not yet processed

    def write(self, event: Event) -> None:
        when = datetime.fromisoformat(event.timestamp.replace("Z", "+00:00"))
        role = "assistant" if event.role == "assistant" else "user"
        result = self._client.add(
            [{"role": role, "content": event.content}], user_id=self.user_id,
            metadata={_KEY: event.id}, timestamp=int(when.timestamp()),
        )
        if result.get("event_id") and result.get("status") not in ("SUCCEEDED", "FAILED"):
            self._pending.append(result["event_id"])

    def flush(self) -> None:
        drain(self._pending, self._event_done, **self.poll)

    def _event_done(self, event_id: str) -> bool:
        event = self._client.client.get(f"/v1/event/{event_id}/").raise_for_status().json()
        if event.get("status") == "FAILED":
            raise RuntimeError(f"mem0 add failed: {event.get('error')}")
        return event.get("status") == "SUCCEEDED"

    def recall(self, query: str, *, session_id: str, at: str) -> RecallResult:
        result = self._client.search(query, filters={"user_id": self.user_id}, top_k=self.top_k)
        items = [RecalledItem(_sources(m), m["memory"], m.get("score"))
                 for m in result.get("results", [])]
        return RecallResult(context="\n".join(i.text for i in items), items=items)

    def forget(self, event_ids: list[str]) -> None:
        """Delete every memory whose source event is a target (after pending adds land)."""
        self.flush()
        targets, doomed, page = set(event_ids), [], 1
        while True:
            result = self._client.get_all(filters={"user_id": self.user_id}, page=page, page_size=100)
            doomed += [m["id"] for m in result.get("results", []) if targets & set(_sources(m))]
            if not result.get("next"):
                break
            page += 1
        for memory_id in doomed:
            self._client.delete(memory_id)

    def reset(self) -> None:
        self._pending.clear()
        self._client.delete_all(user_id=self.user_id)
        self._episode += 1  # a fresh user per episode, even if deletion lags
        self.user_id = f"{self.namespace}-e{self._episode}"

    def describe(self) -> dict[str, Any]:
        return {"top_k": self.top_k, "host": self._client.host}


def _sources(memory: dict[str, Any]) -> list[str]:
    source = (memory.get("metadata") or {}).get(_KEY)
    return [source] if source else []
