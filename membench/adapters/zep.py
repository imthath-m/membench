"""Zep Cloud adapter (`zep-cloud` v4).

Each episode gets its own Zep user; each event becomes a `message` episode on that
user's graph at its own `reference_time` (bi-temporal facts). Recall is fact (edge)
search limited to facts valid by `at`. Ingestion is asynchronous: `flush()` waits for
every add/delete task.
"""

from __future__ import annotations

from typing import Any

from membench.core.adapter import Event, MemoryAdapter, RecalledItem, RecallResult
from membench.core.poll import drain, poll_options

try:
    import zep_cloud
    from zep_cloud.client import Zep

    _RETRYABLE: tuple[type[BaseException], ...] = (zep_cloud.TooManyRequestsError,
                                                   zep_cloud.ServiceUnavailableError,
                                                   zep_cloud.BadGatewayError)
except ImportError:  # pragma: no cover - exercised only when zep-cloud missing
    Zep, _RETRYABLE = None, ()


class ZepAdapter(MemoryAdapter):
    """Hosted Zep (API key from `api_key` or `ZEP_API_KEY`)."""

    retryable = _RETRYABLE
    packages = ("zep-cloud",)
    deployment = "hosted"
    uses_llm = True

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if Zep is None:
            raise ImportError("zep-cloud is not installed: `pip install membench[zep]`.")
        super().__init__(config)
        self.top_k = int(self.config.get("top_k", 3))
        self.poll = poll_options(self.config)
        self._client = Zep(api_key=self.config.get("api_key"))
        self._user: str | None = None  # Zep user uuid (= its graph) for this episode
        self._episode, self._tasks = 0, []  # add/delete tasks still running
        self._sources: dict[str, str] = {}  # Zep episode uuid -> Event.id

    def _graph(self) -> str:
        if self._user is None:
            label = f"{self.namespace}-e{self._episode}"
            self._user = self._client.user.create(metadata={"membench": label}).uuid_
        return self._user

    def write(self, event: Event) -> None:
        result = self._client.graph.episode.add(
            self._graph(), data=f"{event.role}: {event.content}", type="message",
            reference_time=event.timestamp,
        )
        self._sources[result.episode.uuid_] = event.id
        self._tasks += [result.task.uuid_] if result.task else []

    def flush(self) -> None:
        drain(self._tasks, self._task_done, **self.poll)

    def _task_done(self, task_uuid: str) -> bool:
        task = self._client.task.get(task_uuid)
        if task.status == "failed":
            raise RuntimeError(f"zep task {task_uuid} failed: {task.error}")
        return task.status in ("succeeded", "partial")

    def recall(self, query: str, *, session_id: str, at: str) -> RecallResult:
        if self._user is None:
            return RecallResult(context="")
        valid = zep_cloud.EdgeSearchFilters.model_validate({"date_filters": {"any_of": [
            {"all_of": [{"field": "valid_at", "operator": "lte", "value": at}]},
            {"all_of": [{"field": "valid_at", "operator": "is_null"}]},
        ]}})
        edges = self._client.graph.search_edges(self._user, query=query, page_size=self.top_k,
                                                filters=valid).items or []
        items = [RecalledItem([self._sources[u] for u in e.episode_uuids or [] if u in self._sources],
                              f"{e.fact} (valid {e.valid_at or '?'} to {e.invalid_at or 'now'})",
                              e.score) for e in edges]
        return RecallResult(context="\n".join(i.text for i in items), items=items)

    def forget(self, event_ids: list[str]) -> None:
        """Delete the Zep episodes (and the facts only they support) of each target."""
        targets = set(event_ids)
        for uuid, source in list(self._sources.items()):
            if source in targets:
                result = self._client.graph.episode.delete(self._graph(), uuid)
                del self._sources[uuid]
                self._tasks += [result.task.uuid_] if result.task else []

    def reset(self) -> None:
        if self._user is not None:
            self._client.user.delete(self._user)
        self._user, self._sources, self._tasks = None, {}, []
        self._episode += 1

    def describe(self) -> dict[str, Any]:
        return {"top_k": self.top_k}
