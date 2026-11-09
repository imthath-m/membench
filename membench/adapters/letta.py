"""Letta adapter: archival-memory passages plus core memory blocks.

Each episode gets its own Letta agent. Events are inserted as archival
passages (dated at the event time); recall is the agent's core memory blocks
plus an archival search of passages created by `at`. No agent step runs, so no
LLM is called (embeddings only). `letta-agent` is the variant where the agent
manages its own memory from messages.
"""

from __future__ import annotations

from typing import Any

from membench.core.adapter import Event, MemoryAdapter, RecalledItem, RecallResult

try:
    import letta_client
    from letta_client import Letta

    _RETRYABLE: tuple[type[BaseException], ...] = (letta_client.RateLimitError,
                                                   letta_client.APITimeoutError,
                                                   letta_client.APIConnectionError)
except ImportError:  # pragma: no cover - exercised only when letta-client missing
    Letta, _RETRYABLE = None, ()

DEFAULT_BLOCKS = [{"label": "human", "value": ""}, {"label": "persona", "value": ""}]


class LettaAdapter(MemoryAdapter):
    """Letta Cloud or a self-hosted server (`base_url`); key from `api_key` / `LETTA_API_KEY`."""

    retryable = _RETRYABLE
    packages = ("letta-client",)
    deployment = "hosted"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if Letta is None:
            raise ImportError("letta-client is not installed: `pip install membench[letta]`.")
        super().__init__(config)
        self.top_k = int(self.config.get("top_k", 3))
        self.model = self.config.get("model", "openai/gpt-4o-mini")
        self.embedding = self.config.get("embedding", "openai/text-embedding-3-small")
        self._client = Letta(api_key=self.config.get("api_key"), base_url=self.config.get("base_url"))
        self._agent: str | None = None
        self._episode = 0
        self._sources: dict[str, str] = {}  # passage id -> Event.id

    def _agent_id(self) -> str:
        if self._agent is None:
            self._agent = self._client.agents.create(
                name=f"{self.namespace}-e{self._episode}", model=self.model,
                embedding=self.embedding, tags=["membench"],
                memory_blocks=self.config.get("memory_blocks", DEFAULT_BLOCKS),
            ).id
        return self._agent

    def write(self, event: Event) -> None:
        passages = self._client.agents.passages.create(
            self._agent_id(), text=f"{event.role}: {event.content}", created_at=event.timestamp
        )
        for passage in passages:  # long text may be chunked into several passages
            self._sources[passage.id] = event.id

    def recall(self, query: str, *, session_id: str, at: str) -> RecallResult:
        if self._agent is None:
            return RecallResult(context="")
        blocks = [f"<{b.label}>\n{b.value}\n</{b.label}>"
                  for b in self._client.agents.blocks.list(self._agent) if b.value]
        found = self._client.agents.passages.search(
            self._agent, query=query, top_k=self.top_k, end_datetime=at
        ).results
        items = [RecalledItem([self._sources[p.id]] if p.id in self._sources else [], p.content)
                 for p in found]
        return RecallResult(context="\n".join(blocks + [i.text for i in items]), items=items)

    def forget(self, event_ids: list[str]) -> None:
        targets = set(event_ids)
        for passage_id, source in list(self._sources.items()):
            if source in targets:
                self._client.agents.passages.delete(passage_id, agent_id=self._agent_id())
                del self._sources[passage_id]

    def reset(self) -> None:
        if self._agent is not None:
            self._client.agents.delete(self._agent)
        self._agent, self._sources = None, {}
        self._episode += 1

    def describe(self) -> dict[str, Any]:
        return {"llm_model": self.model, "embedding_model": self.embedding, "top_k": self.top_k}
