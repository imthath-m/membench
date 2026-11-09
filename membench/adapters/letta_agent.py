"""Letta agent variant: the agent manages its own memory.

Each event is sent to the agent as a user message; the agent decides what to
keep in its core memory blocks and archival memory through its own memory
tools (an LLM step per write, so the cost profile differs from `letta`).
Recall is the same as `letta`: core blocks plus archival search. Recalled
passages are the agent's own notes, so they trace to no `Event.id`. Forget
asks the agent to erase what it stored from each target event.
"""

from __future__ import annotations

from typing import Any

from membench.adapters.letta import LettaAdapter
from membench.core.adapter import Event
from membench.core.usage import new_usage

FORGET_PROMPT = (
    "Permanently forget the following information. Remove it from your core memory "
    "and delete any archival memory that contains it:\n{content}"
)


class LettaAgentAdapter(LettaAdapter):
    """Letta agent that edits its own memory from messages (`send_message` style)."""

    uses_llm = True

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._contents: dict[str, str] = {}  # Event.id -> content (for forget prompts)
        self._usage = new_usage()

    def _send(self, text: str) -> None:
        response = self._client.agents.messages.create(
            self._agent_id(), messages=[{"role": "user", "content": text}]
        )
        usage = getattr(response, "usage", None)
        self._usage["llm_input_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        self._usage["llm_output_tokens"] += getattr(usage, "completion_tokens", 0) or 0

    def write(self, event: Event) -> None:
        self._contents[event.id] = event.content
        self._send(f"[{event.timestamp}] {event.role}: {event.content}")

    def forget(self, event_ids: list[str]) -> None:
        for event_id in event_ids:
            if event_id in self._contents:
                self._send(FORGET_PROMPT.format(content=self._contents.pop(event_id)))

    def reset(self) -> None:
        super().reset()
        self._contents = {}

    def usage(self) -> dict[str, int]:
        return dict(self._usage)
