"""Count backend token usage from OpenAI SDK responses.

Adapters whose backend calls an LLM / embedder through an `openai.OpenAI`
client (Mem0's default) hand that client to `meter_openai`; every
`chat.completions.create` / `embeddings.create` call then adds its API-reported
`usage` to a shared totals dict, which the adapter returns from `usage()`.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

_LOCK = threading.Lock()  # backends may call the SDK from worker threads


def new_usage() -> dict[str, int]:
    return {"llm_input_tokens": 0, "llm_output_tokens": 0, "embedding_tokens": 0}


def _counted(create: Callable[..., Any], totals: dict[str, int], llm: bool) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        response = create(*args, **kwargs)
        usage = getattr(response, "usage", None)
        with _LOCK:
            if llm:
                totals["llm_input_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
                totals["llm_output_tokens"] += getattr(usage, "completion_tokens", 0) or 0
            else:
                totals["embedding_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        return response

    return wrapper


def meter_openai(client: Any, totals: dict[str, int]) -> None:
    """Wrap `client`'s chat-completion and embedding calls to add up `usage`.

    A None client, or one without those endpoints, is left alone.
    """
    completions = getattr(getattr(client, "chat", None), "completions", None)
    if completions is not None:
        completions.create = _counted(completions.create, totals, llm=True)
    embeddings = getattr(client, "embeddings", None)
    if embeddings is not None:
        embeddings.create = _counted(embeddings.create, totals, llm=False)
