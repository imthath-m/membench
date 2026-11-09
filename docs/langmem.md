# LangMem adapter

LangMem's memory-store manager (LLM extraction) over a LangGraph `BaseStore`.

```bash
pip install membench[langmem]
export OPENAI_API_KEY=sk-...
membench run --backend langmem --pack mini --backend-config configs/default/langmem.yaml
```

| Contract | LangMem / LangGraph call |
|---|---|
| `write` | `create_memory_store_manager(model, namespace=("membench", "{langgraph_user_id}"), store=store).invoke({"messages": [event]}, config={"configurable": {"langgraph_user_id": "<namespace>-e<n>"}})` |
| `recall` | `store.search(("membench", "<namespace>-e<n>"), query=query, limit=top_k)` |
| `forget` | `store.delete(...)` for each memory key the manager put while handling a target event |
| `reset` | deletes the episode namespace's items and moves to a new namespace |

- The store is an `InMemoryStore` with an embedding index (`embedding`,
  `dims`), so state lives in the process; each run starts empty.
- Each write runs the manager synchronously: it searches existing memories,
  then inserts or updates extracted ones. Keys it returns map memories back to
  the events that created or updated them. `flush()` is a no-op.
- Manager settings are LangMem's defaults (`enable_deletes=False` in
  `create_memory_store_manager`, default instructions, `query_limit=5`).
- `usage()` counts chat-model tokens via LangChain's
  `UsageMetadataCallbackHandler`; embedding tokens are not reported (0).

| Key | Default | Meaning |
|---|---|---|
| `top_k` | 3 | memories per recall |
| `model` | `openai:gpt-4o-mini` | extraction chat model (`provider:model`) |
| `embedding` | `openai:text-embedding-3-small` | store index embedder |
| `dims` | 1536 | embedding dimensions |

Retries: OpenAI `RateLimitError`, `APITimeoutError`, `APIConnectionError`.

Live check: `MEMBENCH_LIVE=1 pytest "tests/test_live_backends.py::test_live_mini_runs_clean[langmem]"`
or `gh workflow run live.yml -f backend=langmem`.
