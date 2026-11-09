# Letta adapter (archival memory)

Letta through `letta-client`: archival-memory passages plus core memory blocks.
For the variant where the agent manages its own memory, see
[`letta-agent`](letta-agent.md).

```bash
pip install membench[letta]
export LETTA_API_KEY=...            # Letta Cloud
# or a self-hosted server: base_url: http://localhost:8283 in the config
membench run --backend letta --pack mini --backend-config configs/default/letta.yaml
```

## How it maps onto the contract

| Contract | Letta call |
|---|---|
| (first write) | `agents.create(name="<namespace>-e<n>", model, embedding, memory_blocks, tags=["membench"])` |
| `write` | `agents.passages.create(agent_id, text="<role>: <content>", created_at=<event time>)` |
| `recall` | core blocks (`agents.blocks.list`) + `agents.passages.search(agent_id, query, top_k, end_datetime=at)` |
| `forget` | `agents.passages.delete(passage_id, agent_id=...)` for each passage of a target event |
| `reset` | `agents.delete(agent_id)`; the next write creates a new agent |

- One agent per episode; passages are dated at the event time and recall only
  searches passages created by `at`.
- Context = non-empty core blocks (`<label>\nvalue\n</label>`), then the
  archival hits. Items are the passages, mapped back to `Event.id`.
- No agent step runs, so no LLM is called: only embeddings (`uses_llm` is
  false, one repeat by default). Passage writes are synchronous; `flush()` is
  a no-op.

## Config

| Key | Default | Meaning |
|---|---|---|
| `top_k` | 3 | passages per recall |
| `model` | `openai/gpt-4o-mini` | agent model handle (unused by archival writes) |
| `embedding` | `openai/text-embedding-3-small` | embedding handle |
| `memory_blocks` | `human`, `persona` (empty) | core blocks at agent creation |
| `api_key` / `base_url` | `LETTA_API_KEY` / Letta Cloud | server |

Retries: `letta_client.RateLimitError`, `APITimeoutError`, `APIConnectionError`.

Live check: `MEMBENCH_LIVE=1 pytest "tests/test_live_backends.py::test_live_mini_runs_clean[letta]"`
or `gh workflow run live.yml -f backend=letta`.
