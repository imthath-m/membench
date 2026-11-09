# Mem0 Platform adapter

Hosted Mem0 (`MemoryClient` from `mem0ai`), as opposed to the local `mem0` OSS
backend.

```bash
pip install membench[mem0-platform]
export MEM0_API_KEY=m0-...
membench run --backend mem0-platform --pack mini --backend-config configs/default/mem0-platform.yaml
```

## How it maps onto the contract

| Contract | Mem0 Platform call |
|---|---|
| `write` | `client.add([{role, content}], user_id=..., metadata={"membench_event_id": id}, timestamp=<event unix time>)` (`POST /v3/memories/add/`) |
| `flush` | polls `GET /v1/event/<event_id>/` for every add until `SUCCEEDED` (`FAILED` fails the query) |
| `recall` | `client.search(query, filters={"user_id": ...}, top_k=top_k)` |
| `forget` | flushes, lists the user's memories (`get_all`), deletes those whose metadata names a target event |
| `reset` | `delete_all(user_id=...)`, then switches to a new user_id |

- **Async ingestion:** the v3 add endpoint queues the work and returns an
  `event_id`; memories exist only once the event has succeeded, so `flush()`
  is what makes a write visible.
- **Namespace per episode:** user_id is `<namespace>-e<n>` and `n` moves on at
  every `reset()`, so an episode never sees the last one even if Mem0's
  deletion lags. Repeats use `membench-r<i>` as the namespace.
- **Event ids:** each memory carries the source `Event.id` in its metadata;
  that is how recalled memories map back to events. Roles other than
  `assistant` are sent as `user`.
- **Timestamps:** writes are backdated to the event time (`timestamp`), so
  Mem0's own temporal reasoning sees the pack's timeline.

## Config (`--backend-config`)

| Key | Default | Meaning |
|---|---|---|
| `top_k` | 3 | memories per recall |
| `api_key` | `MEM0_API_KEY` | Mem0 API key |
| `host` | `https://api.mem0.ai` | API host |
| `poll_interval_s` | 1.0 | seconds between event polls |
| `flush_timeout_s` | 600 | give up waiting for events after this; raise `--step-timeout` to match |

The published default is `configs/default/mem0-platform.yaml`; see
`docs/vendor-review.md` for the vendor-tuned slot.

Retries: `mem0.exceptions.RateLimitError` and `NetworkError`. Cost: the
platform does not report token usage, so `usage()` is not tracked
(`cost.backend` is null); extraction is LLM-based, so runs default to 3 repeats.

Live check (needs `MEM0_API_KEY`):
`MEMBENCH_LIVE=1 pytest "tests/test_live_backends.py::test_live_mini_runs_clean[mem0-platform]"`
or `gh workflow run live.yml -f backend=mem0-platform`.
