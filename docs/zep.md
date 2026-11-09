# Zep adapter

Zep Cloud through the `zep-cloud` v4 SDK. For self-hosting, Zep's open-source
engine is Graphiti (Zep Community Edition is deprecated); a Graphiti adapter is
not part of this phase.

```bash
pip install membench[zep]
export ZEP_API_KEY=z_...
membench run --backend zep --pack mini --backend-config configs/default/zep.yaml
```

## How it maps onto the contract

| Contract | Zep call |
|---|---|
| `write` | `graph.episode.add(<user uuid>, data="<role>: <content>", type="message", reference_time=<event time>)` |
| `flush` | polls `task.get(<task uuid>)` for every add / delete until `succeeded` / `partial` (`failed` fails the query) |
| `recall` | `graph.search_edges(<user uuid>, query, page_size=top_k, filters=<valid_at <= at or null>)` |
| `forget` | `graph.episode.delete(...)` for each episode written from a target event |
| `reset` | `user.delete(<user uuid>)`; the next write creates a fresh user |

- **User graph per episode:** each episode gets a new Zep user (metadata
  `{"membench": "<namespace>-e<n>"}`); events go to that user's graph.
- **Bi-temporal facts:** every episode carries the event's `reference_time`, so
  extracted facts get real `valid_at` / `invalid_at` dates. Recall drops facts
  that only become valid after the query time `at`, and each context line
  reads `<fact> (valid <valid_at> to <invalid_at or now>)`.
- **Event ids:** recalled facts map back through their `episode_uuids`.
- **Async ingestion:** adds and deletes return tasks; `flush()` waits for them.
- The context block (`graph.get_context`) is the other recall path Zep
  offers; it is not used by default because it does not return per-fact
  episode ids. Vendor review may tune this.

## Config

| Key | Default | Meaning |
|---|---|---|
| `top_k` | 3 | facts per recall |
| `api_key` | `ZEP_API_KEY` | Zep API key |
| `poll_interval_s` / `flush_timeout_s` | 1.0 / 600 | task polling |

Retries: `TooManyRequestsError`, `ServiceUnavailableError`, `BadGatewayError`.
Token usage is not reported by the API (`cost.backend` null).

Live check: `MEMBENCH_LIVE=1 pytest "tests/test_live_backends.py::test_live_mini_runs_clean[zep]"`
or `gh workflow run live.yml -f backend=zep`.
