# Cognee adapter

Cognee's v1 pipeline (`add` → `cognify` → `search`), run in-process.

```bash
pip install membench[cognee]
export LLM_API_KEY=sk-...            # Cognee's LLM / embedder key (OpenAI by default)
membench run --backend cognee --pack mini --backend-config configs/default/cognee.yaml
```

| Contract | Cognee call |
|---|---|
| `write` | `cognee.add("[<ts>] <role>: <content>", dataset_name="<namespace>-e<n>")` |
| `flush` | `cognee.cognify(datasets=[...])` when anything was added since the last flush |
| `recall` | `cognee.search(query, query_type=SearchType.<search_type>, datasets=[...], top_k=top_k)` |
| `forget` | `cognee.forget(data_id=..., dataset=...)` for each data item of a target event |
| `reset` | `cognee.prune.prune_data()` + `cognee.prune.prune_system(metadata=True)` |

- Cognee's API is async; the adapter runs every call on one event loop it owns.
- `reset()` wipes Cognee's local data, graph, vector and metadata stores (the
  whole local install, not just this run): use a dedicated Cognee data dir
  (`DATA_ROOT_DIRECTORY` / `SYSTEM_ROOT_DIRECTORY`).
- Recall defaults to `CHUNKS`: chunk payloads carry `document_id` (the data
  item), which maps back to `Event.id`. Any other `search_type` (e.g.
  `GRAPH_COMPLETION`) is run with `only_context=True`, so the context is the
  graph context Cognee's LLM would see; those items carry no event ids.
- Identical content dedups to one data item; forgetting it removes it for
  every event that wrote it.
- Token usage is not tracked (`cost.backend` null). If AWS credentials are set
  in the environment, Cognee tries S3 and needs `s3fs`.

| Key | Default | Meaning |
|---|---|---|
| `top_k` | 3 | results per recall |
| `search_type` | `CHUNKS` | a `cognee.SearchType` name |

Retries: OpenAI `RateLimitError`, `APITimeoutError`, `APIConnectionError`.

Live check: `MEMBENCH_LIVE=1 pytest "tests/test_live_backends.py::test_live_mini_runs_clean[cognee]"`
or `gh workflow run live.yml -f backend=cognee`.
