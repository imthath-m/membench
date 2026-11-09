# Vendor review

Every Phase 14 backend runs with its vendor's defaults. Before results are
published, each vendor is invited to review its adapter and propose a tuned
config. Results are then published side by side, labelled **default** and
**vendor-tuned**; the two are never merged into one number.

## Configs

| Backend | Default (published as "default") | Vendor-tuned (published as "vendor-tuned") | Review issue |
|---|---|---|---|
| `mem0-platform` | [`configs/default/mem0-platform.yaml`](../configs/default/mem0-platform.yaml) | pending review | not opened yet |
| `zep` | [`configs/default/zep.yaml`](../configs/default/zep.yaml) | pending review | not opened yet |
| `letta` | [`configs/default/letta.yaml`](../configs/default/letta.yaml) | pending review | not opened yet |
| `letta-agent` | [`configs/default/letta-agent.yaml`](../configs/default/letta-agent.yaml) | pending review | not opened yet |
| `langmem` | [`configs/default/langmem.yaml`](../configs/default/langmem.yaml) | pending review | not opened yet |
| `cognee` | [`configs/default/cognee.yaml`](../configs/default/cognee.yaml) | pending review | not opened yet |

Rules:

- A default config only restates the adapter's built-in defaults (a test
  checks this). It changes only when the adapter's defaults change.
- A vendor-tuned config goes in `configs/vendor-tuned/<backend>.yaml`, with a
  header comment linking the vendor's review comment. It may change config
  keys only; an adapter code change goes through a normal PR and applies to
  both columns.
- Publish both rows, run with `membench run --backend <b> --backend-config
  <file>`. The report manifest records the full redacted config.

## Opening the issues (manual step)

The issues are drafted below but not opened: opening them on the vendors'
repositories is for a maintainer to do. For each vendor:

1. Open an issue on the repository named in its section, with the title and
   body below (fill in the PR / commit link).
2. Put the issue URL in the table above.
3. When the vendor replies with a tuned config, add it under
   `configs/vendor-tuned/` and update the table.

## Draft issues

### Mem0 (`mem0ai/mem0`)

**Title:** Review request: Mem0 Platform adapter and default config in MemBench

> Hi Mem0 team. MemBench (https://github.com/imthath-m/membench) is an
> open-source eval harness for agent memory backends. We have added a Mem0
> Platform adapter (`membench/adapters/mem0_platform.py`, under 100 lines;
> docs in `docs/mem0-platform.md`) and would like your review before we
> publish numbers.
>
> What it does: `add` with `user_id`, a backdated `timestamp` and the source
> event id in `metadata`; waits on `GET /v1/event/<id>/` before each query;
> `search` with `filters={"user_id": ...}` and `top_k=3`; forget via
> `get_all` + `delete`; one user_id per episode, cleared with `delete_all`.
>
> Questions:
> 1. Are these the settings you would recommend for a fair default (e.g.
>    `top_k`, `rerank`, `keyword_search`, `infer`, `reference_date` on search)?
> 2. Is polling the event endpoint the right way to wait for ingestion?
> 3. If you would tune anything, could you propose a config? We publish it
>    next to the default, labelled "vendor-tuned".

### Zep (`getzep/zep`)

**Title:** Review request: Zep Cloud adapter and default config in MemBench

> Hi Zep team. MemBench (https://github.com/imthath-m/membench) has added a
> Zep Cloud adapter on `zep-cloud` v4 (`membench/adapters/zep.py`; docs in
> `docs/zep.md`) and would like your review before we publish numbers.
>
> What it does: one user per episode; each event is
> `graph.episode.add(type="message", reference_time=<event time>)`; waits on
> the returned tasks before each query; recall is `graph.search_edges` with
> `page_size=3` and a `valid_at <= query time` date filter, rendered as
> `<fact> (valid <from> to <to>)`; forget deletes the event's episodes.
>
> Questions:
> 1. Would you recommend `graph.get_context` (context block), node search, a
>    different reranker or a larger page size as the default recall?
> 2. Is `message` the right episode type for single facts / turns, and is the
>    task API the right way to wait for ingestion?
> 3. Should we also cover self-hosted Graphiti, and with which settings?
> 4. If you would tune anything, could you propose a config? It is published
>    next to the default, labelled "vendor-tuned".

### Letta (`letta-ai/letta`)

**Title:** Review request: Letta adapters (archival + agent-managed) in MemBench

> Hi Letta team. MemBench (https://github.com/imthath-m/membench) has added
> two Letta adapters on `letta-client` (`membench/adapters/letta.py`,
> `letta_agent.py`; docs in `docs/letta.md`, `docs/letta-agent.md`) and would
> like your review before we publish numbers.
>
> - `letta`: one agent per episode; events are archival passages
>   (`passages.create`, `created_at` = event time); recall is core blocks +
>   `passages.search(top_k=3, end_datetime=<query time>)`.
> - `letta-agent`: each event is a user message; the agent manages its own
>   memory; forget is a message asking it to erase the content.
>
> Questions:
> 1. Are the default agent settings (model `openai/gpt-4o-mini`, embedding
>    `openai/text-embedding-3-small`, empty `human` / `persona` blocks) a fair
>    default? Would you use a different agent type or sleep-time agents?
> 2. Is a message the right way to ask the agent to forget something?
> 3. If you would tune anything, could you propose a config? It is published
>    next to the default, labelled "vendor-tuned".

### LangMem (`langchain-ai/langmem`)

**Title:** Review request: LangMem memory-store manager adapter in MemBench

> Hi LangMem maintainers. MemBench (https://github.com/imthath-m/membench)
> has added a LangMem adapter (`membench/adapters/langmem.py`; docs in
> `docs/langmem.md`) and would like your review before we publish numbers.
>
> What it does: `create_memory_store_manager("openai:gpt-4o-mini",
> namespace=("membench", "{langgraph_user_id}"), store=InMemoryStore(index=
> {"dims": 1536, "embed": "openai:text-embedding-3-small"}))`, invoked once
> per event; recall is `store.search(namespace, query=..., limit=3)`.
>
> Questions:
> 1. Is one `invoke` per event right, or would you batch a session's turns?
> 2. Would you enable `enable_deletes`, a `query_model`, or custom
>    instructions / schemas for a fair default?
> 3. If you would tune anything, could you propose a config? It is published
>    next to the default, labelled "vendor-tuned".

### Cognee (`topoteretes/cognee`)

**Title:** Review request: Cognee adapter and default config in MemBench

> Hi Cognee team. MemBench (https://github.com/imthath-m/membench) has added
> a Cognee adapter (`membench/adapters/cognee.py`; docs in `docs/cognee.md`)
> and would like your review before we publish numbers.
>
> What it does: `cognee.add` per event into a dataset per episode;
> `cognee.cognify(datasets=[...])` before each query that follows writes;
> recall is `cognee.search(query_type=SearchType.CHUNKS, top_k=3)` (other
> search types run with `only_context=True`); forget is
> `cognee.forget(data_id=..., dataset=...)`; reset is `prune_data()` +
> `prune_system(metadata=True)`.
>
> Questions:
> 1. Which search type would you choose as the default for agent memory
>    (`CHUNKS`, `GRAPH_COMPLETION` context, `HYBRID_COMPLETION`,
>    `TEMPORAL`)? We default to `CHUNKS` because its results trace back to
>    the source data item.
> 2. Should we move to the 1.0 API (`remember` / `recall` / `forget`)?
> 3. If you would tune anything, could you propose a config? It is published
>    next to the default, labelled "vendor-tuned".
