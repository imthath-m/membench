# MemBench

An eval harness for agent memory backends. MemBench replays a scripted
scenario ("pack") — facts stated, updated, contradicted, and queried across
sessions — against a memory backend, then scores the result on six metrics:
recall precision, update correctness, token cost, p95 latency, semantic
recall, and contradiction rate.

Everything runs offline except the optional LLM judge (provider-agnostic;
Claude by default).

## Install

Not published to PyPI yet; install from GitHub:

```bash
pip install "membench @ git+https://github.com/imthath-m/membench"

# optional backends
pip install "membench[chroma] @ git+https://github.com/imthath-m/membench"   # local vector-store baseline
pip install "membench[mem0] @ git+https://github.com/imthath-m/membench"     # Mem0 OSS backend
```

## Quickstart

Run a shipped pack against the full-context baseline (no memory system —
just replays the whole event log every query):

```bash
membench run --backend full-context --pack customer-support --out report.json
```

`--pack` takes a builtin pack name (see `membench list`) or a path to your
own pack YAML. Run without the LLM judge (fully offline, no API key):

```bash
membench run --backend full-context --pack customer-support --no-judge --out report.json
```

Render the JSON report as a human-readable scorecard:

```bash
membench report --in report.json --format md --out report.md
membench report --in report.json --format html --out report.html

# compare two runs side by side
membench report --in report.json --in other_report.json --format html --out compare.html
```

List available packs and backends:

```bash
membench list
```

Generate a new scenario pack from your own transcripts (needs an LLM — see
[docs/generate.md](docs/generate.md)):

```bash
membench generate --transcripts transcript.txt --out pack.yaml
```

### Judge configuration

The LLM judge powers the two semantic metrics (semantic recall, contradiction
rate). It's pluggable, mirroring the backend-adapter pattern:

```bash
export MEMBENCH_JUDGE_API_KEY=sk-...
membench run --backend full-context --pack customer-support \
  --judge-model claude-sonnet-5 \
  --judge-base-url https://api.openai.com/v1   # omit for Anthropic/Claude default
```

Works with Anthropic, OpenAI, Gemini, or any OpenAI-compatible endpoint.
Judge responses are cached on disk (`--judge-cache`) keyed by (model, prompt
hash) so CI can replay scored runs without network access. Rate-limit (429)
and 5xx responses are retried with exponential backoff, honouring
`Retry-After`. Without an API
key, `--judge` degrades gracefully: the two judge metrics score `null` and
everything else still runs.

### Backends

See `membench list` for the current set. Shipped: `full-context` (baseline,
no vendor dep), `chroma` (local vector store, no vendor dep), `mem0`
(`membench[mem0]`, see [docs/mem0.md](docs/mem0.md)).

Pass adapter config with `--backend-config path.yaml` (JSON or YAML mapping,
merged over `{"persist_dir": --persist-dir}`). Each run starts clean: the
adapter's `reset()` clears state left by earlier runs (chroma collection,
mem0 `user_id`); pass `--no-reset` to keep it. If the backend can't be set up
(e.g. its extra isn't installed), `membench run` prints a one-line `Error:`
and exits with code 1.

### Packs

A pack is YAML: `steps:`, or `episodes:` (a list of `{steps: [...]}`) for
independent scenarios with a `reset()` between them. Step types:

```yaml
- type: write                # ingest a fact
  session_id: s1
  id: color-1                # optional opaque Event.id (auto-generated if omitted)
  fact_key: fact_color       # scoring only, never sent to the adapter
  role: user
  content: "My favorite color is blue."
  fact_type: state           # state | update | contradict
- type: query
  session_id: s1
  query: "What is my favorite color?"
  expect_contains: ["blue"]
  expect_key: fact_color
- type: forget               # adapter.forget([...]); targets earlier writes by id
  session_id: s1
  target: color-1            # an id or a list of ids
```

Any step may set `timestamp` (ISO 8601); a query's timestamp is passed to
`recall()` as `at`.

### Failures during a run

A run never stops on a single bad step:

- Adapter calls that raise one of the adapter's `retryable` exceptions
  (e.g. Mem0's OpenAI rate-limit / timeout errors) are retried with
  exponential backoff and jitter; each retry is logged and counted in the
  step's trace (`detail.retries`).
- Each write/flush/recall/forget is capped by `--step-timeout` (default 120s, retries
  included).
- A step that still fails or times out is recorded in its trace's `error`
  and the run continues. A failed query recalls nothing, so it counts as a
  miss for accuracy metrics; it's excluded from `token_cost` and latency.
  The report's `error_rate` is the fraction of failed steps; above 10% the
  run is marked `"valid": false`.
- A judge reply that fails to parse is retried once; if it still fails, or
  the response has an unexpected shape, that query's trace records a
  `judge_error` and it's left out of that judge metric.

## Metric definitions

All six, plus `error_rate`, appear in every JSON report's `scores` block; `null` means the
pack (or the run config) didn't exercise that metric.

- **`recall_precision`** — fraction of queries with an `expect_contains`
  assertion whose recalled context contains every expected substring
  (case-insensitive, exact match). Measures whether the backend surfaces the
  right facts at all.
- **`update_correctness`** — fraction of queries, made after a fact was
  updated or contradicted, whose recalled context reflects the *new* value
  rather than the stale one. Measures whether the backend tracks fact
  supersession correctly.
- **`token_cost`** — mean `token_count` of the injected context across all
  query steps (the runner counts each `RecallResult.context`, i.e. exactly
  what would be billed as prompt tokens). Lower is cheaper.
- **`p95_latency_ms`** — 95th-percentile latency in milliseconds across every
  successful step (write + query + forget), nearest-rank method. Captures tail latency, not just
  the average.
- **`semantic_recall`** *(judge)* — fraction of queries with a
  `judge.semantic_recall` rubric where the judge agrees the recalled context
  semantically conveys the expected fact (catches paraphrase/rewording that
  exact-match would miss). Requires the judge.
- **`contradiction_rate`** *(judge)* — fraction of queries with a
  `judge.contradiction` rubric where the judge finds the recalled context
  contains contradictory statements about the same fact (e.g. a stale value
  surfaced alongside its update). Lower is better. Requires the judge.
- **`error_rate`** — fraction of steps (write + query + forget) whose adapter call
  failed or timed out. Above 10% the run is marked `"valid": false`.

Deterministic metrics (`recall_precision`, `update_correctness`,
`token_cost`, `p95_latency_ms`) always run offline. The two judge metrics
need `--judge` (default on) and `MEMBENCH_JUDGE_API_KEY`; with `--no-judge`
or no key, they score `null`.

## Writing a new adapter

A backend adapter is a thin, stateless config wrapper — the runner owns
timing, retries, token accounting, and scoring. **Budget: under 100 lines**
(enforced by a test). The whole contract (v2, since 0.2) is
`membench/core/adapter.py`:

```python
class Event:
    id: str             # unique opaque id per write (an update gets a new one)
    session_id: str
    role: str
    content: str
    timestamp: str
    metadata: dict

class RecalledItem:
    event_ids: list[str] # the Event.ids this item came from
    text: str
    score: float | None  # backend relevance, higher is better

class RecallResult:
    context: str         # exactly what would be injected into a prompt
                         # (the runner counts its tokens for token_cost)
    items: list[RecalledItem]

class MemoryAdapter(ABC):
    retryable: tuple[type[BaseException], ...] = ()       # transient errors to retry
    def __init__(self, config: dict | None = None): ...   # plain dict from pack/CLI config
    def write(self, event: Event) -> None: ...             # ingest one fact/turn/tool-result
    def recall(self, query: str, *, session_id: str, at: str) -> RecallResult: ...
    def forget(self, event_ids: list[str]) -> None: ...    # remove events by Event.id
    def flush(self) -> None: ...                           # optional: wait for async ingest
    def reset(self) -> None: ...                           # optional: clear persisted state
    def usage(self) -> dict[str, int] | None: ...          # optional: backend LLM/embed tokens
```

What the runner guarantees:

- **Adapters never see scoring ground truth.** `Event.id` is the only id an
  adapter gets; the pack's `fact_key`, expected answers and judge rubrics stay
  with the scorer. An update is a new event with its own id, so only the
  backend can decide which value is current.
- `recall()` gets the query's `session_id` and its time `at` (ISO 8601), for
  session-scoped or temporal backends; others may ignore them.
- `flush()` (default no-op) runs before a query that follows writes or
  forgets; its time is traced as `ingest_latency_ms`, separate from recall
  latency.
- `reset()` runs before the run (unless `--no-reset`) and between a pack's
  episodes.
- `usage()` (default `None`) is read once at the end and stored as the
  report's `usage`.
- Map backend ids back to `Event.id` in `RecalledItem.event_ids`.

`membench/adapters/mem0.py` is the worked example for a real (optional-extra)
vendor backend, in full:

```python
from __future__ import annotations

from typing import Any

from membench.core.adapter import Event, MemoryAdapter, RecalledItem, RecallResult

try:
    from mem0 import Memory
except ImportError:  # pragma: no cover - exercised only when mem0 missing
    Memory = None

try:  # Mem0's default LLM/embedder calls go through the OpenAI SDK.
    import openai

    _RETRYABLE: tuple[type[BaseException], ...] = (
        openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError
    )
except ImportError:  # pragma: no cover - exercised only when openai missing
    _RETRYABLE = ()


class Mem0Adapter(MemoryAdapter):
    """Mem0 OSS baseline backend."""

    retryable = _RETRYABLE

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if Memory is None:
            raise ImportError(
                "mem0ai is not installed. Install it with "
                "`pip install membench[mem0]` (or `pip install mem0ai`) "
                "to use the mem0 backend."
            )
        super().__init__(config)
        self.user_id = self.config.get("user_id", "membench")
        self.top_k = int(self.config.get("top_k", 3))
        mem0_config = self.config.get("mem0_config")
        self._client = Memory.from_config(mem0_config) if mem0_config else Memory()
        self._sources: dict[str, list[str]] = {}  # mem0 memory id -> Event.ids

    def write(self, event: Event) -> None:
        result = self._client.add(event.content, user_id=self.user_id)
        memories = result.get("results", []) if isinstance(result, dict) else result
        for memory in memories or []:  # one add may create or update several memories
            self._sources.setdefault(memory["id"], []).append(event.id)

    def recall(self, query: str, *, session_id: str, at: str) -> RecallResult:
        result = self._client.search(
            query, filters={"user_id": self.user_id}, top_k=self.top_k
        )
        memories = result.get("results", []) if isinstance(result, dict) else result
        items = [
            RecalledItem(self._sources.get(m["id"], []), m["memory"], m.get("score"))
            for m in memories
        ]
        return RecallResult(context="\n".join(i.text for i in items), items=items)

    def forget(self, event_ids: list[str]) -> None:
        """Delete every memory derived from a target event."""
        targets = set(event_ids)
        for memory_id, sources in list(self._sources.items()):
            if targets.intersection(sources):
                self._client.delete(memory_id)
                del self._sources[memory_id]

    def reset(self) -> None:
        self._client.delete_all(user_id=self.user_id)
        self._sources.clear()
```

Things to note from that example, which apply to any new adapter:

1. **Import-guard optional dependencies.** Wrap the vendor SDK import in
   `try/except ImportError` at module scope, and raise a clear, actionable
   `ImportError` from `__init__` telling the user which extra to install;
   `membench run` prints it as a one-line error (exit code 1), not a
   traceback.
2. **No scoring logic inside the adapter.** Token counting, latency,
   precision, etc. are all computed by the runner/scoring modules from what
   `recall()` returns.
3. **`config` is a plain dict**, passed straight through from the CLI/pack —
   don't invent a custom config object.
4. **Register it** in `membench/core/registry.py`'s `_REGISTRY` dict
   (`"my-backend": "membench.adapters.my_backend.MyBackendAdapter"`) — the
   registry lazily imports the module so a missing optional dependency only
   fails when that backend is actually selected.
5. If the backend needs a real vendor account/API key, add it as an
   `optional-dependencies` extra in `pyproject.toml` and a short doc page
   under `docs/` (see `docs/mem0.md`).

That's the whole surface — `write`, `recall`, `forget`, a registry line, and
an import guard. Override `flush()`, `reset()` and `usage()` when the backend
needs them.

## Development

```bash
pip install -e ".[dev,chroma]"
ruff check .
pytest
```

More docs: [docs/mem0.md](docs/mem0.md) (Mem0 backend config),
[docs/generate.md](docs/generate.md) (transcript-to-pack generator).
