# MemBench

An eval harness for agent memory backends. MemBench replays a scripted
scenario ("pack") — facts stated, updated, contradicted, and queried across
sessions — against a memory backend, then scores the result on five metrics:
recall precision, update correctness, token cost, p95 latency, semantic
recall, and contradiction rate.

Everything runs offline except the optional LLM judge (provider-agnostic;
Claude by default).

## Install

```bash
pip install membench

# optional backends
pip install membench[chroma]   # local vector-store baseline
pip install membench[mem0]     # Mem0 OSS backend
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
hash) so CI can replay scored runs without network access. Without an API
key, `--judge` degrades gracefully: the two judge metrics score `null` and
everything else still runs.

### Backends

See `membench list` for the current set. Shipped: `full-context` (baseline,
no vendor dep), `chroma` (local vector store, no vendor dep), `mem0`
(`membench[mem0]`, see [docs/mem0.md](docs/mem0.md)).

## Metric definitions

All five appear in every JSON report's `scores` block; `null` means the
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
  query steps (from `RecallResult.token_count`, i.e. exactly what would be
  billed as prompt tokens). Lower is cheaper.
- **`p95_latency_ms`** — 95th-percentile latency in milliseconds across every
  step (write + query), nearest-rank method. Captures tail latency, not just
  the average.
- **`semantic_recall`** *(judge)* — fraction of queries with a
  `judge.semantic_recall` rubric where the judge agrees the recalled context
  semantically conveys the expected fact (catches paraphrase/rewording that
  exact-match would miss). Requires the judge.
- **`contradiction_rate`** *(judge)* — fraction of queries with a
  `judge.contradiction` rubric where the judge finds the recalled context
  contains contradictory statements about the same fact (e.g. a stale value
  surfaced alongside its update). Lower is better. Requires the judge.

Deterministic metrics (`recall_precision`, `update_correctness`,
`token_cost`, `p95_latency_ms`) always run offline. The two judge metrics
need `--judge` (default on) and `MEMBENCH_JUDGE_API_KEY`; with `--no-judge`
or no key, they score `null`.

## Writing a new adapter

A backend adapter is a thin, stateless config wrapper — the runner owns
timing, token accounting, and scoring. **Budget: under 100 lines.** The
whole contract is `membench/core/adapter.py`:

```python
class Event:
    session_id: str
    key: str            # stable id — forget() and update-correctness target this
    role: str
    content: str
    timestamp: str
    metadata: dict

class RecallResult:
    context: str         # exactly what would be injected into a prompt
                          # (this is what the token_cost metric measures)
    items: list
    token_count: int

class MemoryAdapter(ABC):
    def __init__(self, config: dict | None = None): ...  # plain dict from pack/CLI config
    def write(self, event: Event) -> None: ...            # ingest one fact/turn/tool-result
    def recall(self, query: str) -> RecallResult: ...     # return injected context + items
    def forget(self, key: str) -> None: ...               # remove by stable key
```

`membench/adapters/mem0.py` is the worked example for a real (optional-extra)
vendor backend, in full:

```python
from __future__ import annotations

from typing import Any

from membench.core.adapter import Event, MemoryAdapter, RecallResult
from membench.core.tokens import count_tokens

try:
    from mem0 import Memory
except ImportError:  # pragma: no cover - exercised only when mem0 missing
    Memory = None


class Mem0Adapter(MemoryAdapter):
    """Mem0 OSS baseline backend."""

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
        self._ids: dict[str, str] = {}

    def write(self, event: Event) -> None:
        result = self._client.add(
            event.content, user_id=self.user_id, metadata={"key": event.key}
        )
        memories = result.get("results", []) if isinstance(result, dict) else result
        if memories:
            self._ids[event.key] = memories[0]["id"]

    def recall(self, query: str) -> RecallResult:
        result = self._client.search(query, user_id=self.user_id, limit=self.top_k)
        memories = result.get("results", []) if isinstance(result, dict) else result
        context = "\n".join(m["memory"] for m in memories)
        return RecallResult(
            context=context,
            items=memories,
            token_count=count_tokens(context),
        )

    def forget(self, key: str) -> None:
        memory_id = self._ids.pop(key, None)
        if memory_id is not None:
            self._client.delete(memory_id)
```

Things to note from that example, which apply to any new adapter:

1. **Import-guard optional dependencies.** Wrap the vendor SDK import in
   `try/except ImportError` at module scope, and raise a clear, actionable
   `ImportError` from `__init__` (not a bare traceback) telling the user
   which extra to install.
2. **No scoring logic inside the adapter.** Token counting uses the shared
   `core.tokens.count_tokens` helper; latency, precision, etc. are all
   computed by the runner/scoring modules from what `recall()` returns.
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
an import guard.

## Development

```bash
pip install -e ".[dev,chroma]"
ruff check .
pytest
```

More docs: [docs/mem0.md](docs/mem0.md) (Mem0 backend config),
[docs/generate.md](docs/generate.md) (transcript-to-pack generator).
