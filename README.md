# MemBench

An eval harness for agent memory backends. MemBench replays a scripted
scenario ("pack") — facts stated, updated, contradicted, and queried across
sessions — against a memory backend, then scores the result: recall
precision, update correctness, staleness, temporal accuracy, forget
compliance, evidence recall@k, abstention, semantic recall, contradiction
rate, QA accuracy (optional reader stage), token cost and per-stage latency,
overall and per query category.

Everything runs offline except the optional LLM judge (provider-agnostic;
Claude by default).

## Install

Not published to PyPI yet; install from GitHub:

```bash
pip install "membench @ git+https://github.com/imthath-m/membench"

# optional backends
pip install "membench[chroma] @ git+https://github.com/imthath-m/membench"   # local vector-store baseline
pip install "membench[mem0] @ git+https://github.com/imthath-m/membench"     # Mem0 OSS backend
# also: mem0-platform, zep, letta, langmem, cognee (see "Backends")
pip install "membench[tokens] @ git+https://github.com/imthath-m/membench"   # exact token counts (tiktoken)
```

## Quickstart

Run a shipped pack against the full-context baseline (no memory system —
just replays the whole event log every query):

```bash
membench run --backend full-context --pack customer-support --out report.json
```

`--pack` takes a builtin pack name, a public dataset or a synthetic pack
(see `membench list` and [Datasets](#datasets-and-synthetic-packs)), or a
path to your own pack YAML. Run without the LLM judge (fully offline, no API key):

```bash
membench run --backend full-context --pack customer-support --no-judge --out report.json
```

Render the JSON report as a human-readable scorecard:

```bash
membench report --in report.json --format md --out report.md
membench report --in report.json --format html --out report.html

# compare two runs side by side
membench report --in report.json --in other_report.json --format html --out compare.html

# is the difference real? paired test on the same queries
membench compare --a report.json --b other_report.json --out compare.md
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
Judge responses are cached on disk (`--judge-cache`) keyed by (model,
`JUDGE_PROMPT_VERSION`, prompt hash), with the raw reply and its token
`usage`, so CI can replay scored runs without network access.
`--strict-cache` judges from the cache only (no API key needed) and fails
the run on a cache miss — use it to replay published results. Rate-limit (429)
and 5xx responses are retried with exponential backoff, honouring
`Retry-After`. Without an API
key, `--judge` degrades gracefully: the judge metrics score `null` and
everything else still runs.

### Judge validation

Judge metrics are **`unvalidated`** in every report until human labels show
the judge agrees with people: reports mark a judge metric validated only when
the committed `validation/agreement-v<JUDGE_PROMPT_VERSION>.json` passes the
run's judge model on it (human–human κ ≥ 0.7, judge–human κ ≥ 0.6 and within
0.1 of it, n ≥ 200). No labels are committed yet, so there are no agreement
numbers to cite. Workflow, file formats and the prompt-versioning policy:
[docs/judge-validation.md](docs/judge-validation.md); annotator rules:
[docs/labelling-guidelines.md](docs/labelling-guidelines.md).

```bash
membench judge-sample --in run-a.json --in run-b.json      # stratified items → validation/sample.jsonl
membench judge-labels --annotator a=a.jsonl --annotator b=b.jsonl --adjudication adj.jsonl
membench judge-cross --judges configs/judges.yaml          # Claude, GPT, open-weight (needs keys)
membench judge-agreement --labels validation/labels.jsonl --verdicts validation/verdicts-v1.jsonl
```

### Reader stage (QA accuracy)

`--reader-model` adds a reader: an LLM answers each query that has a gold
`answer` (or `abstain: true`) from the recalled context and the query's
date, and the judge compares its reply to `answer` (`qa_accuracy`, the
number comparable to published LongMemEval / LoCoMo results). The reply is
stored on the trace (`detail.reader_answer`), so `rescore` replays it
offline. Same client config as the judge: `--reader-base-url` for an
OpenAI-compatible endpoint (omit for Claude); key from
`MEMBENCH_READER_API_KEY`, else `MEMBENCH_JUDGE_API_KEY`. Without a key the
run warns and `qa_accuracy` / `abstention` score `null`.

```bash
membench run --backend mem0 --pack customer-support --reader-model claude-haiku-4-5
```

### Backends

See `membench list` for the current set.

| Backend | Kind | Extra | Docs |
|---|---|---|---|
| `full-context` | baseline: the whole timestamped log | — | — |
| `bm25` | baseline: lexical floor (Okapi BM25) | — | [docs/bm25.md](docs/bm25.md) |
| `oracle` | baseline: only the query's `evidence` (upper bound) | — | [docs/oracle.md](docs/oracle.md) |
| `chroma` | local vector store | `chroma` | — |
| `mem0` | Mem0 OSS (local) | `mem0` | [docs/mem0.md](docs/mem0.md) |
| `mem0-platform` | Mem0 Platform (hosted) | `mem0-platform` | [docs/mem0-platform.md](docs/mem0-platform.md) |
| `zep` | Zep Cloud (hosted) | `zep` | [docs/zep.md](docs/zep.md) |
| `letta` | Letta archival passages + core blocks | `letta` | [docs/letta.md](docs/letta.md) |
| `letta-agent` | Letta agent managing its own memory | `letta` | [docs/letta-agent.md](docs/letta-agent.md) |
| `langmem` | LangMem memory-store manager | `langmem` | [docs/langmem.md](docs/langmem.md) |
| `cognee` | Cognee add / cognify / search | `cognee` | [docs/cognee.md](docs/cognee.md) |

Vendor backends run with their vendors' defaults (`configs/default/`);
vendor-tuned configs are published next to them once reviewed
([docs/vendor-review.md](docs/vendor-review.md)). `oracle` is the one adapter
that sees ground truth (the query's `evidence`), by design: report it as an
upper bound, never as a competitor.

Live runs of the key-gated backends go through the manual
`.github/workflows/live.yml` (`gh workflow run live.yml -f backend=zep -f
pack=mini`, or `-f backend=all`); it needs the backend's repository secret
(`MEM0_API_KEY`, `ZEP_API_KEY`, `LETTA_API_KEY`, or `OPENAI_API_KEY` for
LangMem / Cognee) plus `MEMBENCH_JUDGE_API_KEY`. Locally:
`MEMBENCH_LIVE=1 pytest tests/test_live_backends.py` runs `mini` on every
backend whose key is set.

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
  category: single_hop       # single_hop | multi_hop | temporal | update | abstention | forget | …
  answer: "Blue"             # gold answer the reader's reply is judged against
  expect_absent: ["green"]   # stale / forgotten strings the context must not contain
  evidence: [color-1]        # ids of the earlier writes that answer it
  # abstain: true            # no answer exists; the reader should say it doesn't know
  #                          # (an `answer` here only explains why, for the LongMemEval judge)
- type: forget               # adapter.forget([...]); targets earlier writes by id
  session_id: s1
  target: color-1            # an id or a list of ids
```

Any step may set `timestamp` (ISO 8601; quote it in YAML); a query's
timestamp is passed to `recall()` as `at`. Without one, steps default to one
second apart from 2024-01-01; the bundled packs set realistic timestamps
(days to months apart). A `category: forget` query must follow a forget
step in its episode; `evidence` ids must be earlier writes in the episode.

### Datasets and synthetic packs

`--pack` also takes a public dataset or a synthetic pack (`membench list`).
Both are built in memory; nothing is written to disk or committed.

| Pack | What | Size |
|---|---|---|
| `longmemeval-s` / `-m` / `-oracle` | [LongMemEval](https://github.com/xiaowu0162/LongMemEval) (MIT), cleaned release | 500 questions, 1 episode each |
| `locomo` | [LoCoMo](https://github.com/snap-research/locomo) (CC BY-NC 4.0) | 10 conversations, 1,986 questions |
| `tool-call-memory`, `document-chain`, `multi-step-decisions`, `customer-support-xl`, `coding-assistant-xl` | synthetic generator v2 | 216 queries, 36 per category |

- **Download:** a dataset is fetched on first use from a pinned revision,
  its sha256 checked, and cached under `~/.cache/membench/datasets/`
  (`MEMBENCH_CACHE_DIR` overrides `~/.cache/membench`). The manifest records
  `dataset` (id, revision, sha256, options), so `membench rescore` rebuilds
  the same pack; it needs the cached file (or network to fetch it again).
- **LongMemEval:** each haystack turn is a `write` at its session date, the
  question a `query` at `question_date`; `question_type` → `category`,
  `_abs` → `abstain`, `has_answer` turns → `evidence`. Ids are opaque (no
  ground truth reaches the adapter). The reader and QA judge use the
  official LongMemEval prompts (`qa_style: longmemeval` in the manifest), so
  `qa_accuracy` (abstention questions included) is comparable to the paper.
  `-m` is a 2.7 GB file; loading it needs a lot of RAM.
- **LoCoMo:** dialogue ids → `Event.id` (`conv-26/D1:3`), session dates →
  timestamps, QA `evidence` → `evidence`, categories → `multi_hop`,
  `temporal`, `open_domain`, `single_hop`, `adversarial` (→ `abstain`).
  `membench/datasets/locomo_exclusions.yaml` lists items with broken gold
  annotations, with reasons. Score a run with and without them:
  `membench rescore --in report.json --exclude-flagged` (or run with
  `--exclude-flagged`). Non-commercial license: never redistribute raw or
  converted files.
- **Subsets:** `--limit N` keeps the first N queries; `--sample N` a random
  N stratified by category, seeded by `--seed`. Episodes left without a
  query are dropped (their writes too). Recorded as `manifest.pack.subset`.
- **Synthetic:** a deterministic, template-based generator (no LLM): per
  simulated user, facts spread over 4 sessions days to weeks apart with
  filler turns, an update chain A→B→C, a multi-hop pair, a temporal "as of"
  question, an unstated fact (abstention) and a forgotten one. Answerable
  queries carry a `semantic_recall` judge rubric, update queries a
  `contradiction` rubric (generator v2). Same seed, same bytes; `--pack-seed` picks another draw. Knobs: `generate/synthetic.py`
  `Knobs`; templates: `generate/domains.py`.
- **Inspect:** `membench convert --pack locomo --out locomo.yaml` writes any
  pack as YAML (local inspection only). `--paraphrase-model <model>` rewrites
  each query with an LLM so it shares fewer words with its answer turn
  (needs `MEMBENCH_JUDGE_API_KEY`); run the result as a file pack.

```bash
membench run --backend chroma --pack longmemeval-s --sample 50 --seed 1 --no-judge
membench run --backend full-context --pack longmemeval-oracle --reader-model gpt-4o \
  --reader-base-url https://api.openai.com/v1 --judge-model gpt-4o --judge-base-url https://api.openai.com/v1
```

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

All of these appear in every JSON report's `scores` block; `null` means the
pack (or the run config) didn't exercise that metric. `by_category` repeats
the query-level metrics (`n`, `recall_precision`, `evidence_recall_at_k`,
`semantic_recall`, `qa_accuracy`) per query `category` ("uncategorized"
when unset); `membench report` renders it as a table.

- **`recall_precision`** — fraction of queries with an `expect_contains`
  assertion whose recalled context contains every expected substring
  (case-insensitive, exact match). Measures whether the backend surfaces the
  right facts at all.
- **`update_correctness`** — fraction of queries, made after a fact was
  updated or contradicted, whose recalled context reflects the *new* value
  rather than the stale one. Measures whether the backend tracks fact
  supersession correctly.
- **`staleness`** — of the post-update queries above that declare
  `expect_absent`, the fraction whose context still contains a stale value.
  Lower is better. Full-context passes `update_correctness` but scores 100%
  here.
- **`temporal_accuracy`** — fraction of `category: temporal` queries whose
  context contains every `expect_contains` string (e.g. the value that held
  at an earlier date, or the date a change happened).
- **`forget_compliance`** — fraction of `category: forget` queries whose
  context contains none of their `expect_absent` strings and whose recalled
  items trace back to no forgotten event id.
- **`evidence_recall_at_k`** — for queries with `evidence`, the mean
  fraction of evidence event ids among the recalled items'
  `event_ids` (k = the items the backend returned, its own top-k).
- **`abstention`** *(reader)* — fraction of `abstain: true` queries where the
  reader replied that it doesn't know.
- **`qa_accuracy`** *(reader + judge)* — fraction of queries with a gold
  `answer` whose reader reply the judge finds equivalent.
- **`token_cost`** — mean `token_count` of the injected context across all
  query steps (the runner counts each `RecallResult.context`, i.e. exactly
  what would be billed as prompt tokens). Lower is cheaper. `membench run`
  counts with tiktoken `o200k_base` when `membench[tokens]` is installed
  (vocab cached under `~/.cache/membench/tiktoken`), else a whitespace word
  count; `manifest.tokenizer` names it (`tiktoken:o200k_base` or
  `whitespace:approx`) and reports mark the latter `approx`.
- **`write_p50_ms` / `write_p95_ms`, `ingest_p50_ms` / `ingest_p95_ms`,
  `recall_p50_ms` / `recall_p95_ms`** — nearest-rank latency percentiles per
  stage over successful steps: `write()`, `flush()` before a query (async
  ingest), `recall()`. p95 is `null` (`n/a`) below 100 samples.
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

Every score also carries n and a 95% CI in the report's `stats` block —
see [Statistics](#statistics).

Deterministic metrics (everything not marked judge or reader) always run
offline. The judge metrics need `--judge` (default on) and
`MEMBENCH_JUDGE_API_KEY`; with `--no-judge` or no key, they score `null`.
`abstention` needs the reader; `qa_accuracy` needs both.

### Statistics

- **Per-query outcomes.** `outcomes` holds each query-level metric's
  per-query result (`{metric: {step_index: true/false or number}}`); every
  score is their mean.
- **95% CIs.** `stats.scores[metric]` and `stats.by_category[cat][metric]`
  hold `{n, ci, method, indicative}`: Wilson intervals for pass/fail rates,
  a seeded percentile bootstrap over queries (10k resamples) for
  `evidence_recall_at_k` and `token_cost`, and Wilson over steps for
  `error_rate`. Latency entries carry n only. The bootstrap is seeded from
  `--seed` + metric + category, so `rescore` reproduces every interval.
- **Minimum n.** Reports label a rate `indicative` below 385 queries
  overall (±5pp at 95%, p = 0.5) or 100 per category (±10pp); p95 latency
  is suppressed below 100 samples per stage.
- **Repeats.** `--repeats N` runs the pack N times, each with a new backend
  namespace (`config["namespace"]` = `membench-r<i>`, which Chroma uses as
  its collection and Mem0 as its `user_id` unless you set those explicitly)
  and a fresh `reset()`. Default: 3 when the backend (`uses_llm`), judge
  or reader calls an LLM, else 1. The top-level report is repeat 0;
  `repeats` holds the full reports of the others and `repeat_summary` each
  score's mean, run-to-run SD and per-repeat values.
- **Paired comparison.** `membench compare --a A.json --b B.json` pairs the
  two runs' outcomes by query (overall and per category): B − A with a
  paired bootstrap 95% CI, McNemar's exact test for pass/fail metrics
  (bootstrap p otherwise), and Holm's correction across all metric ×
  category rows. A row is a real difference only when its Holm p < 0.05 and
  its CI excludes 0; otherwise it says "CI includes 0". It refuses runs on
  different packs/subsets or with different judge/reader models or prompt
  versions. `--out x.json` writes the raw rows, any other path Markdown.

### Reproducibility

Every report carries a `manifest` and validates against
[`membench/report.schema.json`](membench/report.schema.json):

- `schema_version`; membench version, git commit and dirty flag; Python and OS.
- Backend: name, config with secrets redacted (`api_key`, `token`, …), SDK
  versions (`packages`), models and settings from `describe()` (LLM, base
  URL, temperature, embedding model), deployment (`local`/`hosted`) and region.
- Pack name and content sha256 (of the parsed YAML, so formatting doesn't
  change it); dataset id/revision/sha256 (`null` until dataset loaders land).
- Judge model, base URL and prompt version; reader model, base URL and
  prompt version (`null` without `--reader-model`); tokenizer name; `--seed`;
  repeat index; start/finish times.

Re-score a report from its traces and the judge cache — no adapter, no
network, no API key:

```bash
membench rescore --in report.json --judge-cache .membench_judge_cache.json --check
```

It reloads the pack (`--pack`, default: the manifest's pack name), refuses
one whose sha256 differs from the run's, replays the judge with a strict
cache, and prints whether the scores, outcomes and CIs (of every repeat)
are identical (`--check` exits 1 if not; `--out` writes the re-scored
report). CI re-scores a committed report
(`tests/fixtures/`) on every push.

Published runs are made from a frozen sync of `uv.lock`
(`uv sync --frozen --extra mem0`, as `.github/workflows/live.yml` does);
`pyproject.toml` keeps floor versions for library users.

**Determinism.** `--seed` (default 0) seeds Python's `random` and numpy (when
a backend loaded it). Deterministic: the pack, traces → scores (bit-identical
on rescore), the judge cache, Chroma's local ONNX embedder
(`all-MiniLM-L6-v2`). Not deterministic: backend LLM extraction (Mem0's LLM
runs at its configured `temperature`, recorded in the manifest; OpenAI
offers no hard seed guarantee), the judge on a cache miss (temperature 0 is
not bit-stable across calls or model updates — the cache is what makes
replays exact), approximate vector search ordering on near-ties, and latency.

### Cost

The report's `cost` section prices API-reported token counts against a dated
table, [`membench/prices.yaml`](membench/prices.yaml) (USD per 1M tokens):

- `judge`: this run's judge calls, cache hits and input/output tokens (cache
  hits cost nothing; a fully cached replay reports `$0`).
- `backend`: `adapter.usage()` (Mem0: fact-extraction LLM and embedding
  tokens, counted from the OpenAI SDK's `usage`), priced by the models its
  `describe()` names.
- `reader`: the reader's calls and input/output tokens (`null` without one).
- `ingest_tokens_per_1k_writes`: backend tokens spent outside `recall()`
  (extraction + embedding at write/flush time) per 1000 writes.
- `total_usd` and `usd_per_1k_queries`. A model missing from the table gets
  `usd: null`, and so does the total. Re-check the table against the
  providers' pricing pages before publishing a cost.

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
    packages: tuple[str, ...] = ()                        # SDK dists recorded in the manifest
    deployment: str = "local"                             # "local" | "hosted" (region: config)
    uses_llm: bool = False                                # LLM-backed: default --repeats 3
    namespace: str                                        # config["namespace"], new per repeat
    def __init__(self, config: dict | None = None): ...   # plain dict from pack/CLI config
    def write(self, event: Event) -> None: ...             # ingest one fact/turn/tool-result
    def recall(self, query: str, *, session_id: str, at: str) -> RecallResult: ...
    def forget(self, event_ids: list[str]) -> None: ...    # remove events by Event.id
    def flush(self) -> None: ...                           # optional: wait for async ingest
    def reset(self) -> None: ...                           # optional: clear persisted state
    def usage(self) -> dict[str, int] | None: ...          # optional: backend LLM/embed tokens
    def describe(self) -> dict | None: ...                 # optional: models + settings
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
  episodes. With `--repeats`, each repeat gets a new adapter whose
  `namespace` (`config["namespace"]`) is new; keep persistent state under it.
- `usage()` (default `None`) is read once at the end and stored as the
  report's `usage`. Keys: `llm_input_tokens`, `llm_output_tokens`,
  `embedding_tokens`, priced against `describe()`'s `llm_model` /
  `embedding_model` in the `cost` section. A backend calling an
  `openai.OpenAI` client can hand it to `membench.core.usage.meter_openai`.
- `describe()` (default `None`) names what shapes results (`llm_model`,
  `llm_base_url`, `temperature`, `embedding_model`, …) for the manifest.
- Map backend ids back to `Event.id` in `RecalledItem.event_ids`.
- A backend that ingests asynchronously polls its jobs in `flush()` with
  `membench.core.poll.drain` (config `poll_interval_s` / `flush_timeout_s`).

`membench/adapters/mem0.py` is the worked example for a real (optional-extra)
vendor backend, in full:

```python
from __future__ import annotations

from typing import Any

from membench.core.adapter import Event, MemoryAdapter, RecalledItem, RecallResult
from membench.core.usage import meter_openai, new_usage

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
    packages = ("mem0ai", "openai")
    uses_llm = True

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if Memory is None:
            raise ImportError(
                "mem0ai is not installed. Install it with "
                "`pip install membench[mem0]` (or `pip install mem0ai`) "
                "to use the mem0 backend."
            )
        super().__init__(config)
        self.user_id = self.config.get("user_id", self.namespace)
        self.top_k = int(self.config.get("top_k", 3))
        mem0_config = self.config.get("mem0_config")
        self._client = Memory.from_config(mem0_config) if mem0_config else Memory()
        self._sources: dict[str, list[str]] = {}  # mem0 memory id -> Event.ids
        self._llm = getattr(self._client, "llm", None)
        self._embedder = getattr(self._client, "embedding_model", None)
        self._usage = new_usage()
        for part in (self._llm, self._embedder):  # extraction + embedding tokens
            meter_openai(getattr(part, "client", None), self._usage)

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

    def usage(self) -> dict[str, int]:
        return dict(self._usage)

    def describe(self) -> dict[str, Any]:
        llm = getattr(self._llm, "config", None)
        base_url = getattr(getattr(self._llm, "client", None), "base_url", None)
        return {
            "llm_model": getattr(llm, "model", None),
            "llm_base_url": str(base_url) if base_url else None,
            "temperature": getattr(llm, "temperature", None),
            "embedding_model": getattr(getattr(self._embedder, "config", None), "model", None),
        }
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
6. **Pass the shared contract suite** (`tests/test_contract_suite.py`): add a
   factory to `FACTORIES` (an SDK fake from `tests/vendor_fakes.py` for a
   hosted backend). It checks write → flush → recall finds the event, forget
   removes it, reset empties, and episodes stay isolated. Add the backend to
   `tests/test_live_backends.py` and the `live.yml` matrix if it needs keys.

That's the whole surface — `write`, `recall`, `forget`, a registry line, and
an import guard. Override `flush()`, `reset()`, `usage()` and `describe()`
when the backend needs them.

## Development

```bash
pip install -e ".[dev,chroma]"   # or exact pins: uv sync --frozen --extra dev --extra chroma
ruff check .
pytest
```

After changing dependencies in `pyproject.toml`, run `uv lock` (CI checks
`uv lock --check`). After changing the report format, bump
`REPORT_SCHEMA_VERSION` (`membench/core/manifest.py`), update
`membench/report.schema.json`, and regenerate the committed fixture with
`python tests/fixtures/make_report.py`. After changing the judge prompt or
verdict parsing, bump `JUDGE_PROMPT_VERSION`.

More docs: per-backend pages (see "Backends"), [docs/vendor-review.md](docs/vendor-review.md),
[docs/generate.md](docs/generate.md) (transcript-to-pack generator).
