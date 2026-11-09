# Build progress — roadmap Phases 8–16
Resume protocol: list origin branches `claude/gracious-edison-v8eyxs*`; open the highest phase branch; read this file there; continue the first phase not marked done.

| Phase | Branch | PR | Status |
|---|---|---|---|
| 8 | claude/gracious-edison-v8eyxs | #6 | done (live run blocked) |
| 9 | claude/gracious-edison-v8eyxs-phase-9 | #7 | done |
| 10 | claude/gracious-edison-v8eyxs-phase-10 | #8 | done (cost-vs-dashboard blocked) |
| 11 | claude/gracious-edison-v8eyxs-phase-11 | #9 | done (live reader run blocked) |
| 12 | claude/gracious-edison-v8eyxs-phase-12 | — | in progress |

## Phase 8 — Hardening + first live run
Status: done (with blocked items)
- [x] CLI builds adapter + `reset()` inside error handling; one-line error for missing extra; test with `mem0` import blocked; fix README and `docs/mem0.md`
- [x] `core/retry.py`: exponential backoff + jitter; adapters declare `retryable` exception tuple (Mem0: OpenAI rate limit / timeout)
- [x] Per-step timeout `--step-timeout` (default 120s)
- [x] Per-step error capture: trace `error`, run continues, report `error_rate`, run marked invalid above threshold
- [x] Judge: unparseable reply retried once then `judge_error`; unexpected response shape → `judge_error`, no retry
- [x] Tests: malformed judge reply (no JSON, bad JSON), unexpected shape (no `choices`), mem0 blocked import
- [x] CI: `chroma` job
- [x] CI: `live.yml` (manual dispatch, secret-gated) runs `mini` against `mem0` with judge
- Blocked: live `mem0` + judge run of `customer-support` (Done-when 2) — needs `OPENAI_API_KEY` (Mem0's default LLM/embedder) and `MEMBENCH_JUDGE_API_KEY`. Once the repo secrets exist: `gh workflow run live.yml -f pack=customer-support` (or locally: `pip install -e ".[mem0]" && membench run --backend mem0 --pack customer-support --judge --out report.json`). Retry-and-log of an injected 429 is covered by a mocked test (`tests/test_runner_errors.py::test_injected_429_is_retried_and_logged`).
- Notes for next phases:
  - New report fields: `StepTrace.error` (str | None), `detail.retries` on every trace, `RunResult.valid`, `scores.error_rate`; judge failures land in a query trace's `detail.judge_error` as `{metric: message}`.
  - Errored queries: context `""`, so they count as misses for `recall_precision` / `update_correctness`, and as `False` for `semantic_recall` (no judge call); excluded from `contradiction_rate`, `token_cost`, `p95_latency_ms`.
  - `MAX_ERROR_RATE = 0.1` and `DEFAULT_STEP_TIMEOUT_S = 120` live in `core/runner.py` (threshold is a constant, not a CLI flag).
  - Step timeout wraps the whole step, retries and backoff sleeps included; latency also includes backoff sleeps. A timed-out call runs on in an abandoned daemon thread.
  - Judge parse-retry and shape handling live in `CachedJudge`; a bare client's `score()` still raises. HTTP errors after the judge's own 429/5xx retries still stop the run (not in Phase 8 scope).
  - The judge's `_post_with_retry` was left as is (status-code based, honours `Retry-After`); `core/retry.py` is exception based, for adapters.
  - Newer ruff (0.17) enables extra default rules (e.g. BLE001, UP035); CI installs the latest ruff, so lint with a current version.

## Phase 9 — Adapter contract v2 (breaking, bump to 0.2)
Status: done
- [x] Split identity: `Event.id` (unique, opaque) is the only id adapters see; pack `fact_key` used only for scoring (fixes Chroma overwrite leak)
- [x] `recall(query, *, session_id, at)`
- [x] `flush()` (default no-op); runner calls it before a query that follows writes, records `ingest_latency_ms`
- [x] `RecallResult.items` → `list[RecalledItem(event_ids, text, score)]`; adapters map ids back to `Event.id`
- [x] Token counting in the runner (configured tokenizer); `RecallResult.token_count` removed
- [x] Optional `usage()` (cumulative backend LLM/embedding tokens)
- [x] `forget` step type (`target: <event id(s)>`); runner calls `adapter.forget`
- [x] Episodes: pack holds one or more `episodes`, `reset()` between them
- [x] Port all 3 adapters; README documents the contract; version 0.2.0
- [x] Test: every file in `membench/adapters/` < 100 lines
- [x] Test: update-resolution pack (distinct ids, no `fact_key` reaches adapters, Chroma context holds both values) on all 3 adapters
- [x] Test: Chroma `update_correctness` no longer 1.0 by construction (regression)
- [x] Test: runner `forget` step reaches `adapter.forget` with target ids, traced with latency
- [x] Test: pack with a `forget` step and two episodes runs on all 3 adapters
- Mem0 is covered by the fake `mem0.Memory` client only (no `mem0ai` / `OPENAI_API_KEY` here); a live check is `gh workflow run live.yml -f pack=mini` once secrets exist (same block as Phase 8).
- Notes for next phases:
  - Pack YAML: write `key` → `fact_key` (old `key` gives a clear error); optional write `id` (auto `evt-<global step index>`); `forget` step with `target` (id or list, must be an earlier write in the same episode); `episodes: [{steps: [...]}]` or top-level `steps`. `Pack.steps` stays flat; each step and `StepTrace` carry `episode` (int).
  - Contract: `recall(query, *, session_id, at)` (`at` = query step timestamp, ISO str); `forget(event_ids: list[str])` (one call per forget step); `flush()`, `reset()`, `usage()` optional.
  - Runner: `reset()` between episodes (a failed reset errors every step of that episode); `flush()` before a query that follows a write or forget, timed into query `detail.ingest_latency_ms` (None when no flush ran); a flush error fails that query (`error` "flush: ...") and recall is skipped. `run_pack(tokenizer=count_tokens)` counts `context`; no CLI flag yet (Phase 11 adds the real tokenizer). `RunResult.usage` = `adapter.usage()` read once at the end (all 3 adapters return None; Mem0 extraction-token tracking left for Phase 10 cost accounting).
  - Trace detail: write `{event_id, fact_key, fact_type, retries}`; forget `{target, retries}`; query adds `items` (list of `{event_ids, text, score}`) and drops `item_count`. Forget latency is pooled into `p95_latency_ms` and counted in `error_rate`.
  - `update_correctness` scopes updates per episode. Chroma item score is `-distance`; Mem0 maps memory id → source Event.ids (an add may create/update several memories; forgetting an event deletes every memory derived from it).
  - Chroma still scores 1.0 `update_correctness` on the bundled packs at `top_k=3` (both values are surfaced), now earned; the Phase 11 staleness metric is what penalises that.

## Phase 10 — Reproducibility + cost accounting
Status: done (with blocked items)
- [x] Run manifest in every report (schema_version, membench version + git commit/dirty, Python/OS, backend SDK versions, redacted backend config, pack name + sha256, dataset slot, judge/reader/backend models + base URLs, judge prompt version, tokenizer, seed, repeat index, deployment, start/finish)
- [x] `JUDGE_PROMPT_VERSION` in manifest + cache key; cache stores raw response + `usage`
- [x] Judge `--strict-cache` (cache miss is an error)
- [x] `membench rescore --in report.json` (no adapter, no network)
- [x] `uv.lock` + frozen sync for published (live) runs; floor versions kept
- [x] Cost tracking: judge / reader / backend tokens from API `usage`; dated price table → USD per run and per 1k queries (Mem0 `usage()`)
- [x] Determinism: seed, embedding model id + temperature recorded; non-determinism documented
- [x] Committed JSON Schema; every report validates (test)
- [x] CI test: `rescore` on a committed report reproduces identical scores offline
- Blocked: "cost section matches the provider dashboard within 5%" — needs a live run (`OPENAI_API_KEY` for Mem0, `MEMBENCH_JUDGE_API_KEY`). Once the secrets exist: `gh workflow run live.yml -f pack=customer-support`, then compare `live-report.json` `cost.backend` / `cost.judge` token counts and USD against the OpenAI and Anthropic usage dashboards for that window. Before that, re-verify the OpenAI rows in `membench/prices.yaml` (Claude rows were checked 2026-10-06; OpenAI rows were not checked against the live pricing page here).
- Notes for next phases:
  - Report = `asdict(RunResult)` + new `manifest` (see `core/manifest.py`, `REPORT_SCHEMA_VERSION = "1"`) and `cost` (`core/cost.py`). `membench/report.schema.json` has `additionalProperties: false` at the top level, on traces, the manifest and cost: a new report field needs a schema edit (and a `REPORT_SCHEMA_VERSION` bump if breaking), then `python tests/fixtures/make_report.py` to regenerate the committed fixture (`tests/fixtures/report_customer_support.json` + `judge_cache_customer_support.json`). New score keys are allowed (number|null) without a schema edit.
  - Manifest slots left `null` for later phases: `dataset` ({id, revision, sha256}, Phase 12), `reader` ({model, base_url}, Phase 11), `cost.reader` (Phase 11). Tokenizer is recorded as the callable's dotted name (`membench.core.tokens.count_tokens`); Phase 11 should pass a named tokenizer.
  - Adapter contract additions (non-breaking): class attrs `packages` (SDK dists → `manifest.packages`), `deployment` ("local"/"hosted"; region from `config["region"]`); optional `describe()` → `manifest.backend.models` (keys `llm_model`, `llm_base_url`, `temperature`, `embedding_model`). `usage()` keys are `llm_input_tokens`, `llm_output_tokens`, `embedding_tokens`; `core/usage.py:meter_openai(client, totals)` wraps an `openai.OpenAI` client's chat/embedding `create`. Mem0 adapter is now 93 lines (budget 100).
  - Pack `sha256` = sha256 of canonical JSON of the parsed YAML mapping (`Pack.sha256`, set by `parse_pack`). `rescore` resolves the pack from `--pack` or the manifest's pack name (builtin) and refuses a sha mismatch; dataset-backed packs (Phase 12) will need `--pack` or a loader hook there.
  - Judge: `JUDGE_PROMPT_VERSION = "1"` is in the cache key (`{model}:v1:{sha}`), so pre-Phase-10 caches are all misses. Clients have `respond(prompt) -> (text, {input_tokens, output_tokens})`; `CachedJudge` needs a client with `respond` (or `client=None` with `strict=True`). Cache entries: `{passed, reason, raw, usage}` or `{error, raw, usage}` — judge failures are now cached (changed from Phase 8) so a strict replay reproduces them; a non-strict run still re-asks the client. `CacheMissError` is not a `JudgeError`: it stops the run/rescore (CLI exit 1). `CachedJudge.usage` = {calls, cache_hits, input_tokens, output_tokens} for this run; cost counts fresh calls only (cached replay = $0).
  - `run_pack(..., seed=0, repeat=0)`; CLI `--seed`, `--repeat-index`, `--strict-cache`. Seeding = `random.seed` + numpy if already imported. Phase 13 repeats should pass `repeat`.
  - `uv.lock` (uv 0.11) covers all extras; CI `lock` job runs `uv lock --check`; `live.yml` installs via `uv sync --frozen --extra mem0` and runs `membench rescore --check` after the run. Run `uv lock` after any `pyproject.toml` dependency change.

## Phase 11 — Metrics v2
Status: done (with blocked items)
- [x] Schema: query `category`, `answer`, `expect_absent`, `evidence`, `abstain`
- [x] Packs: realistic timestamps (days to months apart) + migrated to the new fields; test that every write/query sets `timestamp`
- [x] Temporal reasoning accuracy (`temporal` queries)
- [x] Staleness (post-update queries whose context still holds an `expect_absent` value)
- [x] Forget compliance (post-`forget` queries without the forgotten content)
- [x] Abstention (correct "not known" on `abstain: true` queries)
- [x] Evidence recall@k over `RecalledItem.event_ids`
- [x] QA accuracy: optional `--reader-model` reader stage, judged against `answer`; manifest/cost `reader`
- [x] Token cost: `membench[tokens]` (tiktoken `o200k_base`, vocab cached); whitespace fallback labelled `approx`; tokenizer name in report
- [x] Latency: `write` / `ingest` / `recall` p50/p95; p95 `n/a` below 100 samples
- [x] Cost: ingest tokens per 1k writes, USD per 1k queries
- [x] Per-category breakdown in md/html reports
- [x] Fixture-trace unit tests for each new metric; schema + fixture regenerated; README docs
- Blocked: live QA accuracy / abstention numbers (reader + judge against real LLMs) — needs `MEMBENCH_JUDGE_API_KEY` (and `OPENAI_API_KEY` for Mem0). Covered here by stub clients only. Once secrets exist: `gh workflow run live.yml -f pack=customer-support -f reader_model=claude-haiku-4-5` (or locally `membench run --backend full-context --pack customer-support --reader-model claude-haiku-4-5`).
- Notes for next phases:
  - Report schema bumped to `REPORT_SCHEMA_VERSION = "2"` (breaking): `p95_latency_ms` replaced by `{write,ingest,recall}_{p50,p95}_ms` (p95 null below `MIN_P95_SAMPLES = 100`, forget latency no longer reported); new required score keys `staleness`, `temporal_accuracy`, `forget_compliance`, `evidence_recall_at_k`, `abstention`, `qa_accuracy`; new top-level `by_category` ({category: {n, recall_precision, evidence_recall_at_k, semantic_recall, qa_accuracy}}, "uncategorized" when unset); query trace `detail.category`, and `reader_answer` / `reader_error` when a reader ran; `cost.ingest_tokens_per_1k_writes`. v1 reports no longer validate (packs changed too, so they can't be rescored anyway).
  - Decisions: `category` is a free string (no fixed enum). Staleness uses the same post-update rule as `update_correctness` (expect_key updated earlier in the episode) + `expect_absent`. Forget compliance scores only `category: forget` queries (schema requires an earlier forget in the episode); fails on an `expect_absent` string in context OR an item tracing to a forgotten event id. Temporal accuracy = `expect_contains` pass rate on `category: temporal` (deterministic); reader-based temporal QA is `by_category.temporal.qa_accuracy`. Evidence recall@k uses all returned items (k = backend's own top-k), not a fixed k. Abstention is deterministic on the stored reader answer (regex for "I don't know"/"unknown"/"not mentioned"), None without a reader. Failed recall = miss for qa/abstention; reader or judge error = skipped (recorded on the trace).
  - Reader: `core/reader.py` (`Reader(client)`, `READER_PROMPT_VERSION = "1"`, prompt includes `CURRENT DATE: <at>`). Reuses judge clients via `resolve_client`; key `MEMBENCH_READER_API_KEY` else `MEMBENCH_JUDGE_API_KEY`. Reader answers are not cached (stored on traces, so `rescore` replays them). `run_pack(..., reader=)`; reader runs only on queries with `answer` or `abstain`. QA judging reuses the unchanged judge prompt (rubric from `qa_rubric(question, gold)`), so `JUDGE_PROMPT_VERSION` stays "1".
  - `score_all(pack, traces, judge) -> (scores, by_category)` is what the runner and `rescore` call; it memoizes judge verdicts per (rubric, evidence) so the breakdown adds no judge calls. `rescore --check` also diffs `by_category`. `score_run` still returns the flat scores.
  - Tokenizer: `run_pack` default is still `count_tokens` (whitespace, `name = "whitespace:approx"`); the CLI passes `default_tokenizer()` (tiktoken `o200k_base` when installed + vocab available, else whitespace). Manifest `tokenizer` = the callable's `name` attr. `TIKTOKEN_CACHE_DIR` defaults to `~/.cache/membench/tiktoken`. CI `chroma` job installs `tokens` and runs `tests/test_tokens.py`; `live.yml` syncs `--extra tokens` and gained a `reader_model` input.
  - Ingest tokens: the runner reads `adapter.usage()` before/after each `recall()`; `ingest = total backend tokens - recall tokens`.
  - Full-context context lines now carry the event timestamp (`[<ts>] [role] content`) so temporal questions are answerable from the log; Chroma/Mem0 contexts are unchanged (no dates), which `temporal_accuracy` exposes.
  - Bundled packs: `mini` kept its 8-step shape (tests index into it). `customer_support` (22 steps, 11 queries) and `coding_assistant` (24 steps, 11 queries) gained temporal, abstention and forget queries (+ one forget step each). All writes have explicit ids. Pack sha256s changed.
  - Phase 10's checklist still shows "`JUDGE_PROMPT_VERSION` in manifest + cache key; cache stores raw response + `usage`" unticked, though its notes say it is done (code confirms: manifest `judge.prompt_version`, cache key `{model}:v1:{sha}`, entries hold `raw` + `usage`).

## Phase 12 — Datasets: public loaders + harder synthetic packs
Status: in progress
- [ ] `membench/datasets/` loader core: pinned revision download, sha256 check, cache under `~/.cache/membench`, in-memory `Pack`; manifest `dataset`; `rescore` loader hook
- [ ] `membench convert` writes YAML locally (inspection only)
- [ ] LongMemEval loader (S / M / oracle) + official judge prompt for QA accuracy
- [ ] LoCoMo loader (adversarial → abstain) + gold-answer exclusion list; scores with and without it
- [ ] `--limit` / `--sample N --seed` stratified subsets
- [ ] Synthetic generator v2 (seeded, template-based; sessions, distractors, update chains, time gaps, multi-hop, abstention, forget; optional LLM paraphrase pass)
- [ ] Packs ≥200 queries, ≥30 per category: `tool_call_memory`, `document_chain`, `multi_step_decisions`, scaled `customer_support` / `coding_assistant`; byte-identical per seed
- [ ] Tests (fixtures only, no dataset content) + README docs
- [ ] LongMemEval_S and LoCoMo end-to-end on `full-context` and `chroma`
- [ ] Full-context oracle QA accuracy within ±3pp of the paper
