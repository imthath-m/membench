# Build progress — roadmap Phases 8–16
Resume protocol: list origin branches `claude/gracious-edison-v8eyxs*`; open the highest phase branch; read this file there; continue the first phase not marked done.

| Phase | Branch | PR | Status |
|---|---|---|---|
| 8 | claude/gracious-edison-v8eyxs | #6 | done (live run blocked) |
| 9 | claude/gracious-edison-v8eyxs-phase-9 | #7 | done |
| 10 | claude/gracious-edison-v8eyxs-phase-10 | #8 | done (cost-vs-dashboard blocked) |
| 11 | claude/gracious-edison-v8eyxs-phase-11 | #9 | done (live reader run blocked) |
| 12 | claude/gracious-edison-v8eyxs-phase-12 | #10 | done (oracle ±3pp + LoCoMo human audit blocked) |
| 13 | claude/gracious-edison-v8eyxs-phase-13 | #11 | done (LLM-backed repeats blocked) |
| 14 | claude/gracious-edison-v8eyxs-phase-14 | #12 | done (live vendor runs blocked) |
| 15 | claude/gracious-edison-v8eyxs-phase-15 | — | in progress |

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
  - Phase 10's checklist still shows "`JUDGE_PROMPT_VERSION` in manifest + cache key; cache stores raw response + `usage`" unticked, though its notes say it is done (code confirms: manifest `judge.prompt_version`, cache key `{model}:v1:{sha}`, entries hold `raw` + `usage`). (Now ticked.)

## Phase 12 — Datasets: public loaders + harder synthetic packs
Status: done (with blocked items)
- [x] `membench/datasets/` loader core: pinned revision download, sha256 check, cache under `~/.cache/membench`, in-memory `Pack`; manifest `dataset`; `rescore` loader hook
- [x] `membench convert` writes YAML locally (inspection only)
- [x] LongMemEval loader (S / M / oracle) + official judge prompt for QA accuracy
- [x] LoCoMo loader (adversarial → abstain) + gold-answer exclusion list; scores with and without it (structural audit only; see Blocked)
- [x] `--limit` / `--sample N --seed` stratified subsets
- [x] Synthetic generator v2 (seeded, template-based; sessions, distractors, update chains, time gaps, multi-hop, abstention, forget; optional LLM paraphrase pass)
- [x] Packs ≥200 queries, ≥30 per category: `tool_call_memory`, `document_chain`, `multi_step_decisions`, scaled `customer_support` / `coding_assistant`; byte-identical per seed
- [x] Tests (fixtures only, no dataset content) + README docs
- [x] LongMemEval_S and LoCoMo end-to-end on `full-context` and `chroma` (no judge; each followed by `membench rescore --check` → identical): full-context LongMemEval_S 500 q, evidence_recall@k 1.0, error_rate 0; full-context LoCoMo evidence_recall@k 1.0, error_rate 0; chroma LoCoMo full (7,868 traces) evidence_recall@k 0.187, error_rate 0; chroma LongMemEval_S `--limit 5` (2,695 traces) evidence_recall@k 0.60, error_rate 0. Full chroma S run (~14 h) not done — adapter throughput, not a bug.
- [ ] Full-context oracle QA accuracy within ±3pp of the paper — Blocked (API key; see below)
- [ ] Lower priority: BEAM / MemoryAgentBench loaders — Deferred, not in this PR (see below)
- Blocked: ±3pp LongMemEval reproduction — needs a reader + judge API key (paper used GPT-4o for both). Official reader/judge prompts are wired in and checked verbatim against upstream (`qa_style: longmemeval`). Once a key exists: `MEMBENCH_JUDGE_API_KEY=<openai key> membench run --backend full-context --pack longmemeval-oracle --reader-model gpt-4o-2024-08-06 --reader-base-url https://api.openai.com/v1 --judge-model gpt-4o-2024-08-06 --judge-base-url https://api.openai.com/v1 --out lme-oracle.json`, then compare `scores.qa_accuracy` (and `by_category`) to the paper's oracle row for that reader. Known gap sources to check: our context is full-context's `[ts] [role] content` lines, not the official per-session JSON history; Anthropic reader `max_tokens` is 256 (official 500); the cleaned dataset release (2025-09) differs from the paper's original files.
- Blocked: human audit of LoCoMo gold answers — needs a human annotator. `locomo_exclusions.yaml` holds the 9 items an automated structural audit flagged (unresolvable / missing evidence, adversarial items with a contradicting gold answer); add human findings in the same format.
- Deferred (lower priority, per roadmap): BEAM and MemoryAgentBench loaders. License check done: HF `ai-hyz/MemoryAgentBench` is tagged MIT; a HF `Mohammadta/BEAM` dataset is tagged CC BY-SA 4.0 but is not yet confirmed as BEAM's official release.
- Notes for next phases:
  - Pack resolution: `membench.datasets.resolve_pack(spec, *, pack_seed, exclude_flagged, limit, sample, sample_seed)` is the one entry point (CLI `run`, `convert`, `rescore`). Order: YAML path → dataset (`DATASETS`) → synthetic (`SYNTHETIC`) → bundled name; hyphens/underscores interchangeable. `Pack` gained `dataset`, `subset`, `qa_style` ("membench" | "longmemeval"), `flagged` (query step indices on an exclusion list).
  - Manifest: `dataset` = `Pack.dataset` ({id, revision, sha256, url, license, options}; synthetic: revision `synthetic-v1`, sha256 = generated pack sha, options `{pack_seed}`); `pack` gained `subset` ({limit} | {sample, sample_seed}) and `qa_style`. Schema stays v2 (new keys optional). `rescore` rebuilds via `resolve_pack(dataset.id, **dataset.options, **pack.subset)`; dataset packs need the cached download (or network).
  - `select_queries(pack, keep)` keeps chosen queries plus all writes/forgets of episodes that still have a query (others dropped, episodes renumbered); subset sha = sha256(f"{sha}:{sorted(keep)}"). `--sample` is stratified by category (largest remainder), seeded by the run's `--seed`. Phase 13 can reuse it for bootstrap/paired subsets.
  - Downloads: `datasets/download.py` `fetch(Source)`; cache `~/.cache/membench/datasets/<sha256>/<file>` (`MEMBENCH_CACHE_DIR`), re-hashed on each load. Pins: LongMemEval `xiaowu0162/longmemeval-cleaned@98d7416c…` (S/M/oracle file sha256s = HF LFS oids, S + oracle verified by download); LoCoMo `snap-research/locomo@3eb6f2c5…` `data/locomo10.json` (verified). Network to huggingface.co and raw.githubusercontent.com worked here (not blocked).
  - LongMemEval: ids `q<i>-s<j>-t<k>` (opaque; upstream `answer_` session ids and `has_answer` never reach adapters); sessions sorted by date (211/500 S questions are not in date order upstream; oracle is unsorted by design); turn k at session date + k s. `_abs` queries carry `abstain: true` AND `answer` = upstream explanation — the schema rule "abstain has no answer" was relaxed for this; default-style `qa_accuracy` skips abstain queries, longmemeval-style judges them with the official abstention prompt. Judge: `CachedJudge.ask(prompt, parse)` (+ `parse_yes_no`, official "yes in reply" rule); `_MemoJudge.ask` memoizes. Reader: `Reader.answer(..., style=)`; runner passes `pack.qa_style`. `JUDGE_PROMPT_VERSION` / `READER_PROMPT_VERSION` unchanged (the style is recorded in `manifest.pack.qa_style`; cache keys hash the full prompt).
  - LoCoMo: categories mapped 1 multi_hop, 2 temporal, 3 open_domain, 4 single_hop, 5 adversarial (abstain). Queries asked at last turn + 1 day. Note `temporal` category name collides with membench's `temporal_accuracy` (expect_contains-based) but LoCoMo queries have no `expect_contains`, so that metric stays None for it. `--exclude-flagged` on `run` drops flagged queries; on `rescore` it scores a full run without them (refused on --limit/--sample runs).
  - Synthetic: `generate/synthetic.py` (`generate(name, seed, knobs)`, `Knobs`, `paraphrase_queries`) + `generate/domains.py` (5 domains). Default 36 episodes × 6 categories = 216 queries, 1,188 steps. Not committed as YAML; generated on load (~10 ms). `GENERATOR_VERSION` must be bumped on any template/logic change (it is in the manifest revision).
  - Real-data runs here (no judge): full-context LongMemEval_S (247k writes, 500 q) 67 s, report 622 MB (contexts are stored on traces — consider trimming for large runs); full-context LoCoMo 30 s. Chroma's adapter is slow (~5 writes/s on this 4-CPU box), so LongMemEval_S on chroma was run with `--limit 5`; a full S run is ~14 h at that rate.

## Phase 13 — Statistical rigor
Status: done (with blocked items)
- [x] `--repeats N` (default 3 when backend or judge/reader uses an LLM, else 1); fresh `reset()` + new namespace per repeat; report holds mean, run-to-run SD, every repeat
- [x] 95% CIs: seeded bootstrap over queries (10k resamples) / Wilson for plain proportions, next to every rate
- [x] Paired comparison (`membench compare`): paired bootstrap of the difference + McNemar for pass/fail metrics; flags differences whose CI includes 0
- [x] Minimum sample sizes enforced in reports: headline ≥385, per-category ≥100 ("indicative"), p95 ≥100 per op
- [x] Holm correction in comparison reports (metrics × categories)
- [x] md/html reports show n + CI for every metric; suppress/label below min n
- [x] `rescore` reproduces CIs (seeded); schema + fixture regenerated
- [x] Two runs of the same config have overlapping CIs (offline demo on full-context / chroma): `membench run --backend {full-context,chroma} --pack tool-call-memory --no-judge --repeats 2` → both repeats' CIs identical (deterministic backends), e.g. chroma recall_precision 94.4% [89.4–97.2%], evidence_recall@k 87.8% [83.3–92.0%]; chroma used collections `membench-r0` / `membench-r1`; `membench rescore --check` identical on both reports. Also `membench compare` full-context vs chroma on that pack: evidence_recall@k −12.2pp [−16.7, −8.0] real difference; recall_precision −5.6pp (McNemar p 0.008, Holm 0.086, not significant after Holm).
- Blocked: overlapping CIs across repeats of an LLM-backed run (Mem0 and/or reader/judge) — needs `OPENAI_API_KEY` + `MEMBENCH_JUDGE_API_KEY`. Once secrets exist: `gh workflow run live.yml -f pack=customer-support -f repeats=3` (or locally `membench run --backend mem0 --pack customer-support --judge --repeats 3 --out r.json`), then check `repeat_summary` SDs and that each repeat's `stats.scores[*].ci` overlap; `membench rescore --in r.json --check`.
- [x] Tests + README docs
- Notes for next phases:
  - Report (schema stays v2; new keys optional in the schema, always written): `outcomes` = {metric: {"<step_index>": bool | number}} for the 11 query-level metrics (`scoring.OUTCOMES`); `stats` = {"scores": {metric: stat}, "by_category": {cat: {metric: stat}}}, stat = {n, ci: [lo, hi] | null, method: "wilson" | "bootstrap" | null, indicative}; `repeats` = full reports of repeats 1..N-1 (schema `$ref: "#"`, no nesting); `repeat_summary` = {n, scores: {metric: {mean, sd, values}}} or null.
  - Each metric module now has `<metric>_outcomes(...) -> {step_index: value}`; the metric is `stats.mean_of(outcomes)`. `scoring.score_detailed(pack, traces, judge, seed) -> Scoring(scores, by_category, outcomes, stats)` is what the runner and `rescore` call; `score_all` / `score_run` are thin wrappers. `by_category` is now computed by filtering outcomes by category (same values as before, no sub-pack re-scoring). Outcomes are keyed by `trace.step_index`, so hand-built test traces need unique indices.
  - CIs (`core/stats.py`, pure Python): Wilson for all-bool outcomes (and `error_rate` over steps); seeded percentile bootstrap (10k) for numeric (`evidence_recall_at_k`, `token_cost`), key `f"{seed}:{scope}:{metric}"` (`random.Random(str)` is stable across processes). Latency stats carry n only. ~2 s per 2,000-value bootstrap, so a LoCoMo-sized run adds a few seconds of scoring.
  - Minimums: `MIN_HEADLINE_N = 385`, `MIN_CATEGORY_N = 100` (stats.py), p95 `MIN_P95_SAMPLES = 100` (already suppressed). Reports label `indicative` / `suppressed`; no number is hidden except p95.
  - Repeats: `--repeats N` (default 3 if `adapter_cls.uses_llm` (Mem0) or a judge/reader client exists, else 1). Each repeat builds a new adapter with `config["namespace"] = f"{namespace}-r{repeat_index + i}"` (`MemoryAdapter.namespace`; Chroma collection / Mem0 user_id default to it), resets it (unless `--no-reset`), and runs with `repeat=repeat_index + i`; `runner.merge_repeats` folds them. Judge verdicts are still cached by prompt across repeats, so repeats capture backend + reader variance, not judge variance (Phase 15 territory).
  - `rescore` re-scores the report and every repeat, recomputes `repeat_summary`, and `--check` also diffs `outcomes` / `stats` when the report has them (pre-Phase-13 reports are checked on scores + by_category only).
  - `membench compare --a --b [--out x.md|x.json]` (`core/report/compare.py`): pairs top-level (repeat 0) outcomes by step index, overall + per category (`CATEGORY_METRICS`); `stats.paired_diff` (paired bootstrap CI, McNemar exact for bool, bootstrap p otherwise); Holm over all rows; `significant` = Holm p < 0.05 and CI excludes 0. Refuses different pack sha (incl. subset), judge or reader model/prompt version, or reports without `outcomes` (fix: `membench rescore --in old.json --out new.json`).
  - `membench report --in a --in b` (side by side) cells now read "72% [65–78%] n=40 (indicative)" when stats exist.

## Phase 14 — Backend coverage
Status: done (with blocked items)
- [x] Shared adapter contract test suite (write → flush → recall finds it; forget removes it; reset empties; episode isolation)
- [x] `bm25` baseline (lexical floor, offline)
- [x] `oracle` baseline (only the query's `evidence` events; reader upper bound)
- [x] Mem0 Platform (`MemoryClient`; `flush()` polls async ingestion; namespace per episode)
- [x] Zep Cloud (user graph; graph search; `flush()` waits for ingestion; `at` via bi-temporal facts)
- [x] Letta archival passages + core blocks; separate `letta-agent` (agent-managed memory via messages)
- [x] LangMem (memory-store manager over a LangGraph `BaseStore`; recall `store.search`)
- [x] Cognee (`add` / `flush()` → `cognify` / `search` / `reset()` prunes; async wrapped)
- [x] Per backend: optional extra, import guard, `docs/<backend>.md`, mocked unit tests, gated live test
- [x] `live.yml`: matrix over the hosted backends (secret-gated, manual dispatch)
- [x] Vendor review: draft issue text + default / vendor-tuned config slots (`docs/vendor-review.md`)
- [x] README + registry + `uv.lock`
- [ ] Done-when: all 5 run `mini` live via `live.yml` — Blocked (keys)
- [x] Done-when: every adapter file < 100 lines (test); each adapter passes the shared contract suite (offline adapters for real, vendor adapters on SDK fakes)
- Blocked: live `mini` on the 5 hosted backends (+ `letta-agent`) — needs repo secrets `MEM0_API_KEY`, `ZEP_API_KEY`, `LETTA_API_KEY`, `OPENAI_API_KEY` (LangMem; also passed to Cognee as `LLM_API_KEY`) and `MEMBENCH_JUDGE_API_KEY`. Once set: `gh workflow run live.yml -f backend=all -f pack=mini` (or one backend: `-f backend=zep`). Each matrix job first runs the gated contract test (`MEMBENCH_LIVE=1 pytest "tests/test_live_backends.py::test_live_mini_runs_clean[<backend>]"`), then `membench run ... --judge`, `rescore --check`, md report. Locally: `MEMBENCH_LIVE=1 ZEP_API_KEY=... pytest tests/test_live_backends.py -k zep -rs`.
- Manual step (user): open the 5 vendor review issues drafted in `docs/vendor-review.md` (mem0ai/mem0, getzep/zep, letta-ai/letta, langchain-ai/langmem, topoteretes/cognee), then put the issue URLs in its table. Not opened from here by design.
- Notes for next phases:
  - New backends: `bm25`, `oracle`, `mem0-platform`, `zep`, `letta`, `letta-agent`, `langmem`, `cognee` (extras `mem0-platform`, `zep`, `letta`, `langmem`, `cognee`; `letta-agent` uses `letta`). SDK APIs were checked against the installed PyPI packages (mem0ai 2.2.1, zep-cloud 4.1.0, letta-client 1.12.1, langmem 0.0.30 / langgraph 1.2, cognee 1.6.7) and, for mem0-platform / zep / letta / letta-agent, driven end to end through the real SDK clients over an httpx mock transport (request paths + bodies checked); langmem ran through real `create_memory_store_manager` + `InMemoryStore` with a local fake chat model / embedder; cognee's `add` / `forget` / `prune` ran for real (local stores), `cognify` / `search` only signature-bound (they need an LLM / embedder). Floors: `zep-cloud>=4.1` (v4 API: everything is addressed by user/graph uuid), `letta-client>=1.12`, `langmem>=0.0.30`, `cognee>=1.6`.
  - Contract: new class attr `MemoryAdapter.sees_evidence` (False); only `oracle` sets it, and the runner then sets `adapter.evidence = step.evidence` right before each recall. Context of the oracle = full-context line format.
  - Shared poller `core/poll.py`: `drain(pending_ids, done, interval_s, timeout_s)` + `poll_options(config)` (keys `poll_interval_s` 1.0, `flush_timeout_s` 600). Used by mem0-platform (event ids) and zep (task uuids). A failed job raises once and is dropped. `flush_timeout_s` is longer than the default `--step-timeout` 120s: raise `--step-timeout` for slow ingestion.
  - Episode isolation: every vendor adapter moves to a new scope on `reset()` (`<namespace>-e<n>`: mem0-platform user_id, zep user metadata label / new user, letta agent name, langmem namespace, cognee dataset), so the CLI's initial reset makes the first episode `-e1`. Repeats keep `membench-r<i>` as the namespace prefix.
  - Event-id traceability: mem0-platform via memory `metadata.membench_event_id`; zep via edge `episode_uuids`; letta via passage ids; langmem via keys the manager put; cognee via chunk `document_id` (= data id; `CHUNKS` search only). `letta-agent` items carry no ids (agent's own notes), so its `evidence_recall_at_k` reads 0 — not meaningful; same for cognee with a non-CHUNKS `search_type`.
  - Usage/cost: `letta-agent` (response `usage`) and `langmem` (LangChain `UsageMetadataCallbackHandler`, chat tokens only) report `usage()`; mem0-platform, zep, letta, cognee do not (cost.backend null). `cost._usd` now also prices provider-prefixed handles (`openai:gpt-4o-mini`, `openai/gpt-4o-mini`) as the bare id.
  - Default configs live in `configs/default/<backend>.yaml` (a test checks they equal the adapter defaults); vendor-tuned go in `configs/vendor-tuned/` once vendors reply. Phase 16 publishing should show both rows, labelled.
  - Decisions: Cognee recall defaults to `CHUNKS` (traceable) rather than a graph completion type; Zep recall is `graph.search_edges` with a `valid_at <= at` filter (context block not used); Letta archival agents are created with empty `human` / `persona` blocks and `openai/gpt-4o-mini` / `openai/text-embedding-3-small` handles; Graphiti (self-hosted Zep) is documented, not implemented. Cognee's `reset()` prunes the whole local Cognee install (roadmap wording); Cognee picks up AWS env credentials and then needs `s3fs`.
  - Tests: `tests/test_contract_suite.py` (FACTORIES covers every registered backend; a test fails if a backend is registered without one), `tests/vendor_fakes.py` (SDK fakes with async-ingestion semantics: a missing flush fails), `tests/test_vendor_adapters.py`, `tests/test_baselines.py`, `tests/test_live_backends.py` (skipped unless `MEMBENCH_LIVE=1` + keys). CI `chroma` job also runs the contract suite with real chromadb.

## Phase 15 — Judge validation
Status: in progress
- [x] Synthetic packs carry judge rubrics (semantic_recall on answerable queries, contradiction on update queries) so every judge metric has ≥200 candidate items
- [ ] `membench judge-sample`: stratified by metric, backend, verdict, context-length bucket; ≥200 items per judge metric
- [ ] Labelling guidelines (2 independent annotators + adjudication)
- [ ] `validation/labels.jsonl` format + loader/validator; merge of 2 annotators + adjudication
- [ ] `membench judge-agreement`: human–human κ; judge–majority-human κ, accuracy, false-pass/false-fail; by context-length bucket; validated/unvalidated per metric → `validation/agreement-v<N>.json`
- [ ] Cross-judge runner (≥3 judge configs incl. an open-weight model via OpenAI-compatible base URL); pairwise κ + disagreeing items
- [ ] Prompt-versioning policy (docs + test pinning the prompt to `JUDGE_PROMPT_VERSION`)
- [ ] Reports mark judge metrics `unvalidated` unless a committed validation result validates (judge model, prompt version, metric)
- [ ] Offline judge-sample from full-context / chroma / bm25 on synthetic packs
- [ ] Secret-gated manual workflow for cross-judge + agreement; README section
