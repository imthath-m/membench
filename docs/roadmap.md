# MemBench — Roadmap from v0.1 to a credible benchmark

v0.1 (plan.md Phases 1–7) is a working harness. It is not yet a benchmark: packs are tiny, two PRD metrics are missing, the judge is unvalidated, and nothing has run against a live service. Phases continue plan.md's numbering; each is one or more reviewable PRs with explicit "done when" criteria.

## Starting point (verified against the code)

- **Adapters:** `full-context`, `chroma`, `mem0` (OSS, `mem0ai>=2.0`). No Zep, Letta, LangMem, Cognee, or Mem0 Platform.
- **Packs:** `mini` (4 queries), `customer_support` (7), `coding_assistant` (7). No pack sets `timestamp`; the schema fills in epoch + step index in seconds.
- **Step types:** `write` and `query` only. The runner never calls `forget()`, so that part of the contract is untested end to end.
- **Fairness leak:** the pack's `key` is the fact id used for scoring. `ChromaAdapter.write` uses it as the document id (`upsert(ids=[event.key])`), so an update overwrites the stale value. Chroma's `update_correctness` is unearned.
- **Metrics:** 6. `token_cost` is a whitespace word count (`core/tokens.py`), computed inside each adapter, which goes against the "runner owns token accounting" rule. `p95_latency_ms` pools writes and queries over at most 16 steps, so it is close to the max.
- **Missing PRD metrics:** temporal reasoning, staleness.
- **Judge:** yes/no prompt (`build_prompt`), Anthropic by default plus an OpenAI-compatible client, cached by (model, prompt hash). Prompt version is not recorded; `usage` is thrown away; a malformed reply (`parse_verdict` raising `ValueError`) or an unexpected response shape (`complete()` indexing into the reply and raising `KeyError`, `IndexError` or `TypeError`) stops the run.
- **Robustness:** any adapter exception stops the run and no report is written. Mem0's own OpenAI calls are not retried. There are no step timeouts.
- **Missing extra:** `adapter_cls(adapter_config)` in `cli.py run` sits outside the `try`, so a missing `mem0ai` prints a traceback. README and `docs/mem0.md` claim a clean error.
- **Report:** `asdict(RunResult)`. No schema version, manifest, judge model, or config.
- **CI:** installs `.[dev]` only, so `test_chroma_adapter` is skipped. No live or secret-gated jobs.
- **Deps:** lower bounds only, no lock file.

## Known issues

- **Chroma `update_correctness` is unearned:** the fact key is the document id, so an update overwrites the old value. → Phase 9
- **`forget()` is never exercised:** the runner handles `write` and `query` steps only. → Phase 9
- **Packs are tiny, with no timestamps:** 4/7/7 queries; events default to 1 second apart. → Phase 11 (timestamps), Phase 12 (size)
- **Missing extra prints a traceback:** the adapter is built outside the CLI's `try`, against the README and `docs/mem0.md`. → Phase 8
- **A malformed judge reply or unexpected response shape aborts the run:** `parse_verdict` raises `ValueError`; both clients' `complete()` raise `KeyError`, `IndexError` or `TypeError` on a missing `content` / `choices`. → Phase 8

## Guiding constraints (carried over, plus additions)

- Adapter budget stays under 100 lines. Shared logic (retry, timing, tokenization, flushing) lives in the runner.
- Offline-first stays: deterministic metrics and the replay of committed results need no network.
- **New:** adapters never see scoring ground truth (fact keys, expected answers, categories).
- **New:** every published number carries n, a CI, and a manifest. Nothing is compared across judge models or prompt versions.
- **New:** dataset files are never committed. Loaders download a pinned revision on demand and check its sha256.

---

## Phase 8 — Hardening + first live run

**Goal:** stop crashing, survive real APIs, and get one live Mem0 scorecard.

**Deliverables**
- Construct the adapter and call `reset()` inside the CLI's error handling, so a missing extra gives a one-line error. Add a test with `mem0` blocked from import. Fix the README and `docs/mem0.md` claims.
- Runner-level retry: shared `core/retry.py` with exponential backoff and jitter. Adapters can declare a `retryable` exception tuple (e.g. `openai.RateLimitError`, `APITimeoutError`), which covers Mem0's internal LLM calls.
- Per-step timeout (`--step-timeout`, default 120s).
- Per-step error capture: a failed step is written to the trace as `error`, the run continues, and the report gains `error_rate`. A run whose `error_rate` is above a threshold is marked invalid.
- Judge: a reply that fails to parse is retried once, then recorded as a `judge_error` instead of stopping the run. An unexpected response shape from either client's `complete()` (`KeyError`, `IndexError`, `TypeError`) is also recorded as `judge_error`, with no retry.
- Tests for the judge and the CLI: a malformed judge reply (no JSON, bad JSON) is recorded per query as `judge_error` and the other queries still score; a stubbed client returning an unexpected shape (e.g. no `choices`) finishes the run and records that query as `judge_error`; `mem0` blocked from import gives the one-line error.
- CI: add a `chroma` job so its adapter tests run.
- CI: `live.yml` (manual dispatch only, secret-gated) runs `mini` against `mem0` with the judge on.

**Done when**
- `membench run --backend mem0` without `mem0ai` prints a one-line actionable error with exit code 1 (tested).
- A live `mem0` + judge run of `customer-support` completes; any injected 429 is retried and logged.
- A forced adapter exception mid-pack still produces a valid report with the error recorded.
- A pack with one malformed judge reply or unexpected response shape still finishes; that query is recorded as `judge_error` and the rest are scored (tested).

## Phase 9 — Adapter contract v2 (breaking, bump to 0.2)

**Goal:** a contract that is fair, covers time and deletion, and can carry hosted or async backends.

**Deliverables**
- Split identity. `Event.id` is a unique opaque id for each write and is the only id adapters see. `fact_key` stays in the pack for scoring and is never passed to adapters. This fixes the Chroma overwrite leak.
- `recall(query, *, session_id, at)`: pass session scope and query time so temporal questions are answerable.
- `flush()`: default no-op. It blocks until async ingestion is done (Zep, Cognee `cognify`, Mem0 Platform). The runner calls it before a query that follows writes and records the time as `ingest_latency_ms`, kept separate from recall latency.
- `RecallResult.items` becomes `list[RecalledItem(event_ids, text, score)]`, so evidence-retrieval metrics can work. Each adapter maps its own ids back to `Event.id`.
- Token counting moves to the runner, which counts `context` with the configured tokenizer. `RecallResult.token_count` is removed.
- Optional `usage()`: cumulative backend LLM/embedding tokens (ingest cost, e.g. Mem0's extraction calls).
- New `forget` step type (`target: <event id(s)>`). The runner now exercises `forget()`.
- Episodes: a pack holds one or more independent `episodes`, with `reset()` (or a fresh namespace) between them. Public datasets need this; for example, each LongMemEval question has its own haystack.
- Test: every file in `membench/adapters/` is under 100 lines.
- Pack test for backend-side update resolution: an `update` write has a new `Event.id` and different wording from the stale fact, so only the backend can decide which is current. Assert that the adapter receives distinct ids and that Chroma's context holds both values.
- Runner test: a `forget` step calls `adapter.forget` with the target ids (recording fake adapter), and is traced with its latency.

**Done when**
- All 3 adapters are ported, each still under 100 lines; the contract is documented in the README.
- Chroma no longer scores 1.0 on `update_correctness` by construction (regression test).
- The update-resolution pack test passes on all 3 adapters, and no adapter receives a `fact_key`.
- The runner test shows `forget` steps reach `adapter.forget`.
- A pack with a `forget` step and two episodes runs on all 3 adapters.

## Phase 10 — Reproducibility + cost accounting

**Goal:** any published number can be traced back to exact inputs and re-scored offline.

**Deliverables**
- Run manifest in every report. It holds `schema_version`, the membench version and git commit (plus a dirty flag), Python and OS, versions of the backend SDKs (`importlib.metadata`), and the backend config with secrets redacted. It also holds the pack name and content sha256, the dataset id/revision/sha256, and the judge, reader and backend LLM models with base URLs. Plus the judge prompt version, tokenizer, seed, repeat index, deployment (local/hosted, region), and start/finish times.
- `JUDGE_PROMPT_VERSION` constant: recorded in the manifest and included in the cache key. The cache also stores the raw response and `usage`.
- Judge `--strict-cache` mode: a cache miss is an error, which is used to replay published results.
- `membench rescore --in report.json`: recompute scores from the traces and judge cache with no adapter and no network.
- Lock file (`uv.lock`), with published runs made from a frozen sync. Floor versions remain for library users.
- Cost tracking: judge, reader and backend token counts from API `usage`. A dated price table in the repo gives USD per run and per 1k queries.
- Determinism: seed everything that is seedable. Record the embedding model id (Chroma's default ONNX model) and temperature. Document what stays non-deterministic (LLM extraction, judge at temp 0, latency).

**Done when**
- `membench rescore` on a committed report reproduces identical scores offline (CI test).
- Every report validates against a committed JSON Schema.
- A run's cost section matches the provider dashboard within 5%.

## Phase 11 — Metrics v2

**Goal:** all PRD metrics, tagged by category, measured with real tokens.

**Deliverables**
- Schema: query `category` tags (`single_hop`, `multi_hop`, `temporal`, `update`, `abstention`, `forget`, …), `answer` (gold), `expect_absent` (stale or forgotten strings), `evidence` (event ids), `abstain: true`.
- **Temporal reasoning:** accuracy on `temporal` queries, using `at` from Phase 9. Packs gain realistic timestamps (days to months apart).
- **Staleness:** fraction of post-update queries whose context still contains the `expect_absent` stale value. This is deterministic and complements `update_correctness`; full-context passes `update_correctness` but scores badly here.
- **Forget compliance:** fraction of post-`forget` queries in which the forgotten content does not appear.
- **Abstention:** correct "not known" on questions with no answer.
- **Evidence recall@k:** a deterministic retrieval metric over `RecalledItem.event_ids`.
- **QA accuracy (reader stage):** optional `--reader-model` produces an answer from context + question, and the judge compares it to `answer`. This is what makes results comparable to published LongMemEval/LoCoMo numbers.
- **Token cost:** real tokenizer via `membench[tokens]` (`tiktoken` `o200k_base`, vocab cached). The whitespace fallback is labelled `approx` in the report. Report the tokenizer name.
- **Latency:** separate `write`, `ingest` and `recall` p50/p95. p95 is marked `n/a` below 100 samples.
- **Cost:** ingest tokens per 1k writes and USD per 1k queries.
- Per-category breakdown in the md/html reports.
- Test: every `write` and `query` in the bundled packs sets an explicit `timestamp`; none relies on the 1-second default.

**Done when**
- Every PRD metric (recall accuracy, contradiction handling, temporal reasoning, staleness, latency, tokens per recall) appears in the JSON report.
- Each new metric has fixture-trace unit tests.
- The existing packs are migrated to the new fields.

## Phase 12 — Datasets: public loaders + harder synthetic packs

**Goal:** enough questions, from public benchmarks and from our own production-shaped scenarios.

**Deliverables**
- `membench/datasets/`: loaders that build in-memory `Pack`s (download, pin revision, check sha256, cache under `~/.cache/membench`). `membench convert` writes YAML locally for inspection only.
- **LongMemEval** (S and M variants, oracle for reader calibration): 500 questions, each with its own haystack → 1 episode. Haystack sessions → `write` per turn with `haystack_dates` timestamps. Question → `query` with `at=question_date`, `answer` and `evidence` from the answer-session turns. `question_type` → `category`; `_abs` questions → `abstain`. Use the official judge prompt for QA accuracy. License: MIT per the upstream repo; confirm the dataset card before release.
- **LoCoMo:** 10 long conversations → 10 episodes. Dialogue ids (`D1:3`) → `Event.id`; session `date_time` → timestamps; QA `evidence` → `evidence`; categories → `category` (adversarial → `abstain`). License: CC BY-NC 4.0, non-commercial, so never redistribute raw or converted files. Audit gold answers, publish an exclusion list with reasons, and report scores both with and without it.
- **Lower priority:** BEAM (named in the PRD; for scale beyond 1M tokens) and MemoryAgentBench (conflict resolution and selective forgetting). Confirm each license before writing a loader.
- `--limit` / `--sample N --seed`: stratified subsets for cost-bounded runs (LongMemEval_S on an LLM-extracting backend means writes in the order of 10^5).
- **Synthetic generator v2:** a deterministic, seeded, template-based generator that needs no LLM. Knobs cover sessions, distractor turns, update-chain depth (A→B→C), time gaps, multi-hop, abstention and forget. An optional LLM paraphrase pass breaks lexical overlap between the query and the needle (today, "Which subscription plan" matches "Basic plan" by keyword).
- **New packs on the PRD's differentiation axis,** each with at least 200 queries: `tool_call_memory` (facts that only appear in tool results), `document_chain` (facts spread across chained docs), `multi_step_decisions` (constraints set early, used later), plus scaled `customer_support` and `coding_assistant`.

**Done when**
- LongMemEval_S and LoCoMo load and run end-to-end on `full-context` and `chroma`.
- With the oracle haystack and the official reader and judge prompts, full-context QA accuracy is within ±3pp of the number published in the LongMemEval paper for the same reader model, or the gap is explained.
- Each synthetic pack has at least 200 queries and at least 30 per category; generation is byte-identical for a given seed.

## Phase 13 — Statistical rigor

**Goal:** every score has a CI, and comparisons say whether a difference is real.

**Deliverables**
- `--repeats N` (default 3 for backends or judges that use an LLM, 1 otherwise), with a fresh `reset()` and a new namespace on each repeat. The report holds the mean, the run-to-run SD, and every repeat.
- 95% CIs: seeded bootstrap over queries (10k resamples), or Wilson intervals for plain proportions, shown next to every rate.
- Paired comparison of two backends on the same queries: paired bootstrap of the difference, plus McNemar's test for pass/fail metrics. The comparison report flags differences whose CI includes 0.
- Minimum sample sizes, enforced in reports:
  - headline rate: at least 385 queries (±5pp at 95%, p=0.5)
  - per-category rate: at least 100 queries (±10pp), otherwise labelled "indicative"
  - p95 latency: at least 100 samples per operation type
- Holm correction when a comparison report shows many metrics × categories.

**Done when**
- The md/html reports show n and CI for every metric, and suppress or label anything below the minimum n.
- Two runs of the same config have overlapping CIs; the `rescore` path also reproduces the CIs because the bootstrap is seeded.

## Phase 14 — Backend coverage

**Goal:** the backends the PRD names, each under 100 lines, each config checked against vendor docs.

**Deliverables** (each one: optional extra, import guard, `docs/<backend>.md`, mocked unit tests, a gated live test)
- **Mem0 Platform** (`MemoryClient`): hosted, may process asynchronously, so `flush()` polls until done. Uses a separate namespace per episode.
- **Zep** (Cloud SDK; Graphiti for self-hosting, since Community Edition is deprecated): messages go to a thread/user graph; recall is graph search or the context block. Ingestion is asynchronous, so `flush()` is required. Supports bi-temporal facts, which exercise `at`.
- **Letta:** archival-memory passages (insert/search) plus core memory blocks as context. The variant where the agent manages its own memory through `send_message` is a separate, documented adapter because its cost profile is different.
- **LangMem:** memory-store manager (LLM extraction) over a LangGraph `BaseStore`; recall is `store.search`.
- **Cognee:** `add` writes; `flush()` runs `cognify`; recall is `search`; `reset()` prunes data and system. Wrap the async API.
- **Extra baselines:** `bm25` (a lexical floor) and `oracle` (returns only the `evidence` events; an upper bound that isolates the reader).
- Vendor review: open an issue for each vendor, inviting a review of the adapter and default config. Publish default and vendor-tuned configs side by side, labelled as such.

**Done when**
- All 5 run `mini` live via `live.yml`; every adapter file is under 100 lines.
- Each adapter passes a shared contract test suite (write → flush → recall finds it; forget removes it; reset empties; episode isolation).

## Phase 15 — Judge validation

**Goal:** evidence that the judge agrees with humans, and with other judges, on each metric that uses it.

**Deliverables**
- `membench judge-sample`: stratified sample of (rubric or question+answer, context) items by metric, backend, verdict and context length. At least 200 items for each judge metric (`semantic_recall`, `contradiction_rate`, `qa_accuracy`).
- Human labelling: 2 independent annotators with written guidelines, then adjudication. Labels are committed as `validation/labels.jsonl`, with prompt version and model.
- Agreement report (`membench judge-agreement`): human–human Cohen's κ; judge–majority-human κ, accuracy, and false-pass and false-fail rates. Agreement is broken down by context-length bucket, to catch a length bias that would favour or penalise full-context.
- Cross-judge: run at least 3 judges (Claude, GPT, one open-weight model) on the same items. Report pairwise κ and the items where they disagree.
- Prompt versioning policy: any prompt edit bumps `JUDGE_PROMPT_VERSION`, re-runs the agreement report, and stores the result in `validation/`.

**Done when**
- Human–human κ ≥ 0.7, and judge–human κ ≥ 0.6 for each metric, no more than 0.1 below human–human.
- A metric that fails is marked `unvalidated` in reports, and its prompt is reworked.
- The default judge model and prompt version are the ones that were validated, and the README cites the agreement numbers.

## Phase 16 — Results publishing + scheduled CI

**Goal:** committed, reproducible results that anyone can re-score, with a comparison table and failure analysis.

**Deliverables**
- `results/<YYYY-MM-DD>/<dataset>/<backend>/` holds `report.json` (manifest, traces, scores), the judge cache, and the lock file hash. Only derived data is committed (no raw text from non-commercial datasets beyond short failure excerpts, or ids only for LoCoMo).
- `membench leaderboard results/ --out results/README.md`: backend × metric table with CI and n, cost per 1k queries, and recall p95. Columns are grouped by judge model and prompt version, and rows are never mixed across them.
- `membench report --failures`: per-query failure listing, sorted into not retrieved / stale surfaced / retrieved but answered wrong / false abstention / forget violation, with a context excerpt and evidence ids.
- Comparison write-up (the PRD success metric): Mem0 vs baselines vs the Phase 14 backends, covering method, caveats, cost and failure examples. Every number in it links to a committed result.
- CI:
  - every PR: `membench rescore` over `results/` with `--strict-cache`, no network
  - nightly: a secret-gated smoke run (`mini` on each live backend, judge on) to catch SDK and API drift; it opens an issue on failure
  - weekly canary: the same smoke run on the latest unpinned dependencies
  - full benchmark runs: manual dispatch only (too costly and noisy to run nightly)
- Release to PyPI as v0.2 with the frozen contract.

**Done when**
- `results/README.md` is generated from committed reports, and CI re-derives it identically.
- The nightly smoke run has been green for 7 days; the write-up is published.

---

## Sequencing notes

- Phase 8 comes first because nothing else is trustworthy until a live run completes cleanly.
- Phase 9 is the only breaking change. It comes before the new metrics, datasets and backends, so they are built once against the final contract.
- Reproducibility (10) comes before scale (12). Large runs cost money and must be re-scorable from day one.
- Backends (14) come before judge validation (15), so the human labels cover every context style (raw turns, extracted facts, graph triples).
- Publishing (16) is last. A leaderboard without CIs (13) or a validated judge (15) is exactly the self-reported-number problem the PRD sets out to fix.

## Deliberately excluded

- Hosted leaderboard: out of scope in the PRD. A committed markdown table covers the need.
- Multi-agent memory-sharing scenarios: out of scope in the PRD.
- Fine-tuning or training: out of scope in the PRD.
- PRD Monetization section: not engineering scope.
