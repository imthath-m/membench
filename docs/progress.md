# Build progress — roadmap Phases 8–16
Resume protocol: list origin branches `claude/gracious-edison-v8eyxs*`; open the highest phase branch; read this file there; continue the first phase not marked done.

| Phase | Branch | PR | Status |
|---|---|---|---|
| 8 | claude/gracious-edison-v8eyxs | #6 | done (live run blocked) |
| 9 | claude/gracious-edison-v8eyxs-phase-9 | #7 | done |
| 10 | claude/gracious-edison-v8eyxs-phase-10 | — | in progress |

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
Status: in progress
- [ ] Run manifest in every report (schema_version, membench version + git commit/dirty, Python/OS, backend SDK versions, redacted backend config, pack name + sha256, dataset slot, judge/reader/backend models + base URLs, judge prompt version, tokenizer, seed, repeat index, deployment, start/finish)
- [ ] `JUDGE_PROMPT_VERSION` in manifest + cache key; cache stores raw response + `usage`
- [ ] Judge `--strict-cache` (cache miss is an error)
- [ ] `membench rescore --in report.json` (no adapter, no network)
- [ ] `uv.lock` + frozen sync for published (live) runs; floor versions kept
- [ ] Cost tracking: judge / reader / backend tokens from API `usage`; dated price table → USD per run and per 1k queries (Mem0 `usage()`)
- [ ] Determinism: seed, embedding model id + temperature recorded; non-determinism documented
- [ ] Committed JSON Schema; every report validates (test)
- [ ] CI test: `rescore` on a committed report reproduces identical scores offline
