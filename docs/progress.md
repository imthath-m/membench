# Build progress — roadmap Phases 8–16
Resume protocol: list origin branches `claude/gracious-edison-v8eyxs*`; open the highest phase branch; read this file there; continue the first phase not marked done.

| Phase | Branch | PR | Status |
|---|---|---|---|
| 8 | claude/gracious-edison-v8eyxs | #6 | done (live run blocked) |

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
