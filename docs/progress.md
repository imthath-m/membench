# Build progress — roadmap Phases 8–16
Resume protocol: list origin branches `claude/gracious-edison-v8eyxs*`; open the highest phase branch; read this file there; continue the first phase not marked done.

| Phase | Branch | PR | Status |
|---|---|---|---|
| 8 | claude/gracious-edison-v8eyxs | — | in progress |

## Phase 8 — Hardening + first live run
Status: in progress
- [ ] CLI builds adapter + `reset()` inside error handling; one-line error for missing extra; test with `mem0` import blocked; fix README and `docs/mem0.md`
- [ ] `core/retry.py`: exponential backoff + jitter; adapters declare `retryable` exception tuple (Mem0: OpenAI rate limit / timeout)
- [ ] Per-step timeout `--step-timeout` (default 120s)
- [ ] Per-step error capture: trace `error`, run continues, report `error_rate`, run marked invalid above threshold
- [ ] Judge: unparseable reply retried once then `judge_error`; unexpected response shape → `judge_error`, no retry
- [ ] Tests: malformed judge reply (no JSON, bad JSON), unexpected shape (no `choices`), mem0 blocked import
- [ ] CI: `chroma` job
- [ ] CI: `live.yml` (manual dispatch, secret-gated) runs `mini` against `mem0` with judge
