# MemBench v0.1 — Phased Implementation Plan

Solo builder, part-time, 4–6 weeks to v0.1. Each phase = one reviewable PR with explicit "done when" criteria. Everything runs offline except the optional LLM-judge (Claude via Anthropic API).

## Guiding constraints

- **Stack:** Python. CLI via Typer (or Click). Packaged for PyPI.
- **Scoring:** hybrid. Deterministic where possible (latency, token counts, exact-fact assertions); LLM-judge (Claude) for semantic recall and contradiction. Judge calls are cached and seedable for CI reproducibility.
- **Adapter budget:** a new backend adapter must fit in <100 lines. Keep the interface thin; push shared logic into the runner, not the adapters.
- **Offline-first:** Phase 1 has no memory-vendor dependency. Full-context and Chroma baselines only.

## Repo layout

```
membench/
  cli.py            # Typer app: run, report, generate, list
  core/
    adapter.py      # MemoryAdapter ABC + Event/RecallResult types
    runner.py       # replays a pack against an adapter, collects traces
    registry.py     # name -> adapter class discovery
    scoring/        # one module per metric + judge client
    report/         # json, markdown, html emitters
  adapters/
    full_context.py # baseline: append-all, recall returns whole log
    chroma.py       # baseline: embed + top-k vector recall
    mem0.py         # (Phase 5)
  packs/
    schema.py       # YAML pack loader + validation
    customer_support.yaml
    coding_assistant.yaml
  generate/         # transcript -> pack generator (Phase 6)
tests/
pyproject.toml
```

## Adapter interface contract

Every backend implements one thin class. The runner owns timing, token accounting, and scoring — adapters only move data.

```
class MemoryAdapter(ABC):
    def write(self, event: Event) -> None          # ingest one fact/turn/tool-result
    def recall(self, query: str) -> RecallResult    # return injected context + items
    def forget(self, key: str) -> None              # remove by stable key
```

- **Event:** `{ session_id, key, role, content, timestamp, metadata }`. `key` is a stable id so `forget` and update-correctness checks can target a specific fact.
- **RecallResult:** `{ context: str, items: list, token_count: int }`. `context` is exactly what would be injected into a prompt — this is what the token-cost metric measures.
- Adapters are stateless config wrappers over a backend client; construction takes a plain dict from pack/CLI config. No scoring logic inside adapters.

## Scenario pack format (descriptive)

- YAML: ordered list of `steps`, each `write` (state / update / contradict a fact) or `query` (with expected-fact assertions and metric hooks).
- Multi-session: steps carry `session_id` so temporal/staleness behavior can be exercised.
- Assertions: deterministic `expect_contains` / `expect_key`, plus `judge` prompts for semantic checks.

---

## Phase 1 — Walking skeleton (no vendor dep)

**Goal:** end-to-end run of one mini pack against two offline adapters, emitting a JSON report.

**Deliverables**
- `pyproject.toml`, package scaffold, `membench` entry point.
- `core/adapter.py`: `MemoryAdapter` ABC, `Event`, `RecallResult`.
- `adapters/full_context.py` and `adapters/chroma.py` (local persistent Chroma, local embeddings).
- `core/runner.py`: replay a pack, capture per-step traces (latency, tokens, recall output).
- `core/registry.py`: resolve `--backend` name to class.
- `packs/schema.py` + one mini pack (~6 steps).
- `core/report/json.py`: raw run + trace to JSON.
- `cli.py`: `membench run --backend <x> --pack <y> --out report.json`.

**Done when**
- `membench run --backend full-context --pack mini` and `--backend chroma` both produce valid JSON offline.
- JSON contains per-step latency + token counts and final metric stubs.
- `pytest` green in CI with no network/API key.

## Phase 2 — Deterministic metrics

**Goal:** compute the four non-judge metrics from run traces.

**Deliverables**
- `core/scoring/`: recall precision (exact-fact assertions), update correctness, token cost (from `RecallResult.token_count`), p95 latency (from traces).
- Scoring wired into runner output; JSON report gains a `scores` block.
- Assertion fields (`expect_contains`, `expect_key`, updated-fact tracking) in pack schema.

**Done when**
- Running the mini pack yields numeric values for all four deterministic metrics.
- Unit tests cover each metric with fixture traces (known inputs → known scores).
- Full-context vs Chroma produce visibly different, sensible scorecards.

## Phase 3 — LLM-judge + reproducibility

**Goal:** add semantic recall and contradiction-rate metrics via a cached, deterministic Claude judge.

**Deliverables**
- `core/scoring/judge.py`: Anthropic client, structured rubric prompts, fixed model + temperature 0.
- On-disk response cache keyed by (prompt hash, model) so CI replays without API calls.
- `contradiction rate` + judge-scored semantic recall metrics.
- CLI flags: `--judge/--no-judge`, `--judge-cache`; graceful skip when no API key.

**Done when**
- With `--no-judge` everything runs offline; with judge + populated cache, CI is deterministic and network-free.
- All five metrics appear in the JSON scorecard.
- Cache hit produces identical scores across two runs (reproducibility test).

## Phase 4 — Reports + scenario packs

**Goal:** ship the two real packs and human-readable reports.

**Deliverables**
- `packs/customer_support.yaml` and `packs/coding_assistant.yaml` — full multi-session scripts (state/update/contradict/query), each covering all five metrics.
- `core/report/markdown.py` and `core/report/html.py`: scorecard tables, per-metric breakdown, adapter comparison.
- `cli.py`: `membench report` (render md/html from JSON), `membench list` (packs + adapters).

**Done when**
- Both packs run clean on both baseline adapters.
- `membench report` emits markdown + self-contained HTML scorecards.
- JSON remains the source of truth (md/html derive from it, no rescoring).

## Phase 5 — Mem0 adapter

**Goal:** first real vendor backend, proving the <100-line target.

**Deliverables**
- `adapters/mem0.py`: `write`/`recall`/`forget` over Mem0 OSS.
- Mem0 as an optional install extra (`membench[mem0]`); import-guarded registry entry.
- Docs snippet: config + running a pack against Mem0.

**Done when**
- `membench run --backend mem0 --pack customer-support` produces a full scorecard.
- Adapter is <100 lines (success-metric check).
- Missing Mem0 install fails with a clear actionable message, not a crash.

## Phase 6 — Transcript scenario generator

**Goal:** "bring your own transcripts" — LLM-assisted pack creation.

**Deliverables**
- `generate/`: transcript loader → Claude prompt → draft pack (facts, updates, contradictions, queries) → schema-valid YAML.
- `cli.py`: `membench generate --transcripts <path> --out pack.yaml`.
- Validation pass so generated packs load in the runner unedited.

**Done when**
- A sample transcript yields a schema-valid pack that runs end-to-end.
- Generated pack exercises at least state + update + contradiction + query steps.
- `--no-judge` path documented (generation needs API key; running the pack does not).

## Phase 7 — Package + release polish

**Goal:** publishable, documented v0.1.

**Deliverables**
- PyPI packaging finalized (extras, entry point, metadata), tagged release.
- README: quickstart, adapter-authoring guide (the <100-line contract), metric definitions.
- CI: lint + tests + offline example run on push.

**Done when**
- `pip install membench` then `membench run --backend full-context --pack customer-support` works from a clean env.
- Adapter-authoring guide lets a reader write a new adapter in <100 lines.
- Green CI; v0.1 tagged.

---

## Sequencing notes

- Phases 1–4 need no API keys or vendor accounts — the offline core is fully testable first.
- Judge (Phase 3) lands before real packs (Phase 4) so packs are authored against all five metrics.
- Mem0 (Phase 5) is deliberately late: the interface and scoring are frozen by then, so it stress-tests the <100-line goal rather than driving the design.
- Reproducibility (judge cache, temperature 0, fixed model, seeds) is a Phase 3 acceptance gate, not an afterthought.
