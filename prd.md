# PRD: MemBench — Eval Harness for Agent Memory Backends

**Owner:** Solo builder (part-time) · **Type:** Open-source CLI + adapter library · **Timeline:** 4–6 weeks to v0.1

## Problem

- Memory is now a first-class agent component, but evaluation is the stated gap.
- Existing benchmarks (LoCoMo, BEAM) are datasets and papers, not tooling.
- They test conversational recall, not production needs: memory across tool calls, document chains, multi-step decisions.
- Teams choosing between Mem0, Cognee, Letta, Zep, or a DIY vector store have no apples-to-apples way to compare on **their** workload.
- Vendors self-report benchmark wins; buyers can't reproduce them.

## Target user

- Engineers selecting or tuning a memory layer for a production agent.
- Memory-vendor OSS users wanting regression tests after config changes.

## Solution (what to build)

A CLI (`membench`) that runs any memory backend through standardized scenarios and outputs a scorecard.

### Core flow
1. Adapter interface: `write(event)`, `recall(query)`, `forget(key)` — one thin adapter per backend.
2. Scenario packs: YAML-defined multi-session scripts (facts stated, updated, contradicted, then queried later).
3. Run: `membench run --backend mem0 --pack customer-support` replays the script against the backend.
4. Score: recall accuracy, contradiction handling, temporal reasoning, staleness, latency, tokens injected per recall.
5. Report: markdown/HTML scorecard + JSON for CI comparison.

### v0.1 scope
- Three adapters: Mem0 (OSS), a naive vector-store baseline (Chroma), and raw full-context baseline.
- Two scenario packs: customer-support persona, coding-assistant project context.
- Five metrics: recall precision, update correctness, contradiction rate, token cost, p95 latency.
- Scenario generator: LLM-assisted pack creation from a user's sample transcripts.

### Out of scope (v0.1)
- Building a memory system itself.
- Hosted leaderboard.
- Multi-agent memory sharing scenarios.
- Fine-tuning or training anything.

## Differentiation

- Whitespace: no OSS tool does backend-agnostic memory evals today; benchmarks exist only as static datasets.
- Tests tool-call and task memory, not just chat recall.
- "Bring your own transcripts" — evals on the user's real workload, not academic data.

## Success metrics

- Add a new backend adapter in < 100 lines of code.
- Published comparison post (Mem0 vs baseline vs full-context) that drives initial traffic.
- One memory vendor references or contributes to the harness in 3 months.

## Monetization / FDE angle

- Paid service: memory-layer selection audits for companies deploying agents.
- Interview asset: demonstrates evals + context/RAG depth — the exact FDE skill pair.