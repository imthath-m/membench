# MemBench comparison: memory backends vs baselines (draft)

**Status: draft, offline baselines only.** The Mem0 rows and the Phase 14
hosted-backend rows (Mem0 Platform, Zep, Letta, Letta agent, LangMem, Cognee)
are **pending live runs**: they need API keys this draft was written without
(see "Pending" below). No judge or reader ran, so semantic recall,
contradiction rate, QA accuracy and abstention are absent, not estimated.
Every number below links to the committed `report.json` it comes from; the
full tables are in [results/README.md](../results/README.md).

## Method

- **Packs:** two seeded synthetic packs (generator v2, `--pack-seed 0`):
  `tool-call-memory` (facts arrive as tool outputs) and `customer-support-xl`
  (a support conversation). Each has 36 episodes, 216 queries and 1,188 steps,
  with update chains, time gaps, multi-hop, abstention and forget queries plus
  distractor turns. Packs are generated on load, never committed.
- **Backends:** `full-context` (the whole history, timestamped), `bm25`
  (lexical top-3), `chroma` (local vector store, `all-MiniLM-L6-v2`, top-3),
  and `oracle` (only each query's evidence events: an upper bound, not a
  competitor). Default configs, one run each (deterministic offline
  backends), `--seed 0`, tokenizer tiktoken `o200k_base`.
- **Scoring:** deterministic metrics only (recall precision, update
  correctness, staleness, temporal accuracy, forget compliance, evidence
  recall@k), each with a 95% CI (Wilson, or a seeded bootstrap) and n.
- **Reproduce:** `membench run --backend <b> --pack <pack> --no-judge`, then
  `membench publish`; or re-score the committed result with no network:
  `membench rescore --in results/2026-11-09/<pack>/<b>/report.json --judge-cache
  results/2026-11-09/<pack>/<b>/judge_cache.json --check`. CI does this for
  every committed result on every PR.

## Results

| | tool-call-memory | customer-support-xl |
| --- | --- | --- |
| **Recall precision**, bm25 | [51% [43–59%]][tcm-bm25] | [58% [50–66%]][cs-bm25] |
| chroma | [94% [89–97%]][tcm-chroma] | [88% [81–92%]][cs-chroma] |
| full-context | [100% [97–100%]][tcm-fc] | [100% [97–100%]][cs-fc] |
| **Staleness ↓**, bm25 | [69% [53–82%]][tcm-bm25] | [78% [62–88%]][cs-bm25] |
| chroma | [97% [86–100%]][tcm-chroma] | [100% [90–100%]][cs-chroma] |
| full-context | [100% [90–100%]][tcm-fc] | [100% [90–100%]][cs-fc] |
| oracle (upper bound) | [0% [0–10%]][tcm-oracle] | [0% [0–10%]][cs-oracle] |
| **Evidence recall@k**, bm25 | [58% [50–65%]][tcm-bm25] | [58% [50–67%]][cs-bm25] |
| chroma | [88% [83–92%]][tcm-chroma] | [92% [88–95%]][cs-chroma] |
| **Temporal accuracy**, bm25 | [56% [40–70%]][tcm-bm25] | [42% [27–58%]][cs-bm25] |
| chroma | [86% [71–94%]][tcm-chroma] | [83% [68–92%]][cs-chroma] |
| **Tokens / query**, full-context | [835.1][tcm-fc] | [785.1][cs-fc] |
| chroma | [41.0][tcm-chroma] | [32.2][cs-chroma] |
| bm25 | [32.4][tcm-bm25] | [32.3][cs-bm25] |

n = 144 queries for recall precision and evidence recall@k, 36 for staleness
and temporal accuracy. All of these are below the 385-query reporting
minimum, so every number here is **indicative**.

## Findings

- **Retrieval is the gap on these packs.** Full context recalls everything
  ([100%][tcm-fc]) at ~20× the tokens of a top-3 retriever
  ([835][tcm-fc] vs [41][tcm-chroma] tokens/query on tool-call-memory).
  Chroma closes most of the recall gap ([94%][tcm-chroma]); BM25 does not
  ([51%][tcm-bm25]): distractor turns share the query's words.
- **Nobody but the oracle avoids stale values.** Every backend that stores the
  raw turns surfaces a superseded value on almost every post-update query
  (chroma [97%][tcm-chroma] / [100%][cs-chroma], full context
  [100%][tcm-fc]): a backend that keeps all turns and leaves conflict
  resolution to the reader will fail staleness by design. That is the
  behaviour the memory systems (Mem0, Zep, Letta) claim to fix, and what the
  pending rows will test.
- **Temporal questions need dates in the context.** Chroma and BM25 return
  bare text with no timestamps; full context prints them
  ([100%][tcm-fc] vs chroma [86%][tcm-chroma] temporal accuracy).
- **Cost:** $0 per 1k queries for all four rows (no LLM anywhere); the
  memory-system rows will add backend extraction and embedding spend.
- **Latency:** recall p95 is under 1 ms for the in-memory baselines and
  ~290 ms for chroma ([286 ms][tcm-chroma], [295 ms][cs-chroma]) on a 4-CPU
  container: local ONNX embedding of each query, not a network round trip.

## Failure examples

From `membench report --in <report.json> --failures --out failures.md`:

- **Stale surfaced** (chroma, customer-support-xl, step 29, "In which
  language should support write to this customer?"): recalled `e0-w4`,
  `e0-w5`, `e0-w6` — German, then Spanish, then French. The current value is
  there, but so are both superseded ones. [report][cs-chroma]
- **Not retrieved** (chroma, tool-call-memory, step 127, multi-hop "What
  timezone is the person handling OPS-91 in?"): evidence `e3-w2`, `e3-w3`;
  recalled three repeated `get_time()` tool outputs instead. [report][tcm-chroma]
- **Forget violation, a metric artifact** (full-context, customer-support-xl,
  step 98): `Pro` was forgotten and is gone from the context, but the
  case-insensitive substring check matches "ap*pro*ved" and "*pro*blem". This
  explains full context's [92%][cs-fc] forget compliance on this pack; it is a
  known issue in `contains_any` (word-boundary matching is the fix), not a
  backend failure.

Failure counts (not retrieved / stale surfaced): bm25 70 / 25, chroma 8 / 35,
full context 0 / 36 on tool-call-memory; bm25 60 / 28, chroma 18 / 36, full
context 0 / 36 on customer-support-xl. The oracle has none.

## Caveats

- Synthetic, template-generated packs: easier and more regular than real
  conversations. LongMemEval / LoCoMo results are not committed yet (a
  LongMemEval_S full-context report is ~600 MB; LoCoMo's licence allows
  ids only in committed results, which a re-scorable report cannot be).
- n is below the reporting minimum everywhere (indicative only); one run per
  config because the offline backends are deterministic.
- No judge, so no judge metrics; when they arrive they are marked
  `unvalidated` until the human-label validation (Phase 15) is committed.
- Staleness and forget compliance use substring matching on short values,
  which over-counts (see the forget example above).

## Pending

- Mem0 (OSS): `gh workflow run live.yml -f backend=mem0 -f pack=tool-call-memory`
  (secrets `OPENAI_API_KEY`, `MEMBENCH_JUDGE_API_KEY`).
- Phase 14 backends: `gh workflow run live.yml -f backend=all -f pack=tool-call-memory`
  (also `MEM0_API_KEY`, `ZEP_API_KEY`, `LETTA_API_KEY`); vendor-tuned rows with
  `-f config=vendor-tuned` once vendors reply (docs/vendor-review.md).
- Repeat for `customer-support-xl`, commit each job's `results/` artifact,
  regenerate `results/README.md`, then update this write-up with the new rows.

[tcm-bm25]: ../results/2026-11-09/tool-call-memory/bm25/report.json
[tcm-chroma]: ../results/2026-11-09/tool-call-memory/chroma/report.json
[tcm-fc]: ../results/2026-11-09/tool-call-memory/full-context/report.json
[tcm-oracle]: ../results/2026-11-09/tool-call-memory/oracle/report.json
[cs-bm25]: ../results/2026-11-09/customer-support-xl/bm25/report.json
[cs-chroma]: ../results/2026-11-09/customer-support-xl/chroma/report.json
[cs-fc]: ../results/2026-11-09/customer-support-xl/full-context/report.json
[cs-oracle]: ../results/2026-11-09/customer-support-xl/oracle/report.json
