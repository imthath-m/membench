# MemBench walkthrough

A guided tour for someone who did not write this repo. Every claim cites the
file it came from. Nothing here is a live result: the repo has never run
against a real memory service or a real LLM (see
[HANDOFF.md](HANDOFF.md), "Current state").

Start here, then go deeper in [README.md](../README.md),
[HANDOFF.md](HANDOFF.md) and [progress.md](progress.md).

## 1. What this repo is

- MemBench is an eval harness for agent **memory backends** (systems that store what an agent was told and recall it later).
- A scripted scenario (a "pack") states, updates, contradicts, forgets and queries facts across sessions.
- The harness replays the pack against one backend, then scores what the backend recalled.
- Why: to compare backends on the same footing (recall, stale values, forgetting, time, cost, latency) with confidence intervals, not anecdotes.
- Everything runs offline except the optional LLM judge, the optional reader stage, and LLM-backed or hosted backends ([README.md](../README.md), top).

## 2. Repo map

| Path | What it is |
|---|---|
| `membench/cli.py` | The `membench` command (typer): `run`, `generate`, `report`, `leaderboard`, `compare`, `rescore`, `publish`, `convert`, `judge-sample`, `judge-labels`, `judge-cross`, `judge-agreement`, `list`. |
| `membench/packs/` | `schema.py` (pack loader and validator) and three bundled packs: `mini`, `customer_support`, `coding_assistant`. |
| `membench/datasets/` | Loaders for LongMemEval and LoCoMo; `resolve_pack()` turns `--pack` into a `Pack`. Also lists the synthetic pack names. |
| `membench/generate/` | `synthetic.py` (seeded, template-based packs, no LLM), `generator.py` (LLM pack generation from transcripts, see [generate.md](generate.md)). |
| `membench/adapters/` | One file per backend: `full_context`, `bm25`, `oracle`, `chroma` (offline); `mem0`, `mem0_platform`, `zep`, `letta`, `letta_agent`, `langmem`, `cognee` (need keys). |
| `membench/core/adapter.py` | The adapter contract: `write`, `recall`, `forget`, `flush`, `reset`, `usage`. |
| `membench/core/runner.py` | `run_pack()`: replays steps, times them, retries, records traces, scores. |
| `membench/core/scoring/` | One module per metric, plus `judge.py` (LLM judge and its cache). |
| `membench/core/validation/` | Judge validation: `sample.py`, `labels.py`, `cross.py`, `agreement.py`, `status.py`. |
| `membench/core/report/` | `json.py`, `markdown.py`, `html.py`, `compare.py`, `failures.py`, `leaderboard.py`. |
| `membench/core/` (other) | `stats.py` (CIs, tests), `cost.py` + `prices.yaml`, `manifest.py`, `reader.py`, `publish.py`, `tokens.py`. |
| `membench/report.schema.json` | JSON Schema of `report.json`. |
| `configs/` | `default/<backend>.yaml` (default vendor configs), `judges.yaml` (cross-judge panel). |
| `results/` | Committed offline baseline results, one dir per run; `results/README.md` is generated. |
| `validation/` | `sample.jsonl`: items to label by hand. No labels or agreement file yet. |
| `docs/` | [HANDOFF.md](HANDOFF.md), [progress.md](progress.md), [judge-validation.md](judge-validation.md), [labelling-guidelines.md](labelling-guidelines.md), [roadmap.md](roadmap.md), [comparison.md](comparison.md) (draft write-up), one page per backend. |
| `tests/` | pytest suite (offline; live tests skip without `MEMBENCH_LIVE=1`). |
| `.github/workflows/` | `ci.yml`, `live.yml`, `smoke.yml`, `judge-validation.yml`, `release.yml`. |

## 3. Data flow, end to end

```text
 pack (YAML / synthetic / dataset)
   membench.datasets.resolve_pack()  ->  Pack (ordered write / query / forget steps)
        |
        v
 adapter (memory backend)            membench.core.registry.get_adapter_class(name)
   write / recall(at=) / forget / flush / reset / usage
        |
        v
 run                                 membench.core.runner.run_pack(pack, adapter, ...)
   one StepTrace per step (latency, context, items, error)
   optional: Reader.answer() per query (--reader-model)
        |
        v
 scoring                             membench.core.scoring.score_detailed(pack, traces, judge)
   deterministic: recall_precision, update_correctness, staleness, ...
   judge:         semantic_recall, contradiction_rate, qa_accuracy
        |                            (judge = CachedJudge, on-disk cache)
        v
 report.json                         membench.core.report.json.write_json(RunResult)
   scores, by_category, outcomes, stats (n + 95% CI), manifest, cost, traces
        |
        +--> membench report       (md / html / --failures)
        +--> membench rescore      (re-score from traces + judge cache, offline)
        +--> membench compare      (paired test of two runs on one pack)
        +--> membench publish      -> results/<date>/<dataset>/<backend>/
                                       -> membench leaderboard -> results/README.md
```

Step by step:

1. **Pack in.** `membench run --pack X` calls `resolve_pack()` in `membench/datasets/__init__.py`. `X` is a YAML path, a bundled name, a dataset (`longmemeval-s`, `locomo`, ...) or a synthetic pack (`tool-call-memory`, `customer-support-xl`, ...; seeded by `--pack-seed`). Schema: `membench/packs/schema.py`.
2. **Adapter.** `get_adapter_class(backend)` (`core/registry.py`) returns a `MemoryAdapter` subclass. Adapters only move data; they never see scoring ground truth (`core/adapter.py` docstring). `Event.id` is an opaque id; the pack's `fact_key` stays with the scorer.
3. **Run.** `run_pack()` (`core/runner.py`) walks the steps. `reset()` runs between episodes. `flush()` runs before a query that follows writes. Each query calls `recall(query, session_id=, at=)`. Failed steps are recorded in `StepTrace.error` and the run continues. A run with `error_rate` above `MAX_ERROR_RATE` (0.1) is marked `valid: false`.
4. **Reader (optional).** With `--reader-model`, `Reader.answer()` (`core/reader.py`) answers each query that has a gold `answer` or `abstain` from the recalled context. The answer is stored in the trace as `reader_answer`.
5. **Scoring.** `score_detailed()` (`core/scoring/__init__.py`) runs every metric in the `OUTCOMES` table, then latency and `error_rate`. It also builds the per-category table and `stats` (n, 95% CI, "indicative" flag) via `core/stats.py`.
6. **Report.** `RunResult` is written by `write_json()` (`core/report/json.py`). The manifest (`core/manifest.py`) records pack sha256, judge model and prompt version, seed, versions, tokenizer. `cost` comes from `core/cost.py`.
7. **Afterwards.**
   - `rescore` (`cli.py`) rebuilds the pack from the manifest, checks its sha256, and re-scores from the stored traces. Judge verdicts come from the cache only (strict). `--check` exits 1 on any difference.
   - `publish` refuses a report that does not re-score identically, then copies it (plus only the judge-cache entries it uses) to `results/<date>/<dataset>/<backend>/`.
   - `leaderboard` (`core/report/leaderboard.py`) renders `results/README.md` from those dirs. Rows with different judge models, prompt versions or readers never share a table.
   - `compare` (`core/report/compare.py`) pairs per-query outcomes of two runs: paired bootstrap CI, McNemar exact test, Holm correction. It refuses runs on different packs or different judge/reader versions.

## 4. Metrics

All names are the keys in a report's `scores` block (`core/scoring/__init__.py`, README "Metric definitions"). "Det." = deterministic, no network. A `null` score means the pack or run did not exercise it.

| Metric | Meaning | How computed | Code | Caveat |
|---|---|---|---|---|
| `recall_precision` | Did recall surface the expected facts? | Det. Fraction of queries with `expect_contains` whose context holds every string (case-insensitive substring). | `scoring/recall_precision.py` | Plain substring: an expected value inside a longer word still counts. |
| `update_correctness` | After a fact changed, does recall show the new value? | Det. Post-update queries (same episode, `expect_key` was updated or contradicted) whose context holds the `expect_contains` strings. | `scoring/update_correctness.py` | Only checks the new value is present. Pair it with `staleness`. |
| `staleness` (lower is better) | Does recall still show the old value? | Det. Of the post-update queries with `expect_absent`, the fraction whose context holds a stale string (whole-word match). | `scoring/staleness.py` | Full-context scores 100% here (it keeps everything). Not a bug. |
| `temporal_accuracy` | Can it answer "what held on date D"? | Det. `category: temporal` queries whose context holds all `expect_contains`. Query carries a timestamp, passed to `recall(at=)`. | `scoring/temporal_accuracy.py` | Low by construction for backends whose context has no dates (HANDOFF: Chroma, Mem0). |
| `forget_compliance` | Is deleted content really gone? | Det. `category: forget` queries where no recalled item traces to a forgotten event id and no `expect_absent` string appears. Failed recalls are skipped. | `scoring/forget_compliance.py` | Depends on the adapter reporting event ids on items. |
| `evidence_recall_at_k` | Did recall return the writes that answer the query? | Det. Mean fraction of the query's `evidence` event ids found in the recalled items' `event_ids` (k = whatever the backend returned). | `scoring/evidence_recall.py` | Reads 0 for `letta-agent` and for `cognee` with a non-`CHUNKS` search (no event ids). |
| `abstention` | Does the reader say "I don't know" when no answer exists? | Det. on the stored reader answer: regex for "I don't know" / "unknown" / "not mentioned" on `abstain: true` queries. | `scoring/abstention.py` | Needs `--reader-model` (an LLM call), else `null`. The regex is narrow. |
| `semantic_recall` | Does the recalled context convey the fact, even reworded? | Judge. Per query with a `judge.semantic_recall` rubric: pass rate. A failed recall counts as a miss. | `scoring/semantic_recall.py` | Unvalidated: no human labels yet. |
| `contradiction_rate` (lower is better) | Does recall present conflicting current values? | Judge. Per query with a `judge.contradiction` rubric: fraction where the judge says the contradiction is present. | `scoring/contradiction_rate.py` | Unvalidated. Polarity flipped: `passed` means "contradiction present". |
| `qa_accuracy` | Is the reader's answer right? | Judge. Reader answer vs gold `answer`. LongMemEval packs use that benchmark's official yes/no prompt and also score abstain queries. | `scoring/qa_accuracy.py` | Needs reader and judge. Unvalidated. |
| `token_cost` | Prompt tokens injected per query. | Det. Mean `token_count` of the recalled context, counted by the runner. | `scoring/token_cost.py` | Without `tiktoken` installed, counts are a whitespace approximation (marked `approx`). |
| `write_` / `ingest_` / `recall_` `p50_ms`, `p95_ms` | Latency per stage (write, flush, recall). | Det. Nearest-rank percentiles over successful steps. p95 is `null` below 100 samples. | `scoring/latency.py` | Backoff sleeps count as latency. |
| `error_rate` | Share of steps that failed or timed out. | Det. Errored steps over all steps. | `scoring/error_rate.py` | Above 0.1 the run is `valid: false`. |
| cost (`cost.*`, not in `scores`) | USD for judge, reader and backend tokens. | Det. Token counts times `membench/prices.yaml` rates. | `core/cost.py` | Models missing from the table price as `usd: null`. OpenAI rows were not checked against the live page (HANDOFF). |

Also in `report.json`: `by_category` (recall_precision, evidence_recall_at_k, semantic_recall, qa_accuracy per query category) and `stats` (n, 95% CI: Wilson for pass/fail, seeded bootstrap otherwise). Reporting minimums are 385 queries overall and 100 per category (`core/stats.py`); below that a cell is marked indicative.

## 5. The judge

Code: `membench/core/scoring/judge.py`.

- **What it asks.** `build_prompt(rubric, evidence)` sends a "strict evaluator" prompt: decide whether the STATEMENT (the rubric) is true of the EVIDENCE (the recalled context), reply with JSON `{"verdict": "pass" | "fail", "reason": ...}`, and "pass" only if the evidence *clearly* supports it.
- **Parsing.** `parse_verdict` finds the first `{` to the last `}` and reads `verdict`. Anything other than "pass" fails.
- **Model.** `DEFAULT_JUDGE_MODEL = "claude-sonnet-5"` (Anthropic Messages API). Pass `--judge-base-url` to use any OpenAI-compatible endpoint (`OpenAICompatibleClient`).
- **Temperature.** 0, set in both clients. Anthropic calls use `max_tokens` 256.
- **Retries.** HTTP 429 and 5xx retried up to 5 times with backoff, honouring `Retry-After`. A reply that fails to parse is retried once, then recorded as a judge error on the trace (`detail.judge_error`) and skipped, not scored.
- **Cache.** `JudgeCache` is a JSON file (`--judge-cache`, default `.membench_judge_cache.json`).
  - Key: `<model>:v<JUDGE_PROMPT_VERSION>:<sha256(model, version, full prompt)>`. The full prompt contains rubric and evidence.
  - Entry: verdict, reason, raw reply, token usage (or the error).
  - Why: a repeat or replay makes zero API calls, and verdicts from different models or prompt versions can never be mixed.
  - Note: the key does not include `base_url`.
- **Strict / offline mode.** `--strict-cache` (and every `rescore` / `publish`) builds a judge with no client. A cache hit is returned; a miss raises `CacheMissError` and the command fails. No key needed.
- **`JUDGE_PROMPT_VERSION`** is `"1"`. It is in the cache key and the manifest. Bump policy ([judge-validation.md](judge-validation.md)):
  - Any edit to `build_prompt`, `parse_verdict`, `qa_rubric` or the LongMemEval judge templates must bump it.
  - `tests/test_judge_validation.py` pins a hash of the prompts and fails until the bump and new hash are in.
  - After a bump, re-run cross-judge and agreement on the same labelled items; until a new `agreement-v<N>.json` is committed, every judge metric of that version is `unvalidated`. `compare` refuses to pair runs of different versions, and `rescore` refuses a report made with another version.
- **API key.** Env var `MEMBENCH_JUDGE_API_KEY` (`API_KEY_ENV`). The reader reads `MEMBENCH_READER_API_KEY`, falling back to the judge key.
- **No key.** Not an error. `membench run` prints "Judge enabled but no API key..." and the judge metrics score `null`; everything else runs. (`--judge` is on by default, so offline runs should pass `--no-judge` to be explicit.)

## 6. Judge validation

Goal: find out whether the judge agrees with people before trusting `semantic_recall`, `contradiction_rate` and `qa_accuracy`. Full spec: [judge-validation.md](judge-validation.md); annotator rules: [labelling-guidelines.md](labelling-guidelines.md).

Pipeline:

| Step | Command | Output |
|---|---|---|
| 1. Sample | `membench judge-sample --in run.json [--in ...] [--n 200]` | `validation/sample.jsonl`: up to 200 distinct items per judge metric, round-robin over backend x verdict x length bucket. |
| 2. Label (by hand) | Two annotators, working alone, label `true` / `false` per item. Adjudicator settles disagreements. | `a.jsonl`, `b.jsonl`, `adjudication.jsonl` |
| 3. Merge | `membench judge-labels --annotator a=a.jsonl --annotator b=b.jsonl [--adjudication adj.jsonl]` | `validation/labels.jsonl` |
| 4. Cross-judge | `membench judge-cross --judges configs/judges.yaml` | `validation/verdicts-v<N>.jsonl`. Panel: `claude-sonnet-5`, `gpt-4.1`, `meta-llama/Llama-3.3-70B-Instruct-Turbo`. Needs three keys. |
| 5. Agreement | `membench judge-agreement --labels ... --verdicts ...` | `validation/agreement-v<N>.json` and `.md` |

Plain-words terms:

- **Kappa (κ):** agreement between two raters beyond what chance would give. 1 is perfect, 0 is chance-level. Computed in `agreement.py` (`cohen_kappa`; Light's κ, the mean of pairwise κ, with more than two annotators).
- **False pass:** the judge said pass, the humans said fail. Rate = false passes / items humans failed. This is the dangerous direction: it inflates a backend's score.
- **False fail:** the judge said fail, the humans said pass. Rate = false fails / items humans passed.

Thresholds (`membench/core/validation/agreement.py`). A judge is `validated` on a metric only if all hold:

- human-human κ at least `MIN_HUMAN_KAPPA` = 0.7
- judge-human κ at least `MIN_JUDGE_KAPPA` = 0.6
- judge-human κ no more than `MAX_KAPPA_GAP` = 0.1 below human-human κ
- at least `MIN_VALIDATION_N` = 200 settled items
- no annotator disagreement left unadjudicated

Length buckets: items are bucketed by whitespace tokens of the judged evidence: `<128`, `128-511`, `512-2047`, `2048+` (`validation/sample.py`). Agreement is reported per bucket. Reason: **length bias**. A judge may pass long full-context dumps more readily than short top-k contexts. The breakdown does not gate validation by itself; a clearly worse bucket is a reason to rework the prompt.

Status: **not run.**

- No human labels exist. `validation/labels.jsonl`, `validation/verdicts-v1.jsonl` and `validation/agreement-v1.json` do not exist.
- The committed `validation/sample.jsonl` has 400 items (200 `semantic_recall`, 200 `contradiction_rate`; 0 `qa_accuracy`) from synthetic packs, with `verdict: null` on every row (sampled with no judge). All judge metrics are therefore `unvalidated`, and reports and the leaderboard say so (`core/validation/status.py`).
- Cross-judge was never run (needs `MEMBENCH_JUDGE_API_KEY`, `OPENAI_API_KEY`, `TOGETHER_API_KEY`).
- Reports' judge metrics read `unvalidated` until `validation/agreement-v1.json` names the run's judge model as validated.

Forward pointer: A browser labelling tool (`membench judge-label`) and a prompt-tuning harness (`membench judge-tune`, see docs/prompt-tuning.md) are added in the same change as this walkthrough; see those docs for commands.

## 7. Demo in 5 minutes (offline, no API key)

Every command below was run for this page on Python 3.13, from the repo root. Times are wall-clock on a small cloud VM, deps already cached; yours will differ. Outputs go to `/tmp/membench-demo`, not into the repo.

```bash
# 1. Install (about 4 s with a warm pip cache; longer on a cold one)
pip install -e ".[dev]"
export DEMO=/tmp/membench-demo && mkdir -p "$DEMO"

# 2. See what is available (under 1 s)
membench list

# 3. Run bundled and synthetic packs on offline backends, no judge (about 1 s each)
membench run --backend bm25 --pack mini --no-judge --out "$DEMO/mini.json"
membench run --backend bm25 --pack tool-call-memory --no-judge --out "$DEMO/bm25.json"
membench run --backend full-context --pack tool-call-memory --no-judge --out "$DEMO/full.json"

# 4. Read the report, the failure list, and a paired comparison (under 1 s each)
membench report --in "$DEMO/bm25.json" --format md --out "$DEMO/bm25.md"
membench report --in "$DEMO/bm25.json" --failures --out "$DEMO/failures.md"
membench compare --a "$DEMO/bm25.json" --b "$DEMO/full.json" --out "$DEMO/compare.md"
head -30 "$DEMO/bm25.md"

# 5. Re-score offline and check nothing moved (under 1 s each)
membench rescore --in "$DEMO/bm25.json" --check
R=results/2026-11-09/tool-call-memory/bm25
membench rescore --in $R/report.json --judge-cache $R/judge_cache.json --check

# 6. Re-score every committed result (about 4 s for the 8)
for r in results/*/*/*/report.json; do
  membench rescore --in "$r" --judge-cache "$(dirname "$r")/judge_cache.json" --check
done

# 7. Regenerate the leaderboard and confirm it matches the committed one (under 1 s).
#    Links are relative to the output file, so work on a copy of results/.
cp -r results "$DEMO/results"
membench leaderboard "$DEMO/results" --out "$DEMO/results/README.md"
diff "$DEMO/results/README.md" results/README.md && echo "leaderboard identical"

# 8. Sample judge items without a judge (under 1 s). No judge cache is passed,
#    so every item's verdict stratum is "unjudged".
membench judge-sample --in "$DEMO/bm25.json" --in "$DEMO/full.json" --n 20 --out "$DEMO/sample.jsonl"

# 9. Look at the committed sample (not labelled: verdict is null on every row)
python - <<'PY'
import collections, json
rows = [json.loads(l) for l in open("validation/sample.jsonl")]
print(len(rows), "items")
print(collections.Counter(r["metric"] for r in rows))
print(collections.Counter(r["length_bucket"] for r in rows))
print(collections.Counter(r["verdict"] for r in rows))
PY

# 10. Tests and lint (about 15 s for pytest; live-backend tests are skipped)
python -m pytest -q
ruff check .
```

What you should see:

- Step 3 prints `Wrote report for pack=... backend=... to ...`. It also prints "tiktoken unavailable ...; token counts are approximate" unless you install `.[tokens]`. With the approximation, `token_cost` differs from the committed results (which used tiktoken), but nothing else does.
- Step 5 and 6 print `Re-scored: identical to the report's scores.`
- Step 7 prints `leaderboard identical`.
- Step 8 warns that there are no `qa_accuracy` items (that needs `--reader-model`, which needs a key).
- Step 9 prints 400 items; buckets `<128` and `128-511` only; verdict `None` for all 400.
- `judge-sample` has no `--no-judge` flag. Not passing a judge cache is how you get an unjudged sample.

## 8. API key versus offline

Key names are from [HANDOFF.md](HANDOFF.md) "Keys & setup".

| Task | Needs a key? | Which |
|---|---|---|
| `run` with `full-context`, `bm25`, `oracle`, `chroma` + `--no-judge` | No | `chroma` needs `pip install ".[chroma]"` and is slow (HANDOFF: about 5 writes/s); not run in this demo |
| `run` with the judge on | Yes | `MEMBENCH_JUDGE_API_KEY` (without it: runs, judge metrics `null`) |
| `run --reader-model` (`qa_accuracy`, `abstention`) | Yes | `MEMBENCH_READER_API_KEY` or `MEMBENCH_JUDGE_API_KEY` |
| `run` with `mem0`, `langmem`, `cognee` | Yes | `OPENAI_API_KEY` (cognee reads `LLM_API_KEY`) |
| `run` with `mem0-platform`, `zep`, `letta`, `letta-agent` | Yes | `MEM0_API_KEY`, `ZEP_API_KEY`, `LETTA_API_KEY` |
| `report`, `compare`, `leaderboard`, `convert` (no paraphrase) | No | |
| `rescore`, `publish` (always strict cache) | No | |
| `run --strict-cache` | No | Replays judged runs from a cache |
| `generate`, `convert --paraphrase-model` | Yes | `MEMBENCH_JUDGE_API_KEY` |
| `judge-sample`, `judge-labels`, `judge-agreement` | No | (`judge-sample` needs a report; labels need people) |
| `judge-cross` | Yes | `MEMBENCH_JUDGE_API_KEY`, `OPENAI_API_KEY`, `TOGETHER_API_KEY`; `--strict-cache` replays offline |
| LongMemEval / LoCoMo packs | No key, but a download | Cached under `~/.cache/membench` |
| `pytest`, `ruff` | No | Live tests need `MEMBENCH_LIVE=1` |

## 9. Known limitations and caveats

From [HANDOFF.md](HANDOFF.md), [progress.md](progress.md), [results/README.md](../results/README.md) and the code.

- **Nothing live has run.** No real Mem0, Zep, Letta, LangMem or Cognee run, no real judge run, no reader run, no cost check against provider dashboards. Their adapters are covered offline by SDK fakes only.
- **Judge metrics are unvalidated.** No human labels, no agreement file. Do not cite `semantic_recall`, `contradiction_rate` or `qa_accuracy` as accurate.
- **Committed results have no judge or reader.** `results/` holds only deterministic metrics for four offline backends on two synthetic packs, one run each, no CIs across repeats.
- **Small n.** Committed results have 216 queries (144 for some metrics, 36 for others), below the 385-query headline minimum; every cell is marked indicative.
- **The sample has no long items.** `validation/sample.jsonl` has nothing at 512 tokens or more, so length bias cannot be checked from it. It also has no `qa_accuracy` items.
- **Judge variance is not captured.** Verdicts are cached by prompt, so `--repeats` measures backend and reader variance only.
- **Synthetic packs are easy.** They are template-generated, more regular than real conversations. Results on them are not a claim about real use.
- **`oracle` is an upper bound**, not a competitor. `full-context` scores well on recall but badly on `staleness` and token cost.
- **Some metrics are meaningless for some backends** (event ids missing, undated contexts); see section 4.
- **String matching is crude.** `expect_contains` is a plain substring; `expect_absent` is whole-word. Both are deliberate choices to avoid penalising near-misses.
- **No scoring version in the manifest.** A metric change shows up as `rescore --check` diffs; results must then be re-scored and re-published.
- **Price table.** OpenAI rows were not checked against the live page; unknown models price as `usd: null`.
- **Big reports.** Contexts are stored on traces. LongMemEval_S on full-context is about 600 MB, so dataset results are not committed (LoCoMo's licence also forbids it).
- **Stack not merged.** At the time of writing `main` is v0.1; the v0.2 work is in an open PR stack, and the scheduled and dispatched workflows run only from `main`. The PyPI release is not published.
- **Tokenizer.** Without `tiktoken`, token counts are whitespace-based and marked `approx`.

## 10. Glossary

- **Pack:** a scripted scenario (YAML or generated) of write, query and forget steps with expectations.
- **Episode:** an independent run of steps inside a pack; the adapter is `reset()` between episodes.
- **Adapter:** thin wrapper that lets a memory backend be driven through `write` / `recall` / `forget` / `flush` / `reset` / `usage`.
- **Backend:** the memory system under test (e.g. `bm25`, `mem0`).
- **Event:** one thing written to memory, with an opaque `id`.
- **`fact_key`:** the pack's name for a fact, used only for scoring; adapters never see it.
- **Recall / context:** what the backend returns for a query; the context is what would be injected into a prompt.
- **Evidence:** the ids of earlier writes that answer a query; used for `evidence_recall_at_k`.
- **`expect_contains` / `expect_absent`:** strings the context must hold / must not hold.
- **Oracle:** baseline that sees each query's evidence; an upper bound.
- **Rubric:** the statement the judge checks ("The evidence indicates that ...").
- **Judge:** an LLM that answers pass/fail on a rubric and evidence.
- **Reader:** an LLM that answers a query from the recalled context, so answers can be compared with a gold answer.
- **Trace:** per-step record (latency, context, items, error) stored in `report.json`.
- **Manifest:** run metadata (pack hash, judge model and prompt version, seed, versions) that makes a run re-scorable.
- **Rescore:** recompute scores from a report's traces and the judge cache, with no adapter and no network.
- **Strict cache:** judge mode that only reads the cache; a miss is an error.
- **Leaderboard:** `results/README.md`, generated from committed results.
- **Indicative (†):** a number whose n is below the reporting minimum.
- **Kappa (κ):** chance-corrected agreement between two raters.
- **False pass / false fail:** judge passes what humans failed / fails what humans passed.
- **Held-out:** data the judge prompt was not tuned on. Validation labels should be kept apart from any prompt tuning, or the agreement numbers are optimistic. (Tuning tooling is described in docs/prompt-tuning.md, which this page has not read.)
- **Validated / unvalidated:** whether a committed agreement file passes the judge model on a metric. Default is unvalidated.
- **Length bucket:** evidence size class used to look for length bias in the judge.
