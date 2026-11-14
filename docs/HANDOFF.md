# MemBench handoff

Single entry point for whoever picks this repo up next. Read this, then
[progress.md](progress.md).

## Current state

- **v0.2 (unreleased, `pyproject.toml` 0.2.0)** has, offline and tested:
  - Adapter contract v2 (`write` / `recall(at=)` / `forget` / `flush` / `reset` / `usage`), episodes, per-step errors, retries, timeouts.
  - Backends: `full-context`, `bm25`, `oracle`, `chroma` (offline); `mem0` (OSS), `mem0-platform`, `zep`, `letta`, `letta-agent`, `langmem`, `cognee` (need keys; covered offline by SDK fakes).
  - Metrics v2 (staleness, temporal, forget, abstention, evidence recall@k, QA accuracy via a reader), CIs, repeats, paired `compare`.
  - Run manifest, price table + cost, `rescore` (offline, strict cache), JSON Schema.
  - Datasets: LongMemEval, LoCoMo loaders; seeded synthetic packs (generator v2).
  - Judge validation tooling (sampling, labels, agreement, cross-judge); judge metrics show `unvalidated` until labels exist.
  - `publish` / `leaderboard` / `report --failures`; offline baseline results in `results/`; draft write-up `docs/comparison.md`.
  - CI: `ci.yml` (tests, lint, lock, chroma, results re-score), `live.yml` (manual), `smoke.yml` (nightly/weekly), `judge-validation.yml` (manual), `release.yml` (tag).
- **Nothing live has run.** Every live, human or release item is blocked on keys, people or the merge (runbook below).
- **Merged to `main`:** #1–#3 (v0.1: Phases 1–7). `main` is still 0.1.0 and has only `ci.yml`.
- **Open PR stack** (each based on the one before; merge bottom-up, in this order):

| PR | Branch | Content |
|---|---|---|
| #4 | `claude/awesome-bell-toi6w3` → `main` | Unblock live evals (`--backend-config`, reset, judge retry, lint) |
| #5 | `claude/awesome-bell-toi6w3-roadmap` | Roadmap (`docs/roadmap.md`) |
| #6 | `claude/gracious-edison-v8eyxs` | Phase 8: hardening |
| #7 | `claude/gracious-edison-v8eyxs-phase-9` | Phase 9: adapter contract v2 |
| #8 | `…-phase-10` | Phase 10: reproducibility + cost |
| #9 | `…-phase-11` | Phase 11: metrics v2 |
| #10 | `…-phase-12` | Phase 12: datasets + synthetic packs |
| #11 | `…-phase-13` | Phase 13: statistics |
| #12 | `…-phase-14` | Phase 14: backend coverage |
| #13 | `…-phase-15` | Phase 15: judge validation |
| #14 | `…-phase-16` | Phase 16: publishing + scheduled CI |
| #15 | `claude/gracious-edison-v8eyxs-followup` | Follow-up: whole-word `expect_absent` matching; this handoff |

(`…` = `claude/gracious-edison-v8eyxs`.) All open, none merged.

## Where details live

- [roadmap.md](roadmap.md): Phases 8–16 scope and done-when criteria.
- [progress.md](progress.md): "Overall status" checklist, per-phase logs, decisions and notes for next phases.
- Backends: [mem0.md](mem0.md), [mem0-platform.md](mem0-platform.md), [zep.md](zep.md), [letta.md](letta.md), [letta-agent.md](letta-agent.md), [langmem.md](langmem.md), [cognee.md](cognee.md), [bm25.md](bm25.md), [oracle.md](oracle.md); vendor configs + issue drafts: [vendor-review.md](vendor-review.md).
- Judge validation: [judge-validation.md](judge-validation.md), [labelling-guidelines.md](labelling-guidelines.md), `configs/judges.yaml`, `validation/`.
- Results: [results/README.md](../results/README.md) (generated); write-up draft [comparison.md](comparison.md).
- Pack generation from transcripts: [generate.md](generate.md). User-facing docs: [README](../README.md).

## Keys & setup

Names below are the ones the code and workflows read (`grep -rn "secrets\.\|environ" .github membench`).
Workflows: set them under GitHub repo **Settings → Secrets and variables → Actions → New repository secret**.
Local runs: `export NAME=...` in the shell. Never commit a key or paste one into chat.

| Secret / env var | Provider | Get it at | Unlocks | Used with (defaults) | Needed? |
|---|---|---|---|---|---|
| `MEMBENCH_JUDGE_API_KEY` | Anthropic (default); any OpenAI-compatible provider when `--judge-base-url` is set | console.anthropic.com → Settings → API keys | Judge (`--judge`; semantic recall, contradiction, QA accuracy); reader fallback; `generate` / `--paraphrase-model`; `live.yml`, `smoke.yml`, `judge-validation.yml` (panel judge `claude`) | Judge `claude-sonnet-5` (`live.yml` input `judge_model`); reader e.g. `claude-haiku-4-5` | **Required** for every live item |
| `OPENAI_API_KEY` | OpenAI | platform.openai.com → API keys | Mem0 OSS LLM + embedder (`live.yml` job `mem0`, `smoke.yml`); LangMem; Cognee (copied to `LLM_API_KEY` by the workflows); panel judge `gpt` in `judge-validation.yml` | Mem0's built-in OpenAI defaults (exact ids land in `manifest.backend.models`); LangMem `openai:gpt-4o-mini` + `openai:text-embedding-3-small`; panel `gpt-4.1` | **Required** for mem0 / langmem / cognee / judge validation |
| `MEM0_API_KEY` | Mem0 Platform | app.mem0.ai → API keys | `mem0-platform` backend (`live.yml` `backends` job, `smoke.yml`) | Mem0's hosted models | Optional (per backend) |
| `ZEP_API_KEY` | Zep Cloud | app.getzep.com → project → API keys | `zep` backend | Zep's hosted graph | Optional (per backend) |
| `LETTA_API_KEY` | Letta Cloud | app.letta.com → API keys | `letta`, `letta-agent` backends | `openai/gpt-4o-mini` + `openai/text-embedding-3-small` handles (Letta Cloud) | Optional (per backend) |
| `TOGETHER_API_KEY` | Together AI | api.together.ai → Settings → API keys | Panel judge `llama` in `judge-validation.yml` / `membench judge-cross` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | Required for cross-judge only |
| `LLM_API_KEY` | OpenAI (Cognee's own variable) | same as `OPENAI_API_KEY` | `cognee` backend locally (workflows set it from the `OPENAI_API_KEY` secret; **no separate repo secret**) | Cognee's OpenAI defaults | Local cognee runs only |
| `MEMBENCH_READER_API_KEY` | Reader provider | as for the judge | Reader key when it differs from the judge's (`--reader-model`); falls back to `MEMBENCH_JUDGE_API_KEY` | — | Optional, local only (workflows use the judge key) |
| `LETTA_BASE_URL` | self-hosted Letta | your server | Read by the Letta SDK to target a self-hosted server instead of Letta Cloud (or set `base_url` in the config) | — | Optional, local only |
| `MEMBENCH_LIVE=1` | — (not a secret) | — | Enables `tests/test_live_backends.py` | — | Local live tests |
| `MEMBENCH_CACHE_DIR`, `TIKTOKEN_CACHE_DIR`, `MEMBENCH_VALIDATION_DIR` | — (not secrets) | — | Dataset cache (`~/.cache/membench`), tiktoken vocab cache, validation dir override | — | Optional |

- **Minimum set for the first live Mem0 + judge run:** `OPENAI_API_KEY` + `MEMBENCH_JUDGE_API_KEY` (Anthropic key).
- **Workflows only dispatch from `main`:** `live.yml`, `smoke.yml`, `judge-validation.yml`, `release.yml` exist only on the stack branches; GitHub needs a `workflow_dispatch` / `schedule` workflow on the default branch. Merge the stack first, or run the same commands locally.
- **GitHub repo settings:**
  - Settings → Actions → General: Actions enabled. `smoke.yml`'s `report-failure` job requests `issues: write` itself; this works with the default "Read repository contents" workflow permission unless an org policy caps `GITHUB_TOKEN` permissions (then allow it).
  - Scheduled workflows: GitHub pauses schedules after 60 days without repo activity; re-enable under Actions if so.
- **PyPI trusted publishing (one-time):**
  1. pypi.org → Account → Publishing → "Add a new pending publisher" (the `membench` project does not exist yet): owner `imthath-m`, repository `membench`, workflow `release.yml`, environment `pypi`.
  2. GitHub repo Settings → Environments → New environment `pypi`; add required reviewers (recommended); restrict deployment to tags `v*`.
  3. No PyPI token secret is needed (OIDC, `id-token: write`).

## Open items runbook

Do these in order. Each closes a checkbox in [progress.md](progress.md) "Overall status" (and the phase section named).

1. **Merge the stack** #4 → #5 → … → #15 into `main` (retarget each next PR to `main` after its base merges, or merge each into its base top-down then the bottom into `main`). Expect CI green on each. Closes "Merge the stack". Unblocks every `gh workflow run` below and the nightly schedule.
2. **Add secrets** (table above; at least `OPENAI_API_KEY`, `MEMBENCH_JUDGE_API_KEY`). Closes "Repository secrets" once all are set.
3. **P8/P9 first live run:** `gh workflow run live.yml -f backend=mem0 -f pack=customer-support`
   - Local: `uv sync --frozen --extra mem0 --extra tokens && membench run --backend mem0 --pack customer-support --judge --out report.json && membench rescore --in report.json --check`.
   - Expect: job green, `live-report.md` in the step summary, `valid: true`, `error_rate` < 0.1, non-null `semantic_recall`. Closes Phase 8 Done-when 2 and the Phase 9 live check.
4. **P10 cost check:** first re-verify the OpenAI rows of `membench/prices.yaml` against openai.com/api/pricing (bump `as_of`). Then compare that run's `cost.backend` / `cost.judge` tokens + USD with the OpenAI and Anthropic usage dashboards for the run window. Expect within 5%. Closes Phase 10 Blocked.
5. **P11 reader metrics:** `gh workflow run live.yml -f backend=mem0 -f pack=customer-support -f reader_model=claude-haiku-4-5`. Expect non-null `qa_accuracy`, `abstention`. Closes Phase 11 Blocked.
6. **P13 LLM repeats:** `gh workflow run live.yml -f backend=mem0 -f pack=customer-support -f repeats=3`; check `repeat_summary` SDs and that each repeat's `stats.scores[*].ci` overlap; `membench rescore --in <report> --check` → "identical". Closes Phase 13 Blocked.
7. **P14 hosted backends:** `gh workflow run live.yml -f backend=all -f pack=mini` (all keys). Expect every matrix job green (contract test + run + rescore). Closes Phase 14 Done-when.
8. **P14 vendor review (user):** open the issues drafted in [vendor-review.md](vendor-review.md) on the vendors' repos; put URLs in its table; when vendors reply, add `configs/vendor-tuned/<backend>.yaml` and run with `-f config=vendor-tuned`. Closes "P14 vendor review".
9. **P12 LongMemEval ±3pp** (OpenAI key, local): `MEMBENCH_JUDGE_API_KEY=<openai key> membench run --backend full-context --pack longmemeval-oracle --reader-model gpt-4o-2024-08-06 --reader-base-url https://api.openai.com/v1 --judge-model gpt-4o-2024-08-06 --judge-base-url https://api.openai.com/v1 --out lme-oracle.json`. Expect `scores.qa_accuracy` within 3pp of the paper's GPT-4o oracle row (gap sources: Phase 12 notes). Closes Phase 12 "oracle ±3pp".
10. **P12 LoCoMo human audit (person):** review gold answers; add findings to `membench/datasets/locomo_exclusions.yaml` in the existing format. Closes "LoCoMo human audit".
11. **P15 human labels (2 people + adjudicator):** `jq -c 'del(.verdict)' validation/sample.jsonl > items.jsonl`; annotators follow [labelling-guidelines.md](labelling-guidelines.md); `membench judge-labels --annotator a=a.jsonl --annotator b=b.jsonl --adjudication adj.jsonl` → `validation/labels.jsonl`. Closes "P15 human labels".
12. **P15 qa_accuracy items:** reader runs per backend on synthetic packs (`membench run --backend {full-context,bm25,chroma} --pack <synthetic> --reader-model claude-haiku-4-5 --judge --out r.json`), re-run `membench judge-sample --in ...` over all reports, label the new items. Closes "P15 qa_accuracy items".
13. **P15 cross-judge + agreement:** `gh workflow run judge-validation.yml` (`MEMBENCH_JUDGE_API_KEY`, `OPENAI_API_KEY`, `TOGETHER_API_KEY`). Download the `judge-validation` artifact, commit `validation/` (verdicts, `judge_cache.json`, `agreement-v1.json` / `.md`) in a PR; update `tests/test_judge_validation.py::test_no_committed_validation_result_means_unvalidated`; cite κ in the README. Closes "P15 cross-judge" and Phase 15 Done-when.
14. **P16 live result rows:** `gh workflow run live.yml -f backend=mem0 -f pack=tool-call-memory`, then `-f backend=all -f pack=tool-call-memory`; repeat with `pack=customer-support-xl`. Commit them:
    - `gh run download <run-id>`; copy each artifact's `results/<date>/<pack>/<backend>/` into the repo's `results/` (the job already ran `membench publish`).
    - `membench rescore --in <dir>/report.json --judge-cache <dir>/judge_cache.json --check` per new result.
    - `membench leaderboard results/ --out results/README.md`.
    - Add the rows to [comparison.md](comparison.md) (every number linked to its `report.json`), drop "draft", open a PR. Closes "P16 live result rows" + Phase 16 "write-up published".
15. **P16 nightly smoke 7 days:** after steps 1–2, `smoke.yml` runs by itself at 03:17 UTC; watch for 7 consecutive green nights (failures open an issue "Nightly smoke run failed"). Don't dispatch it to count. Closes Phase 16 Done-when 1.
16. **P16 PyPI v0.2.0 (user):** PyPI setup above, then from `main`: `git tag v0.2.0 && git push origin v0.2.0`; approve the `pypi` environment. Expect `pip install membench==0.2.0` to work. Then update the README "Install" section. Closes "P16 PyPI".
17. **Judge prompt tuning (needs step 11's labels + `MEMBENCH_JUDGE_API_KEY`):** annotators label with `membench judge-label --annotator NAME` (each separately), merge with `membench judge-labels`; then `membench judge-tune --prompt-version 1 --prompt-version 2` → `validation/tuning-v2.json` / `.md` (up to 800 judge calls first run; `--strict` replays). Decide on candidate v2 per [prompt-tuning.md](prompt-tuning.md) (held-out once). Closes "Judge labelling tool + prompt tuning" open items.
18. **Optional:** BEAM / MemoryAgentBench loaders (Phase 12 deferred).

## Known limitations / caveats

- **Prices:** `membench/prices.yaml` OpenAI rows were filled in without checking the live pricing page; Claude rows checked 2026-10-06. Models missing from the table (e.g. `gpt-4o-2024-08-06`, `gpt-4.1`, Together models) price as `usd: null`.
- **Version breaks:** report schema v2 (v1 reports don't validate); synthetic generator v2 (pre-Phase-15 synthetic reports can't be re-scored); a judge prompt change must bump `JUDGE_PROMPT_VERSION` (pinned by a test hash). There is no scoring version in the manifest: a metric change shows up as `rescore --check` diffs, and committed results must be re-scored + re-published (done once in #15).
- **String matching:** `expect_absent` (staleness, forget compliance) matches whole words; `expect_contains` (recall precision, update correctness, temporal accuracy) matches plain substrings.
- **Metrics meaningless for some backends:** `letta-agent` items carry no event ids (evidence recall@k reads 0); same for `cognee` with a non-`CHUNKS` search type; `oracle` is an upper bound, not a competitor. Chroma/Mem0 contexts carry no dates, so temporal accuracy is low by construction.
- **Cost gaps:** `mem0-platform`, `zep`, `letta`, `cognee` report no `usage()` (cost.backend null); LangMem counts chat tokens only.
- **Chroma is slow** (~5 writes/s on 4 CPUs): LongMemEval_S on chroma ≈ 14 h; use `--limit`.
- **Large reports:** contexts are stored on traces; LongMemEval_S full-context ≈ 600 MB. LongMemEval / LoCoMo results are not committed (size; LoCoMo licence allows ids only).
- **Judge variance is not in repeats:** verdicts are cached by prompt, so repeats capture backend + reader variance only.
- **Small n:** committed synthetic results are below the 385-query headline minimum (marked indicative).
- **Judge metrics are `unvalidated`** until `validation/agreement-v1.json` exists.
- **Cognee** `reset()` prunes the whole local Cognee install; it picks up AWS env credentials and then needs `s3fs`.
- Step timeout (120 s) is shorter than vendor `flush_timeout_s` (600 s): raise `--step-timeout` for slow ingestion.
- Synthetic packs are template-generated: easier and more regular than real conversations.

## Instructions for the next agent

- **Start:** read this file, then [progress.md](progress.md) ("Overall status" + the latest section). Resume from the first unchecked item.
- **Stacked PRs:** new work goes on a new branch based on the top of the stack (or `main` once merged); one PR per phase / follow-up, base = previous branch; PR body starts "Stacked on #N." Push only your own branch.
- **Progress logging:** add a section to progress.md with a checklist first, commit + push, tick items as you go; record decisions and notes for next phases; record the PR number in the phase table. Keep "Overall status" and this file consistent (keys table lives here only).
- **Checks before every push:**
  - `pip install -e ".[dev,chroma,tokens]"` (or `uv sync --frozen --extra dev --extra chroma --extra tokens`)
  - `ruff check .` and `pytest`
  - `uv lock --check` (run `uv lock` after any `pyproject.toml` dependency change)
  - Results: `for r in results/*/*/*/report.json; do membench rescore --in "$r" --judge-cache "$(dirname "$r")/judge_cache.json" --check; done` then `membench leaderboard results/ --out results/README.md && git diff --exit-code results/README.md`
  - `actionlint` when workflows change.
- **Report format changes:** edit `membench/report.schema.json`, bump `REPORT_SCHEMA_VERSION` if breaking, regenerate the fixture with `python tests/fixtures/make_report.py`.
- **Scoring changes:** re-score committed results (`membench rescore --in <report> --out new.json`, then `membench publish --in new.json --judge-cache <dir>/judge_cache.json`), regenerate `results/README.md`, update `docs/comparison.md` numbers.
- **Never** commit dataset files or converted dataset YAML (LongMemEval / LoCoMo licences); never fake results, labels, costs or vendor replies; never publish to PyPI or open vendor issues without the user.
- Adapter files stay under 100 lines (test-enforced); adapters never see scoring ground truth.
