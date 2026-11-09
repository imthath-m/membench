# Judge validation

The judge metrics (`semantic_recall`, `contradiction_rate`, `qa_accuracy`) are
only as good as the judge. This page is the workflow that measures that, the
file formats it commits, and the prompt-versioning policy.

**Status:** nothing is validated yet. No human labels exist, so every judge
metric is shown as `unvalidated` in reports.

## Workflow

1. **Runs.** Run each backend on the packs to validate (judge on, so each item
   gets a verdict to stratify on; `--no-judge` works too, the verdict stratum is
   then `unjudged`).
2. **Sample.** `membench judge-sample --in <report.json> [--in ...] --out
   validation/sample.jsonl` draws up to 200 distinct items per judge metric
   (`--n`), round-robin over the strata (backend × verdict × context-length
   bucket), so small strata (e.g. long contexts) are fully covered.
3. **Label.** Two annotators label every item on their own, following
   [labelling-guidelines.md](labelling-guidelines.md). They must not see the
   judge's verdict, or each other's labels.
4. **Merge + adjudicate.** `membench judge-labels --annotator a=a.jsonl
   --annotator b=b.jsonl --out validation/labels.jsonl` merges the two and
   prints how many items disagree. A third person adjudicates them
   (`adjudication.jsonl`, same row format); re-run with `--adjudication`.
5. **Cross-judge.** `membench judge-cross --sample validation/sample.jsonl
   --judges configs/judges.yaml` runs every judge of the panel (Claude, GPT, an
   open-weight model through an OpenAI-compatible URL) on the same items →
   `validation/verdicts-v<N>.jsonl`. Calls are cached in
   `validation/judge_cache.json`; `--strict-cache` replays them offline.
6. **Agreement.** `membench judge-agreement --labels validation/labels.jsonl
   --verdicts validation/verdicts-v<N>.jsonl` → `validation/agreement-v<N>.json`
   (+ `.md`): human–human κ; for each judge, κ / accuracy / false-pass /
   false-fail against the majority-human label, overall and by context-length
   bucket; pairwise judge κ and the items each pair disagrees on.
7. **Commit** `validation/` (sample, labels, verdicts, judge cache, agreement).
   Reports then mark a judge metric `validated` only for a judge model that
   passed on it.

`.github/workflows/judge-validation.yml` (manual dispatch, secret-gated) runs
steps 5–6 and uploads the outputs.

## Pass criteria

Per metric and judge: human–human κ ≥ 0.7; judge–human κ ≥ 0.6 and no more
than 0.1 below human–human; at least 200 labelled items; every annotator
disagreement adjudicated. Anything else is `unvalidated`; its prompt is
reworked (see the policy below). The thresholds live in
`membench/core/validation/agreement.py`.

The by-length breakdown is there to spot a length bias (e.g. a judge that
passes long full-context dumps more readily than short top-k contexts); it
does not gate the status on its own, but a bucket that is clearly worse is a
reason to rework the prompt.

## Files

`validation/sample.jsonl` — one item per line:

| Field | Meaning |
|---|---|
| `id` | sha256 of (metric, rubric, evidence, prompt): stable across judge prompt versions |
| `metric` | `semantic_recall`, `contradiction_rate`, `qa_accuracy`, or `qa_accuracy:longmemeval` (the official LongMemEval prompt) |
| `backend`, `pack`, `step_index`, `category` | where the item came from |
| `rubric` | the statement to check (for QA: question + reference answer) |
| `evidence` | what the judge reads: the recalled context (QA: the reader's answer) |
| `prompt` | only for `qa_accuracy:longmemeval`: the exact official prompt |
| `tokens`, `length_bucket` | whitespace tokens of `evidence`; `<128`, `128-511`, `512-2047`, `2048+` |
| `verdict` | the sampling judge's cached verdict (`null` = unjudged); for stratification only |
| `judge_model`, `prompt_version` | judge and `JUDGE_PROMPT_VERSION` at sampling time |

Annotator files (`a.jsonl`, `adjudication.jsonl`): `{"item_id": "...", "label":
true|false, "note": "optional"}` per line.

`validation/labels.jsonl` — one labelled item per line (no text; the text is in
the sample):

```json
{"item_id": "3f2a…", "metric": "semantic_recall", "backend": "chroma",
 "length_bucket": "<128", "prompt_version": "1", "judge_model": "claude-sonnet-5",
 "labels": {"ann1": true, "ann2": false}, "adjudicated": true, "notes": "ann2: …"}
```

A label answers the judge's own question: `true` = the evidence clearly
supports the statement (for `contradiction_rate`: the contradiction is
present), the same polarity as the judge's `passed`. Each metric has the same
annotators on every line. The gold label is `adjudicated` when set, else the
majority. `membench.core.validation.labels.load_labels` validates the file.

`validation/verdicts-v<N>.jsonl` — one row per (judge, item): `item_id`,
`metric`, `judge` (panel name), `model`, `base_url`, `prompt_version`, `passed`
(`null` on a judge error, with `error`).

`validation/agreement-v<N>.json` — the agreement report for judge prompt
version N. Reports look a judge metric up here by the run's judge model and
prompt version (`membench/core/validation/status.py`; `MEMBENCH_VALIDATION_DIR`
overrides the directory).

Only synthetic packs go into a committed sample: dataset text (LoCoMo is
CC BY-NC) is never committed.

## Prompt-versioning policy

- Any edit to a judge prompt (`build_prompt`, `parse_verdict`, `qa_rubric`, or
  the LongMemEval judge templates) bumps `JUDGE_PROMPT_VERSION` in
  `membench/core/scoring/judge.py`. `tests/test_judge_validation.py` pins a hash
  of the prompts per version and fails until the bump and the new hash are in.
- The bump changes the judge cache key and the run manifest, so verdicts of
  different prompt versions never mix, and `compare` refuses to pair them.
- After a bump, re-run steps 5–6 on the same labelled items (item ids don't
  change with the prompt, so the human labels carry over) and commit the new
  `validation/agreement-v<N>.json` next to the old ones. Until it is committed,
  every judge metric of the new version is `unvalidated`.
- The default judge (`DEFAULT_JUDGE_MODEL`) and `JUDGE_PROMPT_VERSION` must be a
  pair that the committed agreement file validates; the README cites its
  numbers.
