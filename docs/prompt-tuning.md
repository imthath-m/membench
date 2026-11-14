# Judge prompt tuning

How to test whether a new judge prompt agrees with people better than the
current one, without fooling yourself. Validation of the default judge is
[judge-validation.md](judge-validation.md); annotator rules are
[labelling-guidelines.md](labelling-guidelines.md).

## 1. Purpose and status

Question the harness answers:

- For the same human-labelled items and the same judge model, does prompt
  version B agree with the human gold better than version A, per metric and
  per context length?

> **Status: tooling only. Nothing has been run.**
>
> | Exists (code + offline tests) | NOT RUN |
> |---|---|
> | `membench judge-label` (browser labelling tool) | No human labels collected; `validation/labels.jsonl` does not exist |
> | `membench judge-tune` (dev / held-out comparison) | v1 and v2 never scored against labels |
> | Prompt-version registry (`PROMPT_BUILDERS` in `membench/core/scoring/judge.py`) | No `validation/tuning-v2.json` exists |
> | Candidate prompt v2 (untested hypothesis, not the default) | No API calls made; no agreement numbers for any version |
> | Tests: `tests/test_judge_tune.py`, `tests/test_judge_label.py`, `tests/test_judge_prompt_v2.py` (synthetic data only) | |

## 2. Method

### Gold labels

- Humans are the gold, not another model. No LLM-generated labels.
- At least 2 annotators label every item alone, blind to judge verdicts.
- A third person adjudicates disagreements (`adjudication.jsonl`).
- The gold of an item is `gold()` in `membench/core/validation/labels.py`:
  the adjudicated label if set, else a clear majority. Items with no settled
  gold (e.g. an unadjudicated tie) are excluded and counted as `n_unresolved`.

### Dev / held-out split

- `split_items` in `membench/core/validation/tune.py`.
- Fixed seed (`--seed`, default `0`), stratified by metric.
- Depends only on the seed, the fraction and each metric's set of resolved
  item ids. It does not depend on which versions are compared, so every
  candidate is measured on the same split.
- Default 50 / 50 (`--held-out 0.5`).

### Iterate on dev only

1. Run `judge-tune` with the baseline and a candidate.
2. Read the dev disagreements (`disagreements` in the JSON: dev item ids where
   the version's verdict differs from the gold). Open those items in the
   sample.
3. Form one hypothesis about why the judge is wrong (e.g. "passes long dumps
   that only mention the topic").
4. Write a **new** version with one change for that hypothesis.
5. Repeat on dev.

### Held-out: once per candidate

- Look at the held-out numbers once, for the candidate you would adopt.
- If you change the prompt after looking at held-out results, held-out has
  become dev; its numbers are now optimistic.
- Held-out item ids are never listed in the output (only dev disagreements),
  so you cannot read held-out failures by accident and tune on them.

### Decision rule (proposal, not enforced by code)

Adopt a candidate only if, on held-out:

- κ does not drop on any metric, and
- the false-pass rate does not rise on any metric, and
- the pass criteria in `membench/core/validation/agreement.py` (see
  [judge-validation.md](judge-validation.md#pass-criteria)) still hold.

### Why runs are reproducible and cheap

- Every verdict is cached in the judge cache keyed by (judge model, prompt
  version, prompt hash).
- Re-running a comparison replays cached verdicts: no API calls, same numbers.
  `--strict` makes that explicit (a missing verdict is an error).
- Adding a new candidate only calls the judge for that version's items.

### Prompt-version policy

- New prompt text = new version number (a new `PROMPT_BUILDERS` entry).
- Never edit a version once any result for it exists. A test pins the text
  of v1 (`tests/test_judge_validation.py`) and v2
  (`tests/test_judge_prompt_v2.py`).
- Promoting a candidate:
  1. Bump `JUDGE_PROMPT_VERSION` in `membench/core/scoring/judge.py`.
  2. Update the pinned hash in `tests/test_judge_validation.py`.
  3. Re-run `judge-cross` and `judge-agreement`; commit the new
     `validation/agreement-v<N>.json`.
  4. Re-score results made with the new version.
- Full policy: [judge-validation.md](judge-validation.md#prompt-versioning-policy).

## 3. Commands, end to end

### Label (each annotator, separately)

Each annotator runs this on their own machine. The page hides the judge
verdict, model, backend and length; each label is saved as soon as it is
given.

```bash
membench judge-label --annotator ann1
# writes validation/annotations/ann1.jsonl; resumes where you left off
```

```bash
membench judge-label --annotator ann2
```

Options (from `membench judge-label --help`): `--sample` (default
`validation/sample.jsonl`), `--out`, `--metric` (repeat), `--seed`, `--host`,
`--port` (default `8765`, `0` = any free port), `--no-open`.

### Merge (+ adjudication)

```bash
membench judge-labels \
  --annotator ann1=validation/annotations/ann1.jsonl \
  --annotator ann2=validation/annotations/ann2.jsonl
# prints how many items disagree; a third person writes adjudication.jsonl
# ({"item_id": "...", "label": true|false, "note": "..."} per line), then:
membench judge-labels \
  --annotator ann1=validation/annotations/ann1.jsonl \
  --annotator ann2=validation/annotations/ann2.jsonl \
  --adjudication adjudication.jsonl \
  --out validation/labels.jsonl
```

### Tune (needs a key on the first run)

The key is read from `MEMBENCH_JUDGE_API_KEY`.

```bash
export MEMBENCH_JUDGE_API_KEY=...   # never commit it
membench judge-tune --prompt-version 1 --prompt-version 2
# writes validation/tuning-v2.json and validation/tuning-v2.md
```

- First = baseline, last = candidate.
- 400 labelled items × 2 versions = up to 800 judge calls on the first run
  (fewer if verdicts are already cached). No cost estimate is given here; the
  `usage` field records the tokens actually used.
- Other options: `--labels`, `--sample`, `--seed`, `--held-out`,
  `--judge-model` (default `claude-sonnet-5`), `--judge-base-url`,
  `--judge-cache` (default `validation/judge_cache.json`), `--out`.
- Without a key it prints `No $MEMBENCH_JUDGE_API_KEY: using cached verdicts
  only (as --strict).`

### Re-run offline

```bash
membench judge-tune --prompt-version 1 --prompt-version 2 --strict
```

- Uses only cached verdicts. A missing verdict exits 1 and names the version
  and the first missing item ids.

## 4. How to read the output

### `tuning-v<N>.json` fields

| Field | Meaning |
|---|---|
| `schema` | Report format version |
| `seed`, `held_out_fraction` | Split settings |
| `judge_model`, `base_url` | Judge used for every version |
| `versions`, `baseline`, `candidate` | Versions in order; first and last |
| `n_items`, `n_unresolved` | Items with a settled gold; labelled items excluded for lack of one |
| `split_sizes` | `{metric: {dev, held_out}}` item counts |
| `prompt_varies` | `{metric: bool}`; `false` = the metric uses a fixed official template (LongMemEval), so the version cannot change its prompt |
| `results` | `{version: {dev, held_out, all: {metric or "overall": stats}}}` |
| `deltas` | `{split: {metric: {kappa, accuracy, false_pass_rate, false_fail_rate}}}`, candidate − baseline |
| `disagreements` | `{version: [dev item ids where the verdict ≠ gold]}` |
| `prompt_sha256` | Hash of each version's prompt template |
| `usage` | Per version: `calls`, `cache_hits`, `input_tokens`, `output_tokens` of this run |
| `warnings`, `notes` | See below |

Each stats block: `n`, `kappa`, `accuracy`, `false_pass`, `false_pass_rate`,
`false_fail`, `false_fail_rate`, `errors`, and `by_length` (same stats per
length bucket; empty buckets omitted).

Shape only (no real numbers exist):

```json
{"versions": ["1", "2"], "baseline": "1", "candidate": "2",
 "results": {"1": {"dev": {"semantic_recall": {"n": …, "kappa": …, "accuracy": …,
   "false_pass_rate": …, "false_fail_rate": …, "errors": …, "by_length": {"<128": {…}}}}}},
 "deltas": {"held_out": {"overall": {"kappa": …, "accuracy": …, "false_pass_rate": …,
   "false_fail_rate": …}}}}
```

### Markdown (`tuning-v<N>.md`)

- One table per split (`dev`, `held_out`, `all`): version × metric.
- "By context length": the same stats per length bucket.
- "Held-out deltas": candidate − baseline per metric.
- "Warnings" and "How to read".

### Kappa vs accuracy

- Accuracy = share of items where judge and gold agree.
- κ corrects for chance. If most items are `false`, a judge that always says
  `fail` has high accuracy and κ near 0. Prefer κ.

### False pass vs false fail

- False pass: judge said pass, humans said fail. Rate = false passes / items
  humans failed.
- False fail: the reverse. Rate = false fails / items humans passed.
- For `semantic_recall` and `qa_accuracy`, false passes are worse: they give
  a backend credit it did not earn, so scores look better than they are.
- For `contradiction_rate`, "pass" means "contradiction present": a false pass
  invents a contradiction (unfairly penalises a backend); a false fail hides
  a real one. Read both rates.

### By length and length bias

- Buckets: whitespace tokens of the evidence, `<128`, `128-511`, `512-2047`,
  `2048+`.
- Length bias: a judge passing long dumps more readily than short contexts.
  Look for a false-pass rate that rises with bucket size.

### Deltas

- Candidate − baseline. Positive Δκ / ΔAccuracy is better; negative ΔFalse
  pass / ΔFalse fail is better.
- `null` (`n/a` in Markdown) when either side is undefined.

### Warnings

- Fewer than 50 items in a metric's split: differences are likely noise.
- Empty length buckets: no evidence about those lengths.
- `prompt_varies: false`: identical numbers across versions are expected
  (fixed official template).
- Judge errors per version: excluded from agreement, never counted as pass
  or fail.
- A standing reminder to use held-out once.

### `None` / `n/a` kappa

- κ is undefined when there are no items, or when a rater is constant (e.g.
  every gold label in a bucket is `true`, or the judge always says `pass`).
  It is shown as `null` / `n/a`, not 0.

## 5. Prompt v2 (candidate)

> **Untested hypothesis.** Never run against any model. No agreement numbers
> exist for it. Not the default (`JUDGE_PROMPT_VERSION` is `"1"`).

Same output contract as v1 (`{"verdict": "pass"|"fail", "reason": ...}`,
parsed by the unchanged `parse_verdict`) and the same STATEMENT / EVIDENCE
framing, so it works for all three rubric families. It only asks whether the
STATEMENT is true of the EVIDENCE. Code: `_build_prompt_v2` in
`membench/core/scoring/judge.py`.

| Failure mode | Source | Instruction added |
|---|---|---|
| Length bias | [judge-validation.md](judge-validation.md) (by-length breakdown); [labelling-guidelines.md](labelling-guidelines.md) "Do not reward length" | The amount of EVIDENCE is not support; long EVIDENCE passes only if it contains the specific support; short EVIDENCE that states it passes |
| Partial matches | labelling-guidelines: "Clearly" (guessing = false), "Wording does not have to match", `qa_accuracy` hedge rule | Every part of the STATEMENT must be supported; guess = fail; wording / format may differ; topical overlap, a near-miss value or a hedge between a right and a wrong answer = fail |
| Abstention | labelling-guidelines `qa_accuracy`: "I don't know" is false unless the reference is unanswerable | Declining / not knowing does not support a STATEMENT needing a specific answer; it does support a STATEMENT that the answer is unknown or unavailable |
| Superseded values | labelling-guidelines `semantic_recall` and `contradiction_rate` | A value clearly marked as replaced (later date, "changed to", "now", "instead") is an old value, not a conflict; unmarked conflicting values leave it unclear which holds |
| Unsupported passes | (hypothesis) evidence-first answering | Find the shortest deciding span first; `reason` (the quoted span, or what is missing) comes before `verdict` in the JSON |

Notes:

- Superseded values: included because the wording matches both rubrics. A
  marked update is not a conflict (`contradiction_rate`: `false`;
  `semantic_recall`: the new value holds). Unmarked conflicting values leave
  the current value unclear (`semantic_recall`: `false`; `contradiction_rate`:
  `true`). Whether a model reads it that way is untested.
- No few-shot examples: examples copied from `validation/sample.jsonl` would
  leak labelled items into the prompt and inflate agreement.
- About 200 words of instructions.

## 6. Limits and threats to validity

- **Small n.** 400 items split 50 / 50 by metric leaves about 100 per metric
  per split, fewer per length bucket. Small deltas are noise.
- **Missing coverage.** The committed sample has no items of 512 tokens or
  more and no `qa_accuracy` items. Nothing can be said about long contexts
  or QA until those are sampled and labelled.
- **Single judge model.** A prompt tuned for one model may not transfer.
  Re-check with `judge-cross` after promotion.
- **Judge variance unmeasured.** Verdicts are temperature 0 and cached, so
  one verdict per item per version; run-to-run variance of the judge is not
  measured.
- **Held-out peeking.** Every look at held-out numbers followed by a prompt
  change overfits held-out. Keep a log of which candidates were checked on
  held-out.
- **Synthetic packs only.** Annotators see items from synthetic,
  template-generated packs; real conversations may fail in other ways.
