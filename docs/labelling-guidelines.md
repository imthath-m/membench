# Labelling guidelines (judge validation)

You label items from `validation/sample.jsonl`. Each item is a **statement**
(`rubric`) and some **evidence** (`evidence`). Answer one question per item:

> Does the EVIDENCE clearly support the STATEMENT?

Write `true` or `false` (and a short `note` when it was a hard call) to your
own file, one line per item: `{"item_id": "...", "label": true, "note": "..."}`.

## Rules for every item

- Judge only the evidence shown. Ignore what you know or guess about the user,
  the pack, or what the "right" answer probably is.
- **Clearly** means a careful reader would agree without assumptions. If the
  evidence only hints at the statement, or you need to guess, label `false`.
- Wording does not have to match. Paraphrases, synonyms and obvious formatting
  differences ("Pro tier" vs "Pro plan", "3 hours" vs "three hours") count.
- Do not reward length. A long log that contains the fact somewhere supports
  it as much as a one-line answer; a long log that buries a contradiction
  contains it as much as a short one.
- Work alone. Do not discuss items with the other annotator, and do not look
  at any judge's verdict. Label from a copy without it:
  `jq -c 'del(.verdict)' validation/sample.jsonl > my-items.jsonl`.
- Take items in the given order; don't skip. If an item is broken (empty
  evidence, garbled text), label `false` and note `broken`.

## Per metric

**`semantic_recall`** — the statement says what a good recall should convey
(e.g. "The evidence indicates that the answer to 'Which plan…?' is Pro.").
`true` if the evidence lets a reader answer that way. When the evidence also
holds an older value, it is still `true` if the evidence makes clear which
value is current (e.g. a later timestamp, "moved to", "now"); `false` if a
reader can't tell which one holds.

**`contradiction_rate`** — the statement describes a contradiction (e.g. "The
evidence names more than one current plan … without indicating which one
superseded the others."). `true` if the contradiction is present: two or more
conflicting values both presented as current, with nothing (dates, "changed
to", "instead") saying which wins. An update that is clearly marked as an
update is **not** a contradiction: `false`.

**`qa_accuracy`** — the statement compares an answer with a reference answer
(question + reference in the statement; the reader's answer is the evidence).
`true` if the answer gives the same answer as the reference: same entity,
value, date or number; extra correct detail is fine; a hedge that names the
right answer plus a wrong one is `false`; "I don't know" is `false` unless the
reference says the question cannot be answered.

**`qa_accuracy:longmemeval`** — same as `qa_accuracy`; the item also carries
the official LongMemEval `prompt`. Follow its question-type rule when it is
stricter (e.g. off-by-one counts for temporal questions are accepted there,
and for abstention items the answer must say the question is unanswerable).

## Adjudication

After both annotators finish, `membench judge-labels` lists the items you
disagree on. A third person (or both of you together, with the guidelines,
not the judge's verdict) decides each one and records it in
`adjudication.jsonl`. When a disagreement shows that a rule above is unclear,
fix this page and note the change in the PR.
