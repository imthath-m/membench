"""Judge validation (Phase 15): sample judge items, load human labels, measure
agreement (human–human, judge–human, judge–judge) and look up which judge
metrics a committed validation result has validated.

See `docs/judge-validation.md` for the workflow and the prompt-versioning policy.
"""

JUDGE_METRICS = ("semantic_recall", "contradiction_rate", "qa_accuracy")
