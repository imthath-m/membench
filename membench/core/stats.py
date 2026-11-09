"""Statistics for scores: CIs, minimum sample sizes, paired tests, Holm, repeats.

Pure Python and seeded, so `rescore` reproduces every interval exactly.

- Plain proportions (pass/fail outcomes) get a Wilson 95% interval.
- Means of non-binary per-query values (evidence recall@k, token cost) get a
  percentile bootstrap over queries (`BOOTSTRAP_RESAMPLES` resamples), seeded
  by a string key (run seed + scope + metric), so intervals don't depend on
  the order metrics are computed in.
- Below `MIN_HEADLINE_N` (overall) or `MIN_CATEGORY_N` (per category) a rate
  is labelled `indicative`.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from typing import Any

Z95 = 1.959963984540054
BOOTSTRAP_RESAMPLES = 10_000
MIN_HEADLINE_N = 385  # ±5pp at 95%, p = 0.5
MIN_CATEGORY_N = 100  # ±10pp at 95%
ALPHA = 0.05

Outcome = bool | float | int


def mean_of(outcomes: dict[int, Outcome]) -> float | None:
    """Mean of per-query outcomes (True = 1); None when there are none."""
    if not outcomes:
        return None
    return sum(outcomes.values()) / len(outcomes)


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float] | None:
    """Wilson score interval for k successes out of n; None for n = 0."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_means(
    values: Sequence[float], key: str, resamples: int = BOOTSTRAP_RESAMPLES
) -> list[float]:
    """Sorted means of `resamples` resamples (with replacement) of `values`."""
    rng = random.Random(key)
    n = len(values)
    values = list(values)
    return sorted(sum(rng.choices(values, k=n)) / n for _ in range(resamples))


def percentile_ci(means: list[float]) -> tuple[float, float]:
    """Central 95% of sorted bootstrap means."""
    r = len(means)
    return means[math.floor(0.025 * r)], means[math.ceil(0.975 * r) - 1]


def is_binary(values: Sequence[Outcome]) -> bool:
    return all(isinstance(v, bool) for v in values)


def ci(values: Sequence[Outcome], key: str) -> tuple[list[float] | None, str | None]:
    """(95% interval, method) for the mean of `values`; Wilson for pass/fail."""
    if not values:
        return None, None
    if is_binary(values):
        return list(wilson(sum(values), len(values))), "wilson"
    if min(values) == max(values):  # every resample has the same mean
        return [float(values[0])] * 2, "bootstrap"
    return list(percentile_ci(bootstrap_means(values, key))), "bootstrap"


def stat(values: Sequence[Outcome], key: str, min_n: int) -> dict[str, Any]:
    """`{n, ci, method, indicative}` for one score."""
    interval, method = ci(values, key)
    return {"n": len(values), "ci": interval, "method": method, "indicative": len(values) < min_n}


def mcnemar(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Exact (binomial) two-sided McNemar p-value for paired pass/fail outcomes."""
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1))
    return min(1.0, 2 * tail / 2**n)


def paired_diff(
    a: Sequence[Outcome], b: Sequence[Outcome], key: str
) -> dict[str, Any]:
    """Paired comparison of b - a over the same queries.

    Paired bootstrap of the mean difference (95% CI); p-value from McNemar for
    pass/fail outcomes, else the two-sided bootstrap p (twice the share of
    resampled differences on the far side of 0).
    """
    diffs = [float(y) - float(x) for x, y in zip(a, b)]
    n = len(diffs)
    mean_a, mean_b = sum(a) / n, sum(b) / n
    if min(diffs) == max(diffs):
        means = [diffs[0]]
    else:
        means = bootstrap_means(diffs, key)
    low, high = percentile_ci(means)
    if is_binary(a) and is_binary(b):
        p, test = mcnemar(a, b), "mcnemar"
    else:
        below = sum(m <= 0 for m in means) / len(means)
        above = sum(m >= 0 for m in means) / len(means)
        p, test = min(1.0, 2 * min(below, above)), "bootstrap"
    return {
        "n": n, "a": mean_a, "b": mean_b, "diff": mean_b - mean_a, "ci": [low, high],
        "ci_includes_zero": low <= 0 <= high, "p": p, "test": test,
    }


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def summarize_repeats(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean, run-to-run SD and every value of each score across repeats.

    `mean` / `sd` are None when any repeat scored the metric as None.
    """
    summary: dict[str, Any] = {}
    for name in scores[0]:
        values = [s.get(name) for s in scores]
        known = None not in values
        summary[name] = {
            "mean": statistics.fmean(values) if known else None,
            "sd": statistics.stdev(values) if known and len(values) > 1 else None,
            "values": values,
        }
    return {"n": len(scores), "scores": summary}
