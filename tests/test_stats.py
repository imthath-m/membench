import pytest

from membench.core.runner import StepTrace
from membench.core.scoring import score_detailed
from membench.core.stats import (
    MIN_CATEGORY_N,
    MIN_HEADLINE_N,
    bootstrap_means,
    ci,
    holm,
    mcnemar,
    paired_diff,
    stat,
    summarize_repeats,
    wilson,
)
from membench.packs.schema import Pack, QueryStep


def test_wilson_matches_known_values():
    low, high = wilson(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-4)
    assert high == pytest.approx(0.7634, abs=1e-4)
    low, high = wilson(0, 10)
    assert low == 0.0
    assert high == pytest.approx(0.2775, abs=1e-4)
    assert wilson(0, 0) is None


def test_bootstrap_is_seeded_by_key():
    values = [0.0, 0.5, 1.0, 0.25, 0.75]
    assert bootstrap_means(values, "k", 200) == bootstrap_means(values, "k", 200)
    assert bootstrap_means(values, "k", 200) != bootstrap_means(values, "other", 200)


def test_ci_picks_wilson_for_pass_fail_and_bootstrap_otherwise():
    interval, method = ci([True, False, True, True], "k")
    assert method == "wilson"
    assert interval == list(wilson(3, 4))
    interval, method = ci([0.0, 0.5, 1.0, 1.0], "k")
    assert method == "bootstrap"
    assert 0.0 <= interval[0] < interval[1] <= 1.0
    assert ci([0.5, 0.5], "k") == ([0.5, 0.5], "bootstrap")  # every resample is identical
    assert ci([], "k") == (None, None)


def test_stat_flags_small_samples_as_indicative():
    assert stat([True] * (MIN_HEADLINE_N - 1), "k", MIN_HEADLINE_N)["indicative"] is True
    big = stat([True] * MIN_HEADLINE_N, "k", MIN_HEADLINE_N)
    assert big["indicative"] is False and big["n"] == MIN_HEADLINE_N


def test_mcnemar_exact():
    assert mcnemar([True] * 5, [True] * 5) == 1.0  # no discordant pairs
    # 10 discordant pairs, all one way: p = 2 * 0.5**10
    assert mcnemar([True] * 10, [False] * 10) == pytest.approx(2 / 1024)
    assert mcnemar([True, False], [False, True]) == 1.0


def test_paired_diff_flags_a_ci_that_includes_zero():
    same = paired_diff([True, False] * 50, [True, False] * 50, "k")
    assert same["diff"] == 0 and same["ci"] == [0.0, 0.0]
    assert same["ci_includes_zero"] and same["p"] == 1.0 and same["test"] == "mcnemar"

    better = paired_diff([False] * 100, [True] * 60 + [False] * 40, "k")
    assert better["diff"] == pytest.approx(0.6)
    assert not better["ci_includes_zero"] and better["p"] < 1e-6

    numeric = paired_diff([0.0, 0.5, 1.0] * 20, [0.5, 1.0, 1.0] * 20, "k")
    assert numeric["test"] == "bootstrap" and numeric["diff"] > 0


def test_holm_adjusts_in_input_order():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]) == [1.0, 1.0]
    assert holm([]) == []


def test_summarize_repeats_mean_sd_values():
    summary = summarize_repeats([{"a": 0.5, "b": None}, {"a": 0.7, "b": 1.0}])
    assert summary["n"] == 2
    assert summary["scores"]["a"]["mean"] == pytest.approx(0.6)
    assert summary["scores"]["a"]["sd"] == pytest.approx(0.1414, abs=1e-4)
    assert summary["scores"]["a"]["values"] == [0.5, 0.7]
    assert summary["scores"]["b"] == {"mean": None, "sd": None, "values": [None, 1.0]}


def _query(category, expect="x"):
    return QueryStep(type="query", session_id="s", query="q", timestamp="t",
                     expect_contains=[expect], category=category)


def _trace(index, context):
    return StepTrace(index, "query", "s", 1.0, {
        "context": context, "items": [], "token_count": len(context), "ingest_latency_ms": None,
    })


def test_score_detailed_outcomes_and_stats():
    steps = [_query("a"), _query("a"), _query("b")]
    traces = [_trace(0, "x"), _trace(1, "no"), _trace(2, "x")]
    scoring = score_detailed(Pack("p", "", steps), traces, seed=3)

    assert scoring.outcomes["recall_precision"] == {"0": True, "1": False, "2": True}
    assert scoring.scores["recall_precision"] == pytest.approx(2 / 3)
    rp = scoring.stats["scores"]["recall_precision"]
    assert rp["n"] == 3 and rp["method"] == "wilson" and rp["indicative"] is True
    assert rp["ci"] == list(wilson(2, 3))
    assert scoring.by_category["a"]["recall_precision"] == 0.5
    cat = scoring.stats["by_category"]["a"]["recall_precision"]
    assert cat["n"] == 2 and cat["indicative"] is (2 < MIN_CATEGORY_N)
    assert scoring.stats["scores"]["recall_p95_ms"]["n"] == 3
    assert scoring.stats["scores"]["token_cost"]["method"] == "bootstrap"
    # Seeded: the same input gives the same intervals.
    again = score_detailed(Pack("p", "", steps), traces, seed=3)
    assert again.stats == scoring.stats
