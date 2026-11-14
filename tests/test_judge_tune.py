"""`judge-tune`: prompt-version plumbing, dev/held-out split and tuning report.

Every label here is SYNTHETIC test data (annotators `synthetic-*`, items `synthetic-*`);
the prompt version "test-2" exists only inside these tests (monkeypatched).
"""

import json
import random

import pytest
from typer.testing import CliRunner

from membench.cli import app
from membench.core.scoring import judge
from membench.core.scoring.judge import (
    CachedJudge,
    CacheMissError,
    JudgeCache,
    build_judge,
    build_prompt,
)
from membench.core.validation.cross import share_cache
from membench.core.validation.sample import item_prompt
from membench.core.validation.tune import (
    join_sample,
    prompt_sha256,
    render_tuning_markdown,
    run_versions,
    split_items,
    tuning_report,
)

MODEL = "synthetic-model"
# Computed from the unversioned code before `version` parameters were added.
V1_KEY_M_R_E = "m:v1:764638828a5f28923e493d711bb98e14d3ea3649efcbfb03f5fc20f65b36954c"
V1_PROMPT_SHA = "aab7974c98a3378ca5864dcee543ec0bb3d748f650ddf7edb213cc0743ae2016"


@pytest.fixture
def v2(monkeypatch):
    monkeypatch.setitem(judge.PROMPT_BUILDERS, "test-2",
                        lambda r, e: "TEST2\n" + judge.PROMPT_BUILDERS["1"](r, e))
    monkeypatch.delenv(judge.API_KEY_ENV, raising=False)
    return "test-2"


class StubClient:
    """Answers from markers in the evidence: `v1:yes` for v1, `v2:yes` for test-2;
    `broken` gives an unparseable reply; LongMemEval-style prompts get yes/no."""

    model = MODEL

    def __init__(self):
        self.calls = 0

    def respond(self, prompt):
        self.calls += 1
        if prompt.startswith("LME"):
            return ("yes" if "lme:yes" in prompt else "no"), {"input_tokens": 1}
        evidence = prompt.split("EVIDENCE:")[-1]
        if "broken" in evidence:
            return "no json here", {"input_tokens": 1}
        marker = "v2:yes" if prompt.startswith("TEST2") else "v1:yes"
        verdict = "pass" if marker in evidence else "fail"
        return json.dumps({"verdict": verdict, "reason": "stub"}), {"input_tokens": 5,
                                                                     "output_tokens": 1}


def _label(item_id, gold, metric="semantic_recall", bucket="<128", tie=False):
    labels = ({"synthetic-a": True, "synthetic-b": False} if tie
              else {"synthetic-a": gold, "synthetic-b": gold})
    return {"item_id": item_id, "metric": metric, "backend": "synthetic-backend",
            "length_bucket": bucket, "prompt_version": "1", "judge_model": MODEL,
            "labels": labels, "adjudicated": None}


def _item(item_id, evidence, metric="semantic_recall", bucket="<128", prompt=None):
    item = {"id": item_id, "metric": metric, "backend": "synthetic-backend",
            "pack": "synthetic", "rubric": f"statement {item_id}", "evidence": evidence,
            "tokens": 3, "length_bucket": bucket}
    if prompt:
        item["prompt"] = prompt
    return item


# id, gold, v1 evidence markers (test-2 is always right), bucket
TINY = [
    ("synthetic-a", True, "v1:yes v2:yes", "<128"),
    ("synthetic-b", True, "v2:yes", "<128"),
    ("synthetic-c", False, "", "<128"),
    ("synthetic-d", False, "v1:yes", "128-511"),
    ("synthetic-e", True, "v1:yes v2:yes", "128-511"),
    ("synthetic-f", False, "", "128-511"),
    ("synthetic-g", True, "broken", "<128"),
]


def _tiny():
    labels = [_label(i, g, bucket=b) for i, g, _, b in TINY]
    labels.append(_label("synthetic-h", True, tie=True))  # unresolved
    sample = [_item(i, e, bucket=b) for i, _, e, b in TINY]
    sample.append(_item("synthetic-h", "v1:yes"))
    return labels, sample


def _judges(versions, cache_path, client=None, strict=False):
    judges = {v: CachedJudge(client, cache_path, strict=strict, model=MODEL, prompt_version=v)
              for v in versions}
    share_cache(judges, cache_path)
    return judges


def _run(tmp_path, versions, labels, sample, client=None, strict=False, **kw):
    items = join_sample(labels, sample)
    judges = _judges(versions, tmp_path / "cache.json", client, strict)
    verdicts, misses = run_versions(items, judges)
    report = tuning_report(labels, items, verdicts, judge_model=MODEL,
                           usage={v: j.usage for v, j in judges.items()}, **kw)
    return report, misses, judges


# --- prompt-version plumbing ---------------------------------------------------------


def test_default_key_and_prompt_are_unchanged():
    assert JudgeCache.key("m", build_prompt("R", "E")) == V1_KEY_M_R_E
    assert JudgeCache.key("m", build_prompt("R", "E", "1"), "1") == V1_KEY_M_R_E
    assert prompt_sha256("1") == V1_PROMPT_SHA
    assert judge.JUDGE_PROMPT_VERSION == "1"


def test_versioned_key_and_prompt_differ(v2):
    prompt = build_prompt("R", "E")
    assert JudgeCache.key("m", prompt, v2) != JudgeCache.key("m", prompt)
    assert JudgeCache.key("m", prompt, v2).startswith("m:vtest-2:")
    assert build_prompt("R", "E", v2) != prompt


def test_unknown_version_is_a_clear_error(tmp_path):
    with pytest.raises(ValueError, match="known: 1"):
        build_prompt("R", "E", "nope")
    with pytest.raises(ValueError, match="unknown judge prompt version"):
        CachedJudge(None, tmp_path / "c.json", strict=True, model="m", prompt_version="nope")


def test_cached_judge_uses_its_prompt_version(tmp_path, v2):
    client = StubClient()
    j = build_judge(model=MODEL, cache_path=tmp_path / "c.json", api_key="dummy",
                    prompt_version=v2)
    j.client = client
    assert j.prompt_version == v2
    assert j.score("R", "v2:yes").passed
    key = JudgeCache.key(MODEL, build_prompt("R", "v2:yes", v2), v2)
    assert JudgeCache(tmp_path / "c.json").get(key)["passed"] is True
    assert build_judge(model=MODEL, cache_path=tmp_path / "c.json",
                       strict=True).prompt_version == "1"


def test_fixed_template_items_ignore_the_version(v2):
    item = _item("x", "e", prompt="LME fixed")
    assert item_prompt(item, v2) == item_prompt(item)
    plain = _item("y", "e")
    assert item_prompt(plain)[0] == build_prompt(plain["rubric"], "e")
    assert item_prompt(plain, v2)[0] == build_prompt(plain["rubric"], "e", v2)


# --- split ---------------------------------------------------------------------------


def _many():
    labels = [_label(f"synthetic-sr-{i:02d}", i % 2 == 0) for i in range(21)]
    labels += [_label(f"synthetic-cr-{i:02d}", i % 3 == 0, metric="contradiction_rate")
               for i in range(10)]
    labels += [_label(f"synthetic-tie-{i}", True, tie=True) for i in range(3)]
    return labels


def test_split_is_deterministic_stratified_and_order_independent():
    labels = _many()
    split = split_items(labels, seed=3)
    shuffled = labels[:]
    random.Random(1).shuffle(shuffled)
    assert split_items(shuffled, seed=3) == split
    assert not any(k.startswith("synthetic-tie") for k in split)
    sr = [v for k, v in split.items() if "-sr-" in k]
    cr = [v for k, v in split.items() if "-cr-" in k]
    assert sr.count("held_out") == round(21 * 0.5) and cr.count("held_out") == 5
    assert split_items(labels, seed=4) != split
    # A metric's split does not depend on the other metrics' items.
    only_sr = split_items([x for x in labels if "-sr-" in x["item_id"]], seed=3)
    assert only_sr == {k: v for k, v in split.items() if "-sr-" in k}


def test_split_fraction():
    labels = _many()
    assert set(split_items(labels, held_out_fraction=0).values()) == {"dev"}
    assert set(split_items(labels, held_out_fraction=1).values()) == {"held_out"}
    held = list(split_items(labels, held_out_fraction=0.3).values()).count("held_out")
    assert held == round(21 * 0.3) + round(10 * 0.3)
    with pytest.raises(ValueError):
        split_items(labels, held_out_fraction=1.5)


# --- report --------------------------------------------------------------------------


def test_stats_on_a_hand_computed_case(tmp_path, v2):
    labels, sample = _tiny()
    client = StubClient()
    report, misses, _ = _run(tmp_path, ["1", v2], labels, sample, client)
    assert misses == {}
    assert report["n_items"] == 7 and report["n_unresolved"] == 1
    v1 = report["results"]["1"]["all"]["overall"]
    # TP a,e; FN b; TN c,f; FP d; g errored.
    assert v1 == {**v1, "n": 6, "kappa": 0.3333, "accuracy": 0.6667, "false_pass": 1,
                  "false_pass_rate": 0.3333, "false_fail": 1, "false_fail_rate": 0.3333,
                  "errors": 1}
    short = v1["by_length"]["<128"]  # judge [T,F,F] vs human [T,T,F], + g errored
    assert short == {"n": 3, "kappa": 0.4, "accuracy": 0.6667, "false_pass": 0,
                     "false_pass_rate": 0.0, "false_fail": 1, "false_fail_rate": 0.5,
                     "errors": 1}
    assert v1["by_length"]["128-511"]["n"] == 3
    assert "512-2047" not in v1["by_length"]  # empty buckets omitted
    cand = report["results"][v2]["all"]["semantic_recall"]
    assert (cand["kappa"], cand["accuracy"], cand["errors"]) == (1.0, 1.0, 1)
    assert report["deltas"]["all"]["overall"]["kappa"] == round(1.0 - 0.3333, 4)
    assert report["baseline"] == "1" and report["candidate"] == v2
    assert report["prompt_varies"] == {"semantic_recall": True}
    assert report["prompt_sha256"]["1"] == V1_PROMPT_SHA
    assert any("errors excluded" in w for w in report["warnings"])
    assert any("512-2047, 2048+" in w for w in report["warnings"])
    assert any("held-out once" in w for w in report["warnings"])
    # Per split: dev and held-out partition `all`, each counted the same way.
    split = split_items(labels)
    gold = {i: g for i, g, _, _ in TINY}
    v1_pass = {"synthetic-a": True, "synthetic-b": False, "synthetic-c": False,
               "synthetic-d": True, "synthetic-e": True, "synthetic-f": False}
    for s in ("dev", "held_out"):
        ids = [i for i in v1_pass if split[i] == s]
        got = report["results"]["1"][s]["overall"]
        assert got["n"] == len(ids)
        assert got["false_pass"] == sum(v1_pass[i] and not gold[i] for i in ids)
        assert got["false_fail"] == sum(gold[i] and not v1_pass[i] for i in ids)
        assert got["errors"] == (split["synthetic-g"] == s)
    sizes = report["split_sizes"]["semantic_recall"]
    assert sizes["dev"] + sizes["held_out"] == 7


def test_disagreements_list_dev_items_only(tmp_path, v2):
    labels, sample = _tiny()
    report, _, _ = _run(tmp_path, ["1", v2], labels, sample, StubClient())
    split = split_items(labels)
    dev_wrong = sorted(i for i in ("synthetic-b", "synthetic-d") if split[i] == "dev")
    assert report["disagreements"]["1"] == dev_wrong
    assert report["disagreements"][v2] == []
    listed = {i for ids in report["disagreements"].values() for i in ids}
    assert not listed & {i for i, s in split.items() if s == "held_out"}


def test_strict_misses_are_reported_not_raised(tmp_path, v2):
    labels, sample = _tiny()
    report, misses, _ = _run(tmp_path, ["1", v2], labels, sample, strict=True)
    assert set(misses) == {"1", v2} and len(misses["1"]) == 8
    assert report["results"]["1"]["all"]["overall"]["errors"] == 7


def test_second_run_is_served_from_cache(tmp_path, v2):
    labels, sample = _tiny()
    labels, sample = labels[:-2], sample[:-2]  # drop `broken` (errors are re-asked) and tie
    first = StubClient()
    _run(tmp_path, ["1", v2], labels, sample, first)
    assert first.calls == 12
    second = StubClient()
    report, _, _ = _run(tmp_path, ["1", v2], labels, sample, second)
    assert second.calls == 0
    assert report["usage"]["1"]["cache_hits"] == 6 and report["usage"][v2]["calls"] == 0


def test_fixed_template_metric_is_flagged(tmp_path, v2):
    metric = "qa_accuracy:longmemeval"
    labels = [_label("synthetic-l1", True, metric=metric),
              _label("synthetic-l2", False, metric=metric)]
    sample = [_item("synthetic-l1", "x", metric, prompt="LME lme:yes"),
              _item("synthetic-l2", "x", metric, prompt="LME no")]
    client = StubClient()
    report, _, _ = _run(tmp_path, ["1", v2], labels, sample, client)
    assert client.calls == 4  # still judged once per version (keyed under each)
    assert report["prompt_varies"] == {metric: False}
    assert any("prompt_varies is false" in w for w in report["warnings"])
    assert report["results"]["1"]["all"][metric] == report["results"][v2]["all"][metric]


def test_labels_missing_from_sample_are_an_error():
    labels, sample = _tiny()
    with pytest.raises(ValueError, match="not in the sample"):
        join_sample(labels, sample[:-1])


def test_markdown_renders_none_as_na(tmp_path, v2):
    labels, sample = _tiny()
    report, _, _ = _run(tmp_path, ["1", v2], labels, sample, StubClient(),
                        held_out_fraction=0)
    assert report["deltas"]["held_out"]["overall"]["kappa"] is None
    md = render_tuning_markdown(report)
    assert "n/a" in md and "## Held-out deltas (vtest-2 − v1)" in md
    assert "## By context length" in md and "## How to read" in md


# --- CLI -----------------------------------------------------------------------------


def _write(tmp_path, labels, sample):
    (tmp_path / "labels.jsonl").write_text("".join(json.dumps(x) + "\n" for x in labels))
    (tmp_path / "sample.jsonl").write_text("".join(json.dumps(x) + "\n" for x in sample))


def _args(tmp_path, *versions):
    args = ["judge-tune", "--labels", str(tmp_path / "labels.jsonl"),
            "--sample", str(tmp_path / "sample.jsonl"), "--judge-model", MODEL,
            "--judge-cache", str(tmp_path / "cache.json"), "--out",
            str(tmp_path / "tuning.json")]
    for v in versions:
        args += ["--prompt-version", v]
    return args


def test_cli_happy_path_from_a_prefilled_cache(tmp_path, v2):
    labels, sample = _tiny()
    _write(tmp_path, labels, sample)
    _run(tmp_path, ["1", v2], labels, sample, StubClient())  # pre-fill the cache
    result = CliRunner().invoke(app, _args(tmp_path, "1", v2))
    assert result.exit_code == 0, result.output
    assert "using cached verdicts only" in result.output
    report = json.loads((tmp_path / "tuning.json").read_text())
    assert report["versions"] == ["1", v2] and report["n_items"] == 7
    assert report["results"]["1"]["all"]["overall"]["kappa"] == 0.3333
    assert "# Judge prompt tuning" in (tmp_path / "tuning.md").read_text()


def test_cli_strict_misses_exit_1(tmp_path, v2):
    labels, sample = _tiny()
    _write(tmp_path, labels, sample)
    result = CliRunner().invoke(app, _args(tmp_path, "1", v2) + ["--strict"])
    assert result.exit_code == 1
    assert "v1 lacks cached verdicts for 8 items" in result.output
    assert judge.API_KEY_ENV in result.output
    assert not (tmp_path / "tuning.json").exists()


def test_cli_errors(tmp_path, v2):
    labels, sample = _tiny()
    _write(tmp_path, labels, sample)
    runner = CliRunner()
    one = runner.invoke(app, _args(tmp_path, "1", "1"))
    assert one.exit_code == 1 and "at least two distinct" in one.output
    unknown = runner.invoke(app, _args(tmp_path, "1", "nope"))
    assert unknown.exit_code == 1 and "unknown prompt version(s) nope" in unknown.output
    (tmp_path / "labels.jsonl").unlink()
    missing = runner.invoke(app, _args(tmp_path, "1", v2))
    assert missing.exit_code == 1 and "Error:" in missing.output


def test_strict_cache_miss_error_type_is_unchanged(tmp_path):
    j = CachedJudge(None, tmp_path / "c.json", strict=True, model="m")
    with pytest.raises(CacheMissError):
        j.score("R", "E")
