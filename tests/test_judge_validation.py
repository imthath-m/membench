"""Judge validation (Phase 15): sampler, labels, agreement math, cross-judge, status.

Every label and verdict here is SYNTHETIC test data (annotators `synthetic-*`,
judges `fixture-*`): it checks the arithmetic, it is not a validation result.
"""

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from membench.adapters.full_context import FullContextAdapter
from membench.cli import app
from membench.core.reader import Reader
from membench.core.report.html import render_html
from membench.core.report.json import to_dict
from membench.core.report.markdown import render_markdown
from membench.core.runner import run_pack
from membench.core.scoring import score_detailed
from membench.core.scoring.judge import (
    JUDGE_PROMPT_VERSION,
    CachedJudge,
    CacheMissError,
    JudgeCache,
    Verdict,
    build_prompt,
)
from membench.core.scoring.qa_accuracy import qa_rubric
from membench.core.validation.agreement import (
    agreement_report,
    binary_agreement,
    cohen_kappa,
    render_agreement_markdown,
)
from membench.core.validation.cross import (
    JudgeConfig,
    build_panel_judge,
    load_panel,
    run_judges,
    share_cache,
)
from membench.core.validation.labels import (
    LabelsError,
    gold,
    load_labels,
    merge_annotations,
    validate_labels,
)
from membench.core.validation.sample import (
    item_prompt,
    judge_items,
    length_bucket,
    stratified_sample,
)
from membench.core.validation.status import metric_status, report_metric_status
from membench.generate.synthetic import Knobs, generate
from membench.packs.schema import builtin_pack_path, load_pack, parse_pack

FIXTURES = Path(__file__).parent / "fixtures" / "validation"
REPO = Path(__file__).parent.parent
runner = CliRunner()

T, F = True, False


# --- Cohen's kappa and judge-vs-human rates -------------------------------------------


def test_cohen_kappa_textbook_example():
    # 20 both-yes, 5 A-yes/B-no, 10 A-no/B-yes, 15 both-no: po .7, pe .5 -> kappa .4
    a = [T] * 20 + [T] * 5 + [F] * 10 + [F] * 15
    b = [T] * 20 + [F] * 5 + [T] * 10 + [F] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)


def test_cohen_kappa_edge_cases():
    assert cohen_kappa([T, F, T, F], [T, F, T, F]) == 1.0
    assert cohen_kappa([T, F, T, F], [F, T, F, T]) == -1.0
    assert cohen_kappa([T, T, T], [T, T, T]) is None  # undefined: no variation at all
    assert cohen_kappa([], []) is None


def test_binary_agreement_false_pass_and_false_fail():
    judge = [T, T, F, F, T, F]
    human = [T, F, F, T, T, F]  # one false pass (idx 1), one false fail (idx 3)
    got = binary_agreement(judge, human)
    assert got["n"] == 6 and got["accuracy"] == pytest.approx(4 / 6, abs=1e-4)
    assert got["false_pass"] == 1 and got["false_pass_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert got["false_fail"] == 1 and got["false_fail_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert got["kappa"] == pytest.approx(cohen_kappa(judge, human), abs=1e-4)


# --- labels format -----------------------------------------------------------------------


def test_fixture_labels_load_and_gold_follows_adjudication():
    lines = load_labels(FIXTURES / "synthetic_labels.jsonl")
    by_id = {line["item_id"]: line for line in lines}
    assert gold(by_id["fixture-01"]) is True  # both annotators agree
    assert gold(by_id["fixture-03"]) is False  # disagreement, adjudicated false
    assert gold({"labels": {"a": T, "b": F}, "adjudicated": None}) is None
    assert gold({"labels": {"a": T, "b": F, "c": T}}) is True  # majority of three


def _line(item_id="x1", **over):
    line = {"item_id": item_id, "metric": "semantic_recall", "backend": "bm25",
            "length_bucket": "<128", "prompt_version": "1", "judge_model": "m",
            "labels": {"synthetic-a": T, "synthetic-b": T}, "adjudicated": None}
    return {**line, **over}


@pytest.mark.parametrize("lines, message", [
    ([{k: v for k, v in _line().items() if k != "backend"}], "`backend`"),
    ([_line(metric="recall_precision")], "unknown metric"),
    ([_line(), _line()], "duplicate item_id"),
    ([_line(labels={"synthetic-a": T})], ">= 2 annotators"),
    ([_line(labels={"synthetic-a": T, "synthetic-b": "yes"})], ">= 2 annotators"),
    ([_line(adjudicated="yes")], "`adjudicated`"),
    ([_line(), _line("x2", labels={"synthetic-a": T, "synthetic-c": T})], "differ"),
])
def test_validate_labels_rejects_bad_lines(lines, message):
    with pytest.raises(LabelsError, match=message):
        validate_labels(lines)


def test_load_labels_rejects_bad_json(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text(json.dumps(_line()) + "\n{not json\n")
    with pytest.raises(LabelsError, match="labels.jsonl:2"):
        load_labels(path)


def test_merge_annotations_keeps_items_every_annotator_labelled():
    sample = [{"id": i, "metric": "semantic_recall", "backend": "bm25", "length_bucket": "<128",
               "prompt_version": "1", "judge_model": "m"} for i in ("i1", "i2", "i3")]
    lines = merge_annotations(
        sample,
        {"synthetic-a": [{"item_id": "i1", "label": T}, {"item_id": "i2", "label": T},
                         {"item_id": "i3", "label": F}],
         "synthetic-b": [{"item_id": "i1", "label": T, "note": "clear"},
                         {"item_id": "i2", "label": F}]},
        [{"item_id": "i2", "label": F}],
    )
    assert [line["item_id"] for line in lines] == ["i1", "i2"]  # i3: only one annotator
    assert lines[0]["notes"] == "synthetic-b: clear" and gold(lines[1]) is False
    with pytest.raises(LabelsError, match="unknown item_id"):
        merge_annotations(sample, {"a": [{"item_id": "zz", "label": T}], "b": []})


# --- agreement report --------------------------------------------------------------------


def test_agreement_on_fixture_labels_and_verdicts():
    report = agreement_report(load_labels(FIXTURES / "synthetic_labels.jsonl"),
                              [json.loads(x) for x in
                               (FIXTURES / "synthetic_verdicts.jsonl").read_text().splitlines()])
    human = report["human"]["semantic_recall"]
    # annotators disagree on fixture-03 and fixture-07 only (8 of 10 agree)
    assert human["n"] == 10 and human["agreement"] == 0.8 and human["unresolved"] == 0
    a = [T, T, T, F, T, F, F, T, F, F]
    b = [T, T, F, F, T, F, T, T, F, F]
    assert human["kappa"] == pytest.approx(cohen_kappa(a, b), abs=1e-4)

    x = report["judges"]["fixture-x"]["metrics"]["semantic_recall"]
    assert x["n"] == 10 and x["false_pass"] == 1 and x["false_fail"] == 1
    assert x["errors"] == 0
    assert set(x["by_length"]) == {"<128", "512-2047"}
    assert x["by_length"]["512-2047"]["n"] == 4
    assert x["status"] == "unvalidated" and any("n 10 < 200" in r for r in x["reasons"])
    assert report["judges"]["fixture-y"]["metrics"]["semantic_recall"]["errors"] == 1

    pair = report["pairwise"]["semantic_recall"][0]
    assert (pair["a"], pair["b"], pair["n"]) == ("fixture-x", "fixture-y", 9)
    assert pair["disagreements"] == ["fixture-02", "fixture-05"]
    md = render_agreement_markdown(report)
    assert "fixture-02, fixture-05" in md and "## By context length" in md


def _synthetic_study(n, human_flips, judge_flips, metric="semantic_recall"):
    """n items with a balanced truth; annotator b flips `human_flips` items (adjudicated
    back to the truth), the judge flips `judge_flips` items."""
    labels, verdicts = [], []
    for i in range(n):
        truth = i % 2 == 0
        b = not truth if i in human_flips else truth
        labels.append(_line(f"s{i}", metric=metric, length_bucket=length_bucket(i * 10),
                            labels={"synthetic-a": truth, "synthetic-b": b},
                            adjudicated=truth if b != truth else None))
        verdicts.append({"item_id": f"s{i}", "metric": metric, "judge": "fixture-j",
                         "model": "fixture-model", "base_url": None, "prompt_version": "1",
                         "passed": not truth if i in judge_flips else truth})
    return labels, verdicts


def test_validated_when_thresholds_are_met():
    labels, verdicts = _synthetic_study(220, set(range(0, 220, 20)), set(range(1, 220, 12)))
    report = agreement_report(validate_labels(labels), verdicts)
    m = report["judges"]["fixture-j"]["metrics"]["semantic_recall"]
    assert report["human"]["semantic_recall"]["kappa"] >= 0.7
    assert m["kappa"] >= 0.6 and m["status"] == "validated" and m["reasons"] == []


@pytest.mark.parametrize("judge_flips, human_flips, adjudicate, reason", [
    (set(range(0, 220, 3)), set(), True, "judge-human kappa"),  # judge too noisy
    (set(range(0, 220, 5)), set(), True, "more than 0.1 below"),  # within .6 but far below
    (set(), set(range(0, 220, 4)), True, "human-human kappa"),  # humans disagree too much
    (set(), {0}, False, "not adjudicated"),
])
def test_unvalidated_reasons(judge_flips, human_flips, adjudicate, reason):
    labels, verdicts = _synthetic_study(220, human_flips, judge_flips)
    if not adjudicate:
        for line in labels:
            line["adjudicated"] = None
    m = agreement_report(labels, verdicts)["judges"]["fixture-j"]["metrics"]["semantic_recall"]
    assert m["status"] == "unvalidated"
    assert any(reason in r for r in m["reasons"]), m["reasons"]


def test_agreement_refuses_mixed_prompt_versions():
    _, verdicts = _synthetic_study(4, set(), set())
    verdicts[0]["prompt_version"] = "0"
    with pytest.raises(ValueError, match="mix judge prompt versions"):
        agreement_report(None, verdicts)


# --- sampler -----------------------------------------------------------------------------


def _synthetic_report(episodes=3):
    pack = parse_pack(generate("customer_support_xl", knobs=Knobs(episodes=episodes)))
    return pack, to_dict(run_pack(pack, FullContextAdapter({}), backend_name="full-context"))


class RecordingJudge:
    model = "rec"

    def __init__(self):
        self.calls = []

    def score(self, rubric, evidence):
        self.calls.append((rubric, evidence))
        return Verdict(passed=True)


def test_judge_items_are_exactly_what_the_judge_sees():
    pack, report = _synthetic_report()
    judge = RecordingJudge()
    score_detailed(pack, run_pack(pack, FullContextAdapter({}), backend_name="fc").traces,
                   judge)
    items = judge_items(report, pack)
    assert {(i["rubric"], i["evidence"]) for i in items} == set(judge.calls)
    counts = {m: sum(i["metric"] == m for i in items) for m in ("semantic_recall",
                                                                 "contradiction_rate")}
    assert counts == {"semantic_recall": 12, "contradiction_rate": 3}  # 4 and 1 per episode
    assert all(i["verdict"] is None and i["length_bucket"] == length_bucket(i["tokens"])
               for i in items)
    assert all("judge_error" not in t["detail"] for t in report["traces"])  # not mutated
    assert item_prompt(items[0])[0] == build_prompt(items[0]["rubric"], items[0]["evidence"])


def test_judge_items_take_cached_verdicts(tmp_path):
    pack, report = _synthetic_report(1)
    first = judge_items(report, pack)[0]
    cache = JudgeCache(tmp_path / "cache.json")
    cache.set(JudgeCache.key("m", item_prompt(first)[0]), {"passed": False, "reason": ""})
    items = judge_items(report, pack, cache, "m")
    assert items[0]["verdict"] is False and items[0]["judge_model"] == "m"
    assert items[1]["verdict"] is None


def test_judge_items_ignore_the_runs_own_judge_errors():
    pack, report = _synthetic_report(1)
    expected = judge_items(report, pack)
    report["traces"][0]["detail"]["judge_error"] = {"semantic_recall": "old failure"}
    assert judge_items(report, pack) == expected


def test_qa_items_use_reader_answers_and_longmemeval_prompts():
    class Client:
        model, base_url = "reader", None

        def respond(self, prompt):
            return "Blue", {"input_tokens": 1, "output_tokens": 1}

    pack = load_pack(builtin_pack_path("mini"))
    for style in ("membench", "longmemeval"):
        pack.qa_style = style
        result = run_pack(pack, FullContextAdapter({}), backend_name="fc",
                          reader=Reader(Client()))
        qa = [i for i in judge_items(to_dict(result), pack) if i["metric"].startswith("qa")]
        assert qa and all(i["evidence"] == "Blue" for i in qa)
        if style == "membench":
            assert {i["metric"] for i in qa} == {"qa_accuracy"} and "prompt" not in qa[0]
            assert qa[0]["rubric"].startswith(qa_rubric("", "")[:20])
        else:
            assert {i["metric"] for i in qa} == {"qa_accuracy:longmemeval"}
            assert item_prompt(qa[0])[0] == qa[0]["prompt"]


def test_stratified_sample_covers_every_stratum_and_is_seeded():
    items = [{"id": f"{b}{k}", "metric": "semantic_recall", "backend": b, "verdict": None,
              "length_bucket": "<128"} for b, n in (("big", 50), ("small", 3)) for k in range(n)]
    items.append(dict(items[0]))  # duplicate id: counted once
    sample = stratified_sample(items, 10, seed=1)
    assert len(sample) == 10 and len({i["id"] for i in sample}) == 10
    assert sum(i["backend"] == "small" for i in sample) == 3  # small stratum fully covered
    assert sample == stratified_sample(items, 10, seed=1)
    assert sample != stratified_sample(items, 10, seed=2)
    assert len(stratified_sample(items, 500)) == 53


def test_cli_judge_sample(tmp_path):
    out = tmp_path / "report.json"
    result = runner.invoke(app, ["run", "--backend", "full-context", "--pack", "document-chain",
                                 "--no-judge", "--out", str(out)])
    assert result.exit_code == 0, result.output
    sample_path = tmp_path / "sample.jsonl"
    result = runner.invoke(app, ["judge-sample", "--in", str(out), "--n", "200",
                                 "--judge-cache", str(tmp_path / "none.json"),
                                 "--out", str(sample_path)])
    assert result.exit_code == 0, result.output
    sample = [json.loads(x) for x in sample_path.read_text().splitlines()]
    assert sum(i["metric"] == "semantic_recall" for i in sample) == 144
    assert sum(i["metric"] == "contradiction_rate" for i in sample) == 36
    assert "fewer than 200" in result.output


# --- cross-judge -------------------------------------------------------------------------


def test_committed_panel_has_three_judges_incl_open_weight():
    panel = load_panel(REPO / "configs" / "judges.yaml")
    assert len(panel) >= 3 and len({j.model for j in panel}) == len(panel)
    assert any(j.base_url is None for j in panel)  # Anthropic (Claude)
    assert sum(j.base_url is not None for j in panel) >= 2  # OpenAI-compatible
    assert any("llama" in j.model.lower() for j in panel)  # open weight


def test_panel_judge_needs_its_own_key(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMBENCH_JUDGE_API_KEY", "other-key")
    monkeypatch.delenv("FIXTURE_KEY", raising=False)
    config = JudgeConfig("x", "m", "https://example.invalid/v1", "FIXTURE_KEY")
    with pytest.raises(ValueError, match=r"\$FIXTURE_KEY"):
        build_panel_judge(config, tmp_path / "c.json", strict=False)
    monkeypatch.setenv("FIXTURE_KEY", "k")
    judge = build_panel_judge(config, tmp_path / "c.json", strict=False)
    assert judge.client.api_key == "k" and judge.base_url == "https://example.invalid/v1"


def test_run_judges_rows_and_errors(tmp_path):
    class Client:
        def __init__(self, model, reply):
            self.model, self.base_url, self.reply = model, None, reply

        def respond(self, prompt):
            return self.reply(prompt), {"input_tokens": 1, "output_tokens": 1}

    items = [{"id": "a", "metric": "semantic_recall", "rubric": "r", "evidence": "green"},
             {"id": "b", "metric": "semantic_recall", "rubric": "r", "evidence": "red"}]
    judges = {
        "fixture-x": CachedJudge(Client("x", lambda p: '{"verdict": "pass"}'), tmp_path / "c"),
        "fixture-y": CachedJudge(
            Client("y", lambda p: '{"verdict": "pass"}' if "green" in p else "no json"),
            tmp_path / "c"),
    }
    share_cache(judges, tmp_path / "c")  # one cache file, many judges: no lost writes
    rows = run_judges(items, judges)
    assert [(r["judge"], r["item_id"], r["passed"]) for r in rows] == [
        ("fixture-x", "a", True), ("fixture-x", "b", True),
        ("fixture-y", "a", True), ("fixture-y", "b", None)]
    assert "failed to parse" in rows[-1]["error"]
    assert all(r["prompt_version"] == JUDGE_PROMPT_VERSION for r in rows)
    strict = {"fixture-x": CachedJudge(None, tmp_path / "c", strict=True, model="x")}
    assert run_judges(items, strict)[0]["passed"] is True  # replay from the cache
    with pytest.raises(CacheMissError):
        run_judges(items, {"z": CachedJudge(None, tmp_path / "c", strict=True, model="z")})


def test_cli_cross_judge_and_agreement(tmp_path):
    sample = tmp_path / "sample.jsonl"
    sample.write_text(json.dumps({"id": "a", "metric": "semantic_recall", "rubric": "r",
                                  "evidence": "e"}) + "\n")
    panel = tmp_path / "judges.yaml"
    panel.write_text("judges:\n  - {name: fixture-x, model: x}\n")
    cache = tmp_path / "cache.json"
    JudgeCache(cache).set(JudgeCache.key("x", build_prompt("r", "e")),
                          {"passed": True, "reason": ""})
    verdicts = tmp_path / "verdicts.jsonl"
    args = ["judge-cross", "--sample", str(sample), "--judges", str(panel),
            "--judge-cache", str(cache), "--strict-cache", "--out", str(verdicts)]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    out = tmp_path / "agreement-v1.json"
    result = runner.invoke(app, ["judge-agreement", "--verdicts", str(verdicts),
                                 "--labels", str(FIXTURES / "synthetic_labels.jsonl"),
                                 "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text())["judges"]["fixture-x"]["model"] == "x"
    assert out.with_suffix(".md").exists()
    assert runner.invoke(app, ["judge-agreement"]).exit_code == 1


# --- status lookup and reports -----------------------------------------------------------


def _agreement_file(directory, model="fixture-model", status="validated", version="1"):
    data = {"prompt_version": version, "judges": {"j": {"model": model, "metrics": {
        "semantic_recall": {"status": status}}}}}
    (directory / f"agreement-v{version}.json").write_text(json.dumps(data))


def test_metric_status_lookup(tmp_path):
    assert metric_status("semantic_recall", "fixture-model", "1", tmp_path) == "unvalidated"
    _agreement_file(tmp_path)
    assert metric_status("semantic_recall", "fixture-model", "1", tmp_path) == "validated"
    assert metric_status("contradiction_rate", "fixture-model", "1", tmp_path) == "unvalidated"
    assert metric_status("semantic_recall", "other-model", "1", tmp_path) == "unvalidated"
    assert metric_status("semantic_recall", "fixture-model", "2", tmp_path) == "unvalidated"
    assert metric_status("semantic_recall", None, "1", tmp_path) == "unvalidated"


def test_no_committed_validation_result_means_unvalidated():
    # Nothing is validated until real human labels exist (docs/judge-validation.md).
    assert not list((REPO / "validation").glob("agreement-v*.json"))
    report = json.loads((REPO / "tests/fixtures/report_customer_support.json").read_text())
    for metric in ("semantic_recall", "contradiction_rate", "qa_accuracy"):
        assert report["scores"][metric] is not None
        assert report_metric_status(report, metric) == "unvalidated"
    md = render_markdown([report])
    for label in ("Semantic recall", "Contradiction rate", "QA accuracy"):
        row = next(line for line in md.splitlines() if line.startswith(f"| {label} |"))
        assert row.rstrip(" |").endswith("unvalidated")
    assert "Recall precision" in md and md.count("unvalidated") > 3
    assert "(unvalidated)" in render_markdown([report, report])
    assert "unvalidated" in render_html([report])


def test_validated_metric_is_not_flagged(tmp_path, monkeypatch):
    report = json.loads((REPO / "tests/fixtures/report_customer_support.json").read_text())
    _agreement_file(tmp_path, model=report["manifest"]["judge"]["model"])
    monkeypatch.setenv("MEMBENCH_VALIDATION_DIR", str(tmp_path))
    md = render_markdown([report])
    row = next(line for line in md.splitlines() if line.startswith("| Semantic recall |"))
    assert "unvalidated" not in row
    row = next(line for line in md.splitlines() if line.startswith("| Contradiction rate |"))
    assert "unvalidated" in row


# --- prompt-versioning policy ------------------------------------------------------------

# sha256 of every judge prompt template, per JUDGE_PROMPT_VERSION. If this test
# fails you changed a judge prompt: bump JUDGE_PROMPT_VERSION, add the new hash
# here, re-run `membench judge-cross` + `membench judge-agreement` and commit the
# new validation/agreement-v<N>.json (docs/judge-validation.md).
PROMPT_SHA256 = {
    "1": "a7899d8776749320fd4d8b0a229908294fbcc1eee677534bce83d4030e1e4025",
}


def _prompt_fingerprint():
    from membench.datasets.longmemeval import _JUDGE

    parts = [build_prompt("<RUBRIC>", "<EVIDENCE>"), qa_rubric("<QUESTION>", "<GOLD>"),
             *(f"{k}\n{v}" for k, v in sorted(_JUDGE.items()))]
    return hashlib.sha256("\n\x00\n".join(parts).encode()).hexdigest()


def test_judge_prompt_is_pinned_to_its_version():
    assert PROMPT_SHA256.get(JUDGE_PROMPT_VERSION) == _prompt_fingerprint(), (
        "judge prompt changed: bump JUDGE_PROMPT_VERSION and follow the prompt-versioning "
        "policy in docs/judge-validation.md"
    )
