"""Phase 13: --repeats, rescore of CIs, and `membench compare`."""

import json
from pathlib import Path

import jsonschema
import pytest
from typer.testing import CliRunner

import membench.cli as cli_module
from membench.adapters.full_context import FullContextAdapter
from membench.cli import app
from membench.core.report.compare import compare_reports, render_compare_markdown
from membench.core.report.markdown import render_markdown

runner = CliRunner()
SCHEMA = json.loads(
    (Path(__file__).parent.parent / "membench" / "report.schema.json").read_text()
)
_instances: list = []


class LLMBackedAdapter(FullContextAdapter):
    """Full-context stand-in for a backend that calls an LLM."""

    uses_llm = True

    def __init__(self, config=None):
        super().__init__(config)
        self.resets = 0
        _instances.append(self)

    def reset(self):
        self.resets += 1
        super().reset()


def _run(tmp_path, *args, name="report.json"):
    out = tmp_path / name
    result = runner.invoke(app, ["run", "--pack", "mini", "--no-judge", "--out", str(out), *args])
    assert result.exit_code == 0, result.output
    return json.loads(out.read_text())


def test_deterministic_backend_defaults_to_one_repeat(tmp_path):
    data = _run(tmp_path, "--backend", "full-context")
    assert data["repeats"] == [] and data["repeat_summary"] is None
    jsonschema.validate(data, SCHEMA)


def test_llm_backend_defaults_to_three_repeats_on_fresh_namespaces(monkeypatch, tmp_path):
    _instances.clear()
    monkeypatch.setattr(cli_module, "get_adapter_class", lambda name: LLMBackedAdapter)
    data = _run(tmp_path, "--backend", "stub")

    assert [a.namespace for a in _instances] == ["membench-r0", "membench-r1", "membench-r2"]
    assert all(a.resets >= 1 for a in _instances)
    assert [r["manifest"]["repeat"] for r in [data, *data["repeats"]]] == [0, 1, 2]
    summary = data["repeat_summary"]
    assert summary["n"] == 3
    rp = summary["scores"]["recall_precision"]
    assert rp["values"] == [data["scores"]["recall_precision"]] * 3
    assert rp["mean"] == data["scores"]["recall_precision"] and rp["sd"] == 0.0
    jsonschema.validate(data, SCHEMA)


def test_two_runs_of_the_same_config_have_overlapping_cis(tmp_path):
    data = _run(tmp_path, "--backend", "full-context", "--repeats", "2")
    first, second = data["stats"]["scores"], data["repeats"][0]["stats"]["scores"]
    for name, s in first.items():
        if s["ci"] is not None:
            (lo1, hi1), (lo2, hi2) = s["ci"], second[name]["ci"]
            assert lo1 <= hi2 and lo2 <= hi1, name


def test_rescore_reproduces_cis_of_every_repeat(tmp_path):
    _run(tmp_path, "--backend", "full-context", "--repeats", "2")
    path = tmp_path / "report.json"
    ok = runner.invoke(app, ["rescore", "--in", str(path), "--check"])
    assert ok.exit_code == 0, ok.output
    assert "identical" in ok.output

    data = json.loads(path.read_text())
    data["repeats"][0]["stats"]["scores"]["recall_precision"]["ci"] = [0.0, 1.0]
    path.write_text(json.dumps(data))
    bad = runner.invoke(app, ["rescore", "--in", str(path), "--check"])
    assert bad.exit_code == 1
    assert "repeat 1: stats" in bad.output

    out = tmp_path / "rescored.json"
    assert runner.invoke(app, ["rescore", "--in", str(path), "--out", str(out)]).exit_code == 0
    fixed = json.loads(out.read_text())
    assert fixed["repeats"][0]["stats"] == fixed["stats"]
    assert fixed["repeat_summary"]["n"] == 2
    jsonschema.validate(fixed, SCHEMA)


def test_markdown_shows_repeat_mean_and_sd(tmp_path):
    data = _run(tmp_path, "--backend", "full-context", "--repeats", "2")
    md = render_markdown([data])
    assert "**Repeats:** 2" in md
    assert "Mean ± SD (2 repeats)" in md
    assert "100% ± 0.0pp" in md
    assert "indicative (n < 385)" in md


# --- paired comparison ------------------------------------------------------

def _report(backend, passes, categories=None, judge=None, sha="abc"):
    n = len(passes)
    categories = categories or ["c"] * n
    return {
        "backend_name": backend,
        "manifest": {"pack": {"name": "p", "sha256": sha}, "seed": 0, "judge": judge,
                     "reader": None},
        "traces": [
            {"step_index": i, "step_type": "query", "detail": {"category": categories[i]}}
            for i in range(n)
        ],
        "outcomes": {
            "recall_precision": {str(i): p for i, p in enumerate(passes)},
            "evidence_recall_at_k": {str(i): float(p) for i, p in enumerate(passes)},
        },
    }


def test_compare_flags_real_and_null_differences():
    a = _report("a", [True] * 200 + [False] * 200)
    b = _report("b", [True] * 400)
    same = _report("same", [True] * 200 + [False] * 200)

    rows = {(r["metric"], r["category"]): r for r in compare_reports(a, b)["rows"]}
    rp = rows[("recall_precision", "all")]
    assert rp["n"] == 400 and rp["diff"] == pytest.approx(0.5)
    assert rp["test"] == "mcnemar" and rp["significant"] and not rp["ci_includes_zero"]
    assert not rp["indicative"]
    assert rows[("evidence_recall_at_k", "all")]["test"] == "bootstrap"
    assert ("recall_precision", "c") in rows  # per-category rows too
    assert all(r["p_holm"] >= r["p"] for r in rows.values())

    null = compare_reports(a, same)["rows"]
    assert all(r["ci_includes_zero"] and not r["significant"] for r in null)
    md = render_compare_markdown(compare_reports(a, same))
    assert "CI includes 0" in md and "Holm" in md


def test_compare_marks_small_samples_indicative():
    rows = compare_reports(_report("a", [True] * 10), _report("b", [False] * 10))["rows"]
    assert all(r["indicative"] for r in rows)
    assert "(indicative)" in render_compare_markdown(
        compare_reports(_report("a", [True] * 10), _report("b", [False] * 10))
    )


def test_compare_refuses_incomparable_runs():
    a = _report("a", [True] * 5)
    with pytest.raises(ValueError, match="different packs"):
        compare_reports(a, _report("b", [True] * 5, sha="other"))
    judge_a = {"model": "m1", "prompt_version": "1"}
    judge_b = {"model": "m2", "prompt_version": "1"}
    with pytest.raises(ValueError, match="judge"):
        compare_reports(_report("a", [True], judge=judge_a), _report("b", [True], judge=judge_b))
    old = _report("b", [True] * 5)
    del old["outcomes"]
    with pytest.raises(ValueError, match="rescore"):
        compare_reports(a, old)


def test_compare_cli_on_two_real_runs(tmp_path):
    _run(tmp_path, "--backend", "full-context", name="a.json")
    _run(tmp_path, "--backend", "full-context", name="b.json")
    md = runner.invoke(app, ["compare", "--a", str(tmp_path / "a.json"),
                             "--b", str(tmp_path / "b.json")])
    assert md.exit_code == 0, md.output
    assert "MemBench paired comparison: mini" in md.output

    out = tmp_path / "cmp.json"
    result = runner.invoke(app, ["compare", "--a", str(tmp_path / "a.json"),
                                 "--b", str(tmp_path / "b.json"), "--out", str(out)])
    assert result.exit_code == 0, result.output
    rows = json.loads(out.read_text())["rows"]
    assert rows and all(r["diff"] == 0 and r["ci_includes_zero"] for r in rows)

    other = runner.invoke(app, ["run", "--backend", "full-context", "--pack", "customer-support",
                                "--no-judge", "--out", str(tmp_path / "c.json")])
    assert other.exit_code == 0
    refused = runner.invoke(app, ["compare", "--a", str(tmp_path / "a.json"),
                                  "--b", str(tmp_path / "c.json")])
    assert refused.exit_code == 1
    assert "different packs" in refused.output
