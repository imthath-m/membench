import json
import shutil
from pathlib import Path

from typer.testing import CliRunner

from membench.cli import app
from membench.core.report.failures import classify
from membench.core.report.leaderboard import collect, render_leaderboard
from membench.packs.schema import load_pack

ROOT = Path(__file__).parent.parent
FIXTURES = Path(__file__).parent / "fixtures"
REPORT = FIXTURES / "report_customer_support.json"
CACHE = FIXTURES / "judge_cache_customer_support.json"

runner = CliRunner()


def _invoke(*args):
    result = runner.invoke(app, [str(a) for a in args])
    assert result.exit_code == 0, result.output
    return result


def _run(tmp_path, backend, pack="customer_support"):
    out = tmp_path / f"{backend}-{pack}.json"
    _invoke("run", "--backend", backend, "--pack", pack, "--no-judge", "--out", out)
    return out


def _publish(report, results, *extra):
    return _invoke("publish", "--in", report, "--results", results, *extra)


def test_publish_writes_layout_and_rescores_identically(tmp_path):
    results = tmp_path / "results"
    _publish(_run(tmp_path, "bm25"), results)

    [directory] = list(results.glob("*/*/*"))
    assert directory.parts[-2:] == ("customer_support", "bm25")
    assert sorted(p.name for p in directory.iterdir()) == [
        "judge_cache.json", "meta.json", "report.json",
    ]
    meta = json.loads((directory / "meta.json").read_text())
    assert meta["config"] == "default"
    assert meta["uv_lock_sha256"] and meta["trimmed"] == ["traces[].detail.items[].text"]
    assert json.loads((directory / "judge_cache.json").read_text()) == {}
    report = json.loads((directory / "report.json").read_text())
    items = [i for t in report["traces"] for i in t["detail"].get("items", [])]
    assert items and all("text" not in i and "event_ids" in i for i in items)

    _invoke("rescore", "--in", directory / "report.json",
            "--judge-cache", directory / "judge_cache.json", "--check")


def test_publish_keeps_only_the_judge_cache_entries_the_report_uses(tmp_path):
    cache = tmp_path / "cache.json"
    entries = json.loads(CACHE.read_text())
    cache.write_text(json.dumps({**entries, "unrelated:v1:abc": {"passed": True}}))
    _publish(REPORT, tmp_path / "results", "--judge-cache", cache)

    [directory] = list((tmp_path / "results").glob("*/*/*"))
    assert json.loads((directory / "judge_cache.json").read_text()) == entries
    _invoke("rescore", "--in", directory / "report.json",
            "--judge-cache", directory / "judge_cache.json", "--check")


def test_publish_refuses_a_report_that_does_not_rescore(tmp_path):
    report = json.loads(REPORT.read_text())
    report["scores"]["recall_precision"] = 0.5
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(report))

    result = runner.invoke(app, ["publish", "--in", str(path), "--results",
                                 str(tmp_path / "results"), "--judge-cache", str(CACHE)])

    assert result.exit_code == 1
    assert "does not re-score identically" in result.output
    assert not (tmp_path / "results").exists()


def test_leaderboard_groups_by_judge_and_labels_rows(tmp_path):
    results = tmp_path / "results"
    _publish(REPORT, results, "--judge-cache", CACHE)  # full-context, judge + reader
    _publish(_run(tmp_path, "bm25"), results)
    _publish(_run(tmp_path, "oracle"), results)
    _publish(_run(tmp_path, "full-context"), results, "--config", "vendor-tuned")

    text = render_leaderboard(collect(results), results)

    sections = text.split("\n## ")[1:]
    judged = [s for s in sections if "Judge: claude-sonnet-5 (prompt v1)" in s]
    unjudged = [s for s in sections if "Judge: none" in s]
    assert len(judged) == 1 and len(unjudged) == 1
    assert "Semantic recall (unvalidated)" in judged[0]
    assert "| full-context |" in judged[0] and "bm25" not in judged[0]
    assert "Semantic recall" not in unjudged[0]  # judge metrics absent, not zero
    assert "| bm25 |" in unjudged[0]
    assert "| full-context (vendor-tuned) |" in unjudged[0]
    assert "| oracle *(upper bound)* |" in unjudged[0].splitlines()[-1]  # oracle last
    assert "n=" in unjudged[0] and "/customer_support/bm25/report.json)" in unjudged[0]
    assert render_leaderboard(collect(results), results) == text  # deterministic


def test_leaderboard_cli_and_committed_readme_is_up_to_date(tmp_path):
    shutil.copytree(ROOT / "results", tmp_path / "results")
    out = tmp_path / "results" / "README.md"
    out.unlink()

    _invoke("leaderboard", tmp_path / "results", "--out", out)

    assert out.read_text() == (ROOT / "results" / "README.md").read_text()


def test_failures_listing_classes_and_ids_only(tmp_path):
    report_path = _run(tmp_path, "bm25", "tool-call-memory")
    out = tmp_path / "failures.md"

    _invoke("report", "--in", report_path, "--failures", "--out", out)

    text = out.read_text()
    assert "# Failures: bm25 on tool_call_memory" in text
    assert "## Not retrieved" in text and "## Stale surfaced" in text
    assert "evidence ids: " in text and "context: `" in text

    _invoke("report", "--in", report_path, "--failures", "--excerpt-chars", "0", "--out", out)
    ids_only = out.read_text()
    assert "context: `" not in ids_only and "?" not in ids_only  # no query text either


def test_failure_classes_follow_outcomes():
    report = json.loads(REPORT.read_text())
    pack = load_pack(ROOT / "membench" / "packs" / "customer_support.yaml")
    query = next(t["step_index"] for t in report["traces"] if t["step_type"] == "query")
    report["outcomes"]["recall_precision"][str(query)] = False
    report["outcomes"]["staleness"][str(query)] = True

    classes = classify(report, pack)

    assert query in classes["not_retrieved"] and query in classes["stale_surfaced"]
    assert query not in classes["answered_wrong"]


def test_report_failures_needs_one_markdown_report(tmp_path):
    result = runner.invoke(app, ["report", "--in", str(REPORT), "--in", str(REPORT),
                                 "--failures", "--out", str(tmp_path / "f.md")])
    assert result.exit_code == 1
