import json
import socket
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from membench.cli import app
from membench.packs.schema import builtin_pack_path

FIXTURES = Path(__file__).parent / "fixtures"
REPORT = FIXTURES / "report_customer_support.json"
CACHE = FIXTURES / "judge_cache_customer_support.json"

runner = CliRunner()


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("rescore must not touch the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.delenv("MEMBENCH_JUDGE_API_KEY", raising=False)


def _rescore(*args):
    return runner.invoke(app, ["rescore", *map(str, args)])


def test_rescore_of_committed_report_is_identical_offline(no_network, tmp_path):
    out = tmp_path / "rescored.json"

    result = _rescore("--in", REPORT, "--judge-cache", CACHE, "--out", out, "--check")

    assert result.exit_code == 0, result.output
    assert "identical" in result.output
    original = json.loads(REPORT.read_text())
    assert original["scores"]["semantic_recall"] is not None  # the judge was replayed
    assert json.loads(out.read_text())["scores"] == original["scores"]


def test_rescore_reports_changed_scores(no_network, tmp_path):
    report = json.loads(REPORT.read_text())
    report["scores"]["recall_precision"] = 0.5
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(report))

    result = _rescore("--in", tampered, "--judge-cache", CACHE, "--check")

    assert result.exit_code == 1
    assert "recall_precision 0.5 -> 1.0" in result.output


def test_rescore_judge_cache_miss_is_an_error(no_network, tmp_path):
    result = _rescore("--in", REPORT, "--judge-cache", tmp_path / "empty.json")

    assert result.exit_code == 1
    assert "no cached verdict" in result.output


def test_rescore_refuses_a_changed_pack(no_network, tmp_path):
    raw = yaml.safe_load(builtin_pack_path("customer_support").read_text())
    raw["description"] += " (edited)"
    pack = tmp_path / "customer_support.yaml"
    pack.write_text(yaml.safe_dump(raw))

    result = _rescore("--in", REPORT, "--pack", pack, "--judge-cache", CACHE)

    assert result.exit_code == 1
    assert "differs from the run's" in result.output


def test_rescore_refuses_a_report_without_manifest(tmp_path):
    report = json.loads(REPORT.read_text())
    del report["manifest"]
    old = tmp_path / "old.json"
    old.write_text(json.dumps(report))

    result = _rescore("--in", old)

    assert result.exit_code == 1
    assert "no run manifest" in result.output


def test_run_with_strict_cache_replays_without_a_key(no_network, tmp_path):
    out = tmp_path / "replay.json"

    result = runner.invoke(app, [
        "run", "--backend", "full-context", "--pack", "customer-support",
        "--strict-cache", "--judge-cache", str(CACHE), "--out", str(out),
    ])

    assert result.exit_code == 0, result.output
    replay = json.loads(out.read_text())
    assert replay["scores"]["semantic_recall"] == json.loads(REPORT.read_text())["scores"][
        "semantic_recall"
    ]
    assert replay["cost"]["judge"]["calls"] == 0
    assert replay["cost"]["judge"]["usd"] == 0.0
