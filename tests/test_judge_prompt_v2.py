"""Candidate judge prompt v2: registered, not the default, pinned.

v2 is an UNTESTED hypothesis (never run against a model). Every label and item here
is SYNTHETIC test data (`synthetic-*`).
"""

import hashlib
import json

from typer.testing import CliRunner

from membench.cli import app
from membench.core.scoring import judge
from membench.core.scoring.judge import (
    JUDGE_PROMPT_VERSION,
    PROMPT_BUILDERS,
    JudgeCache,
    build_prompt,
    parse_verdict,
)
from membench.core.validation.tune import prompt_sha256
from tests.test_judge_tune import MODEL, _args, _tiny, _write

# Pin of the candidate text. If this fails you edited v2: once any v2 result exists,
# add a new version (v3) instead of editing v2 (docs/prompt-tuning.md).
V2_PROMPT_SHA = "73b89ceced20861de5d9766152d4eda62faabd44b6cf4decfbd8451db918f7e7"

RUBRIC = "synthetic-rubric: The evidence indicates the plan is Pro."
EVIDENCE = "synthetic-evidence: user moved to the Pro plan on 2026-01-02."


def test_v2_is_registered_and_not_the_default():
    assert "2" in PROMPT_BUILDERS
    assert JUDGE_PROMPT_VERSION == "1"
    assert build_prompt(RUBRIC, EVIDENCE) == build_prompt(RUBRIC, EVIDENCE, "1")


def test_v2_prompt_is_pinned():
    assert prompt_sha256("2") == V2_PROMPT_SHA


def test_v2_prompt_contains_inputs_verbatim_and_differs_from_v1():
    v1, v2 = build_prompt(RUBRIC, EVIDENCE, "1"), build_prompt(RUBRIC, EVIDENCE, "2")
    assert v1 != v2
    assert f"STATEMENT:\n{RUBRIC}\n\nEVIDENCE:\n{EVIDENCE}\n" in v2
    assert v2.endswith(f"EVIDENCE:\n{EVIDENCE}\n")


def test_v2_cache_key_differs_from_v1():
    k1 = JudgeCache.key(MODEL, build_prompt(RUBRIC, EVIDENCE, "1"), "1")
    k2 = JudgeCache.key(MODEL, build_prompt(RUBRIC, EVIDENCE, "2"), "2")
    assert k1 != k2 and ":v2:" in k2


def test_v2_style_reply_parses():
    reply = '{"reason": "\\"moved to the Pro plan\\"", "verdict": "pass"}'
    v = parse_verdict(reply)
    assert v.passed and "Pro plan" in v.reason
    assert not parse_verdict('{"reason": "no span", "verdict": "fail"}').passed


def test_cli_accepts_versions_1_and_2_strict_miss(tmp_path, monkeypatch):
    monkeypatch.delenv(judge.API_KEY_ENV, raising=False)
    labels, sample = _tiny()
    _write(tmp_path, labels, sample)
    result = CliRunner().invoke(app, _args(tmp_path, "1", "2") + ["--strict"])
    assert result.exit_code == 1
    assert "unknown prompt version" not in result.output
    assert "lacks cached verdicts" in result.output and judge.API_KEY_ENV in result.output


def test_cli_versions_1_and_2_from_a_prefilled_cache(tmp_path, monkeypatch):
    monkeypatch.delenv(judge.API_KEY_ENV, raising=False)
    labels, sample = _tiny()
    _write(tmp_path, labels, sample)
    cache = JudgeCache(tmp_path / "cache.json")
    for item in sample:
        for v in ("1", "2"):
            prompt = build_prompt(item["rubric"], item["evidence"], v)
            cache.set(JudgeCache.key(MODEL, prompt, v),
                      {"passed": True, "reason": "synthetic", "raw": "", "usage": {}})
    result = CliRunner().invoke(app, _args(tmp_path, "1", "2") + ["--strict"])
    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "tuning.json").read_text())
    assert report["versions"] == ["1", "2"] and report["candidate"] == "2"
    assert report["prompt_sha256"]["2"] == V2_PROMPT_SHA
    expected = hashlib.sha256(build_prompt("{rubric}", "{evidence}", "1").encode()).hexdigest()
    assert report["prompt_sha256"]["1"] == expected
