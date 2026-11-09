import json
import sys
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import membench.cli as cli_module
from membench.adapters.full_context import FullContextAdapter
from membench.cli import app
from membench.core.scoring.judge import CachedJudge, OpenAICompatibleClient

runner = CliRunner()


def test_run_full_context_writes_valid_json_report(tmp_path):
    out_path = tmp_path / "report.json"

    result = runner.invoke(
        app,
        ["run", "--backend", "full-context", "--pack", "mini", "--out", str(out_path)],
    )

    assert result.exit_code == 0, result.output
    data = json.loads(out_path.read_text())
    assert data["pack_name"] == "mini"
    assert data["backend_name"] == "full-context"
    assert len(data["traces"]) == 8


def test_run_unknown_backend_fails_cleanly(tmp_path):
    out_path = tmp_path / "report.json"

    result = runner.invoke(
        app,
        ["run", "--backend", "nonexistent", "--pack", "mini", "--out", str(out_path)],
    )

    assert result.exit_code == 1
    assert not out_path.exists()


_instances: list = []


class RecordingAdapter(FullContextAdapter):
    """Full-context adapter that records its config and reset() calls."""

    def __init__(self, config=None):
        super().__init__(config)
        self.reset_calls = 0
        _instances.append(self)

    def reset(self):
        self.reset_calls += 1


def _run_recording(monkeypatch, tmp_path, *extra_args):
    _instances.clear()
    monkeypatch.setattr(cli_module, "get_adapter_class", lambda name: RecordingAdapter)
    return runner.invoke(
        app,
        ["run", "--backend", "stub", "--pack", "mini", "--no-judge",
         "--out", str(tmp_path / "report.json"), *extra_args],
    )


def test_run_merges_backend_config_yaml(monkeypatch, tmp_path):
    config_path = tmp_path / "backend.yaml"
    config_path.write_text("user_id: eval-1\ntop_k: 5\nmem0_config:\n  version: v1.1\n")

    result = _run_recording(monkeypatch, tmp_path, "--backend-config", str(config_path))

    assert result.exit_code == 0, result.output
    assert _instances[0].config == {
        "persist_dir": ".membench_chroma",
        "user_id": "eval-1",
        "top_k": 5,
        "mem0_config": {"version": "v1.1"},
    }


def test_run_accepts_backend_config_json(monkeypatch, tmp_path):
    config_path = tmp_path / "backend.json"
    config_path.write_text(json.dumps({"user_id": "eval-2", "persist_dir": "x"}))

    result = _run_recording(monkeypatch, tmp_path, "--backend-config", str(config_path))

    assert result.exit_code == 0, result.output
    assert _instances[0].config == {"persist_dir": "x", "user_id": "eval-2"}


def test_run_rejects_non_mapping_backend_config(monkeypatch, tmp_path):
    config_path = tmp_path / "backend.yaml"
    config_path.write_text("- not\n- a mapping\n")

    result = _run_recording(monkeypatch, tmp_path, "--backend-config", str(config_path))

    assert result.exit_code == 1
    assert _instances == []


def test_run_resets_backend_by_default(monkeypatch, tmp_path):
    result = _run_recording(monkeypatch, tmp_path)

    assert result.exit_code == 0, result.output
    assert _instances[0].reset_calls == 1


def test_run_no_reset_skips_reset(monkeypatch, tmp_path):
    result = _run_recording(monkeypatch, tmp_path, "--no-reset")

    assert result.exit_code == 0, result.output
    assert _instances[0].reset_calls == 0


# --- Phase 8: hardening -----------------------------------------------------

def test_run_mem0_without_mem0ai_prints_one_line_error(monkeypatch, tmp_path):
    # Block `import mem0` and force the adapter module to re-import under the block.
    monkeypatch.setitem(sys.modules, "mem0", None)
    monkeypatch.delitem(sys.modules, "membench.adapters.mem0", raising=False)
    out_path = tmp_path / "report.json"

    result = runner.invoke(
        app, ["run", "--backend", "mem0", "--pack", "mini", "--no-judge", "--out", str(out_path)]
    )

    assert result.exit_code == 1
    lines = result.output.strip().splitlines()
    assert len(lines) == 1, result.output
    assert lines[0].startswith("Error:")
    assert "pip install membench[mem0]" in lines[0]
    assert "Traceback" not in result.output
    assert not out_path.exists()


class ExplodingAdapter(FullContextAdapter):
    """Raises on the second recall to simulate a backend failure mid-pack."""

    def __init__(self, config=None):
        super().__init__(config)
        self.recalls = 0

    def recall(self, query, **kwargs):
        self.recalls += 1
        if self.recalls == 2:
            raise RuntimeError("backend exploded")
        return super().recall(query, **kwargs)


def test_run_forced_adapter_exception_still_writes_report(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_module, "get_adapter_class", lambda name: ExplodingAdapter)
    out_path = tmp_path / "report.json"

    result = runner.invoke(
        app, ["run", "--backend", "stub", "--pack", "mini", "--no-judge", "--out", str(out_path)]
    )

    assert result.exit_code == 0, result.output
    data = json.loads(out_path.read_text())
    errors = [t["error"] for t in data["traces"] if t["error"]]
    assert errors == ["RuntimeError: backend exploded"]
    assert data["scores"]["error_rate"] == 1 / 8
    assert data["valid"] is False  # 12.5% is above the 10% threshold
    assert "run is invalid" in result.output


class RubricRoutedHTTP:
    """Fake judge endpoint: replies by which rubric the prompt carries."""

    def __init__(self, routes):
        self.routes = routes  # list of (substring, body); first match wins

    def post(self, url, headers=None, json=None):
        prompt = json["messages"][0]["content"]
        body = next(b for needle, b in self.routes if needle in prompt)
        return SimpleNamespace(status_code=200, headers={}, json=lambda: body,
                               raise_for_status=lambda: None)


def _run_with_judge(monkeypatch, tmp_path, http):
    client = OpenAICompatibleClient(
        api_key="k", model="stub", base_url="https://gateway.example/v1", http_client=http
    )
    monkeypatch.setattr(
        cli_module, "build_judge", lambda **kw: CachedJudge(client, tmp_path / "cache.json")
    )
    out_path = tmp_path / "report.json"
    result = runner.invoke(
        app, ["run", "--backend", "full-context", "--pack", "customer-support",
              "--out", str(out_path)],
    )
    assert result.exit_code == 0, result.output
    return json.loads(out_path.read_text())


def _openai(text):
    return {"choices": [{"message": {"content": text}}]}


def _judged(data):
    return [t for t in data["traces"] if t["step_type"] == "query" and "judge_error" in t["detail"]]


@pytest.mark.parametrize("bad_reply", ["no json at all", '{"verdict": pass'])
def test_run_malformed_judge_reply_records_judge_error(monkeypatch, tmp_path, bad_reply):
    http = RubricRoutedHTTP([
        ("correct order number is ORD-7788", _openai(bad_reply)),
        ("", _openai('{"verdict": "pass", "reason": "ok"}')),
    ])

    data = _run_with_judge(monkeypatch, tmp_path, http)

    [errored] = _judged(data)
    assert errored["detail"]["query"] == "What is the customer's correct order number?"
    assert set(errored["detail"]["judge_error"]) == {"semantic_recall"}
    assert data["scores"]["semantic_recall"] == 1.0  # the other judged query still scores
    assert data["scores"]["contradiction_rate"] == 1.0


def test_run_unexpected_judge_response_shape_records_judge_error(monkeypatch, tmp_path):
    http = RubricRoutedHTTP([
        ("moved from the Basic plan", {"error": "no choices here"}),
        ("", _openai('{"verdict": "fail", "reason": "no"}')),
    ])

    data = _run_with_judge(monkeypatch, tmp_path, http)

    [errored] = _judged(data)
    assert errored["detail"]["query"] == "Summarize the customer's plan history."
    assert "KeyError" in errored["detail"]["judge_error"]["semantic_recall"]
    assert data["scores"]["semantic_recall"] == 0.0
    assert data["scores"]["contradiction_rate"] == 0.0
