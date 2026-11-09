import json

from typer.testing import CliRunner

import membench.cli as cli_module
from membench.adapters.full_context import FullContextAdapter
from membench.cli import app

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
