import json

from typer.testing import CliRunner

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
