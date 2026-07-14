from typer.testing import CliRunner

from membench.cli import app

runner = CliRunner()


def _run(pack, out_path):
    result = runner.invoke(
        app,
        ["run", "--backend", "full-context", "--pack", pack, "--no-judge", "--out", str(out_path)],
    )
    assert result.exit_code == 0, result.output
    return out_path


def test_report_md_from_single_run(tmp_path):
    report_json = _run("customer_support", tmp_path / "cs.json")
    out_md = tmp_path / "out.md"

    result = runner.invoke(
        app, ["report", "--in", str(report_json), "--format", "md", "--out", str(out_md)]
    )

    assert result.exit_code == 0, result.output
    text = out_md.read_text()
    assert "MemBench scorecard: customer_support" in text
    assert "Recall precision" in text


def test_report_html_comparison_is_self_contained(tmp_path):
    fc = _run("coding_assistant", tmp_path / "ca_fc.json")
    out_html = tmp_path / "out.html"

    result = runner.invoke(
        app,
        ["report", "--in", str(fc), "--in", str(fc), "--format", "html", "--out", str(out_html)],
    )

    assert result.exit_code == 0, result.output
    html = out_html.read_text()
    assert html.lstrip().startswith("<!doctype html>")
    assert "src=" not in html and "<script" not in html.lower()


def test_report_unknown_format_fails(tmp_path):
    report_json = _run("mini", tmp_path / "mini.json")
    out = tmp_path / "out.xyz"

    result = runner.invoke(
        app, ["report", "--in", str(report_json), "--format", "xyz", "--out", str(out)]
    )

    assert result.exit_code == 1
    assert not out.exists()


def test_list_shows_packs_and_backends():
    result = runner.invoke(app, ["list"])

    assert result.exit_code == 0, result.output
    assert "mini" in result.output
    assert "customer_support" in result.output
    assert "coding_assistant" in result.output
    assert "full-context" in result.output
    assert "chroma" in result.output
