from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

import membench.cli as cli_module
from membench.adapters.full_context import FullContextAdapter
from membench.core.runner import run_pack
from membench.generate.generator import GenerationError, generate_pack
from membench.generate.transcript import load_transcripts
from membench.packs.schema import QueryStep, WriteStep, load_pack

SAMPLE_TRANSCRIPT = Path(__file__).parent.parent / "membench" / "generate" / "samples" / "sample_transcript.txt"

CANNED_PACK_YAML = """\
name: generated
description: Jamie's profile across two sessions.
steps:
  - type: write
    session_id: s1
    key: fact_name
    role: user
    content: "My name is Jamie Rivera."
    fact_type: state
  - type: write
    session_id: s1
    key: fact_language
    role: user
    content: "My favorite programming language is Python."
    fact_type: state
  - type: query
    session_id: s1
    query: "What is the user's favorite programming language?"
    expect_contains: ["Python"]
    expect_key: fact_language
  - type: write
    session_id: s2
    key: fact_name
    role: user
    content: "Actually I go by Jamie now, not James."
    fact_type: update
  - type: write
    session_id: s2
    key: fact_language
    role: user
    content: "I was wrong earlier: my favorite language is actually Go, not Python."
    fact_type: contradict
  - type: query
    session_id: s2
    query: "What is the user's favorite programming language now?"
    expect_contains: ["Go"]
    expect_key: fact_language
"""


class FakeClient:
    """Stand-in for AnthropicClient/OpenAICompatibleClient — no network."""

    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


# --- transcript loader -------------------------------------------------------


def test_load_transcripts_reads_file():
    text = load_transcripts(SAMPLE_TRANSCRIPT)
    assert "Jamie Rivera" in text
    assert "session s2" in text


def test_load_transcripts_reads_directory(tmp_path):
    (tmp_path / "a.txt").write_text("first file")
    (tmp_path / "b.txt").write_text("second file")

    text = load_transcripts(tmp_path)

    assert "first file" in text
    assert "second file" in text


def test_load_transcripts_empty_dir_raises(tmp_path):
    with pytest.raises(ValueError):
        load_transcripts(tmp_path)


# --- generator ---------------------------------------------------------------


def test_generate_pack_produces_schema_valid_pack_with_all_step_kinds():
    client = FakeClient(CANNED_PACK_YAML)

    raw = generate_pack(SAMPLE_TRANSCRIPT, client, pack_name="generated")

    assert client.prompts and "Jamie Rivera" in client.prompts[0]

    fact_types = {step["fact_type"] for step in raw["steps"] if step["type"] == "write"}
    assert fact_types == {"state", "update", "contradict"}
    assert any(step["type"] == "query" for step in raw["steps"])


def test_generate_pack_strips_markdown_code_fence():
    client = FakeClient(f"```yaml\n{CANNED_PACK_YAML}```")

    raw = generate_pack(SAMPLE_TRANSCRIPT, client)

    assert raw["name"] == "generated"


def test_generate_pack_invalid_yaml_raises():
    client = FakeClient("{not: valid: yaml: [")

    with pytest.raises(GenerationError):
        generate_pack(SAMPLE_TRANSCRIPT, client)


def test_generate_pack_schema_invalid_raises():
    client = FakeClient("name: bad\ndescription: no steps here\n")

    with pytest.raises(GenerationError):
        generate_pack(SAMPLE_TRANSCRIPT, client)


def test_generated_pack_loads_and_runs_end_to_end_offline(tmp_path):
    client = FakeClient(CANNED_PACK_YAML)
    raw = generate_pack(SAMPLE_TRANSCRIPT, client)

    out_path = tmp_path / "pack.yaml"
    out_path.write_text(yaml.safe_dump(raw, sort_keys=False))

    pack = load_pack(out_path)
    assert any(isinstance(s, WriteStep) and s.fact_type == "state" for s in pack.steps)
    assert any(isinstance(s, WriteStep) and s.fact_type == "update" for s in pack.steps)
    assert any(isinstance(s, WriteStep) and s.fact_type == "contradict" for s in pack.steps)
    assert any(isinstance(s, QueryStep) for s in pack.steps)

    result = run_pack(pack, FullContextAdapter(), backend_name="full-context", judge=None)

    assert result.pack_name == "generated"
    assert len(result.traces) == len(pack.steps)
    assert result.scores["recall_precision"] == 1.0
    assert result.scores["update_correctness"] == 1.0


# --- CLI ----------------------------------------------------------------------

runner = CliRunner()


def test_generate_cli_no_api_key_errors_cleanly(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_module, "resolve_client", lambda **kwargs: None)
    out_path = tmp_path / "pack.yaml"

    result = runner.invoke(
        cli_module.app,
        ["generate", "--transcripts", str(SAMPLE_TRANSCRIPT), "--out", str(out_path)],
    )

    assert result.exit_code == 1
    assert "MEMBENCH_JUDGE_API_KEY" in result.output
    assert not out_path.exists()


def test_generate_cli_writes_schema_valid_pack(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_module, "resolve_client", lambda **kwargs: FakeClient(CANNED_PACK_YAML))
    out_path = tmp_path / "pack.yaml"

    result = runner.invoke(
        cli_module.app,
        ["generate", "--transcripts", str(SAMPLE_TRANSCRIPT), "--out", str(out_path)],
    )

    assert result.exit_code == 0, result.output
    pack = load_pack(out_path)
    assert pack.name == "generated"
