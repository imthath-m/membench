import json
from pathlib import Path

import pytest

import membench.adapters.mem0 as mem0_module
from membench.adapters.full_context import FullContextAdapter
from membench.core.report.json import to_dict
from membench.core.runner import run_pack
from membench.packs.schema import builtin_pack_path, load_pack
from tests.test_mem0_adapter import MeteredFakeMem0Client
from tests.test_scoring_judge_metrics import StubJudge

jsonschema = pytest.importorskip("jsonschema")

SCHEMA = json.loads((Path(__file__).parent.parent / "membench" / "report.schema.json").read_text())
FIXTURE = Path(__file__).parent / "fixtures" / "report_customer_support.json"


def _validate(report: dict) -> None:
    jsonschema.Draft202012Validator.check_schema(SCHEMA)
    jsonschema.validate(json.loads(json.dumps(report)), SCHEMA)  # JSON round trip, like disk


def test_committed_report_validates():
    _validate(json.loads(FIXTURE.read_text()))


@pytest.mark.parametrize("pack", ["mini", "customer_support", "coding_assistant"])
def test_full_context_reports_validate(pack):
    result = run_pack(load_pack(builtin_pack_path(pack)), FullContextAdapter(), "full-context",
                      judge=StubJudge("a"))
    _validate(to_dict(result))


def test_report_with_errors_validates():
    class Broken(FullContextAdapter):
        def recall(self, query, **kwargs):
            raise RuntimeError("down")

    result = run_pack(load_pack(builtin_pack_path("mini")), Broken(), "broken")
    assert not result.valid
    _validate(to_dict(result))


def test_mem0_report_with_usage_and_cost_validates(monkeypatch):
    monkeypatch.setattr(mem0_module, "Memory", MeteredFakeMem0Client)
    result = run_pack(load_pack(builtin_pack_path("mini")), mem0_module.Mem0Adapter(), "mem0")

    assert result.usage["llm_input_tokens"] > 0
    assert result.cost["backend"]["usd"] > 0
    _validate(to_dict(result))


def test_schema_rejects_a_report_without_manifest():
    report = json.loads(FIXTURE.read_text())
    del report["manifest"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(report, SCHEMA)


def test_chroma_report_validates(tmp_path):
    pytest.importorskip("chromadb", reason="chromadb not installed")
    from membench.adapters.chroma import ChromaAdapter

    adapter = ChromaAdapter({"persist_dir": str(tmp_path / "chroma")})
    result = run_pack(load_pack(builtin_pack_path("mini")), adapter, "chroma")

    assert result.manifest["backend"]["models"]["embedding_model"] == "all-MiniLM-L6-v2"
    assert result.manifest["packages"]["chromadb"]
    _validate(to_dict(result))
