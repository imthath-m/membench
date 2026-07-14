import json

from membench.adapters.full_context import FullContextAdapter
from membench.core.report.json import write_json
from membench.core.runner import run_pack
from membench.packs.schema import builtin_pack_path, load_pack


def test_write_json_produces_valid_report(tmp_path):
    pack = load_pack(builtin_pack_path("mini"))
    adapter = FullContextAdapter()
    result = run_pack(pack, adapter, backend_name="full-context")

    out_path = tmp_path / "report.json"
    write_json(result, out_path)

    data = json.loads(out_path.read_text())
    assert data["pack_name"] == "mini"
    assert data["backend_name"] == "full-context"
    assert len(data["traces"]) == len(pack.steps)
    assert data["scores"] == {}
    for trace in data["traces"]:
        assert "latency_ms" in trace
        assert "step_type" in trace
