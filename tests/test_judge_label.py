import json
import threading
import urllib.error
import urllib.request

import pytest
from typer.testing import CliRunner

from membench.cli import app
from membench.core.validation.label_server import LabelStore, make_server
from membench.core.validation.labels import LabelsError, merge_annotations, read_jsonl

BLINDED = ("verdict", "judge_model", "prompt_version", "backend", "pack", "length_bucket",
           "tokens")


def _sample(n: int = 6) -> list[dict]:
    return [{
        "id": f"synthetic-{k}", "metric": "semantic_recall" if k % 2 else "contradiction_rate",
        "backend": f"synthetic-backend-{k % 3}", "pack": "synthetic-pack", "step_index": k,
        "category": "update", "rubric": f"synthetic statement {k}",
        "evidence": f"synthetic evidence {k}", "tokens": 10, "length_bucket": "<128",
        "prompt_version": "1", "judge_model": "synthetic-judge",
        "verdict": {"passed": True, "score": 1.0},
    } for k in range(n)]


def test_state_is_blinded(tmp_path):
    store = LabelStore(_sample(), "synthetic-a", tmp_path / "a.jsonl")
    state = json.dumps(store.state())
    for item in store.state()["items"]:
        assert not set(item) & set(BLINDED)
        assert {"id", "metric", "rubric", "evidence"} <= set(item)
    assert "synthetic-judge" not in state and "synthetic-backend" not in state
    assert "synthetic-pack" not in state


def test_labels_written_and_merge(tmp_path):
    sample = _sample()
    a = LabelStore(sample, "synthetic-a", tmp_path / "a.jsonl")
    b = LabelStore(sample, "synthetic-b", tmp_path / "b.jsonl")
    for item in sample:
        a.set(item["id"], True, "synthetic note")
        b.set(item["id"], False)
    rows = read_jsonl(tmp_path / "a.jsonl")
    assert rows[0] == {"item_id": rows[0]["item_id"], "label": True, "note": "synthetic note"}
    assert all(r["label"] is True for r in rows) and len(rows) == len(sample)
    assert "note" not in read_jsonl(tmp_path / "b.jsonl")[0]
    lines = merge_annotations(sample, {"synthetic-a": rows,
                                       "synthetic-b": read_jsonl(tmp_path / "b.jsonl")})
    assert len(lines) == len(sample)
    assert lines[0]["labels"] == {"synthetic-a": True, "synthetic-b": False}


def test_resume_and_clear(tmp_path):
    sample, out = _sample(), tmp_path / "a.jsonl"
    store = LabelStore(sample, "synthetic-a", out)
    first = store.items[0]["id"]
    store.set(first, False)
    resumed = LabelStore(sample, "synthetic-a", out)
    assert resumed.labels[first]["label"] is False
    assert resumed.state()["start"] == 1
    resumed.clear(first)
    assert read_jsonl(out) == []
    assert LabelStore(sample, "synthetic-a", out).state()["start"] == 0


def test_bad_inputs_rejected(tmp_path):
    store = LabelStore(_sample(), "synthetic-a", tmp_path / "a.jsonl")
    with pytest.raises(LabelsError):
        store.set("synthetic-missing", True)
    with pytest.raises(LabelsError):
        store.set("synthetic-0", "yes")
    with pytest.raises(LabelsError):
        store.set("synthetic-0", 1)
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"item_id": "synthetic-0", "label": "true"}) + "\n")
    with pytest.raises(LabelsError, match="true or false"):
        LabelStore(_sample(), "synthetic-a", bad)
    bad.write_text(json.dumps({"item_id": "synthetic-zzz", "label": True}) + "\n")
    with pytest.raises(LabelsError, match="not in the sample"):
        LabelStore(_sample(), "synthetic-a", bad)


def test_order_deterministic_and_filter(tmp_path):
    sample = _sample(20)
    ids = lambda s: [i["id"] for i in s.items]
    a1 = LabelStore(sample, "synthetic-a", tmp_path / "1.jsonl", seed=0)
    a2 = LabelStore(list(reversed(sample)), "synthetic-a", tmp_path / "2.jsonl", seed=0)
    b = LabelStore(sample, "synthetic-b", tmp_path / "3.jsonl", seed=0)
    c = LabelStore(sample, "synthetic-a", tmp_path / "4.jsonl", seed=1)
    assert ids(a1) == ids(a2)
    assert ids(a1) != ids(b) and ids(a1) != ids(c)
    only = LabelStore(sample, "synthetic-a", tmp_path / "5.jsonl", metrics=["semantic_recall"])
    assert {i["metric"] for i in only.items} == {"semantic_recall"} and len(only.items) == 10


def test_http_round_trip(tmp_path):
    store = LabelStore(_sample(), "synthetic-a", tmp_path / "a.jsonl")
    server = make_server(store, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def post(body):
        req = urllib.request.Request(f"{base}/api/label", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        return urllib.request.urlopen(req)

    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "Download JSONL" in page
        state = json.loads(urllib.request.urlopen(base + "/api/state").read())
        assert "verdict" not in json.dumps(state)
        item_id = state["items"][0]["id"]
        assert post({"item_id": item_id, "label": True, "note": "synthetic"}).status == 200
        assert read_jsonl(tmp_path / "a.jsonl") == [
            {"item_id": item_id, "label": True, "note": "synthetic"}]
        export = urllib.request.urlopen(base + "/export.jsonl").read().decode()
        assert json.loads(export) == {"item_id": item_id, "label": True, "note": "synthetic"}
        for body in ({"item_id": "synthetic-nope", "label": True},
                     {"item_id": item_id, "label": "true"}):
            with pytest.raises(urllib.error.HTTPError) as err:
                post(body)
            assert err.value.code == 400
        post({"item_id": item_id, "label": None})
        assert read_jsonl(tmp_path / "a.jsonl") == []
    finally:
        server.shutdown()
        server.server_close()


def test_cli_missing_sample(tmp_path):
    result = CliRunner().invoke(app, [
        "judge-label", "--annotator", "synthetic-a", "--sample", str(tmp_path / "none.jsonl"),
        "--out", str(tmp_path / "a.jsonl"), "--no-open",
    ])
    assert result.exit_code == 1
    assert "Error" in result.output


def test_cli_bad_out_file(tmp_path):
    sample = tmp_path / "sample.jsonl"
    sample.write_text("".join(json.dumps(r) + "\n" for r in _sample()))
    out = tmp_path / "a.jsonl"
    out.write_text(json.dumps({"item_id": "synthetic-0", "label": 1}) + "\n")
    result = CliRunner().invoke(app, ["judge-label", "--annotator", "synthetic-a", "--sample",
                                      str(sample), "--out", str(out), "--no-open"])
    assert result.exit_code == 1
    assert "true or false" in result.output
