"""Dataset loaders, download/cache, subsets, convert and rescore of dataset packs.

Uses tiny hand-made files shaped like LongMemEval / LoCoMo
(`tests/fixtures/datasets/`); no real dataset content is committed.
"""

import importlib.util
import json
from collections import Counter
from pathlib import Path

import jsonschema
import pytest
import yaml
from typer.testing import CliRunner

from membench import datasets
from membench.cli import app
from membench.datasets import download, locomo, longmemeval, resolve_pack
from membench.datasets.download import DatasetError, Source
from membench.packs.schema import QueryStep, WriteStep, load_pack

FIXTURES = Path(__file__).parent / "fixtures" / "datasets"
SCHEMA = json.loads((Path(datasets.__file__).parent.parent / "report.schema.json").read_text())
runner = CliRunner()


@pytest.fixture
def tiny(monkeypatch):
    """Point every dataset at the tiny fixtures; flag one LoCoMo item."""
    files = {
        "locomo": FIXTURES / "locomo_tiny.json",
        **{s.id: FIXTURES / "longmemeval_tiny.json" for s in longmemeval.SOURCES.values()},
    }
    monkeypatch.setattr(datasets, "fetch", lambda source: files[source.id])
    monkeypatch.setattr(locomo, "exclusions", lambda: {"conv-a/q4": "test reason"})


def _queries(pack):
    return [s for s in pack.steps if isinstance(s, QueryStep)]


# --- download -----------------------------------------------------------------

def _source(data: bytes, url="https://example.test/rev1/data.json") -> Source:
    import hashlib

    return Source("toy", url, "rev1", hashlib.sha256(data).hexdigest(), "MIT")


def test_fetch_downloads_once_checks_sha_and_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMBENCH_CACHE_DIR", str(tmp_path))
    calls = []

    def fake_download(url, dest):
        calls.append(url)
        dest.write_bytes(b"[1, 2]")

    monkeypatch.setattr(download, "_download", fake_download)
    source = _source(b"[1, 2]")
    path = download.fetch(source)
    assert path == tmp_path / "datasets" / source.sha256 / "data.json"
    assert download.fetch(source) == path
    assert len(calls) == 1  # second load is a verified cache hit

    path.write_bytes(b"tampered")
    download.fetch(source)
    assert len(calls) == 2 and path.read_bytes() == b"[1, 2]"


def test_fetch_rejects_a_sha_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMBENCH_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(download, "_download", lambda url, dest: dest.write_bytes(b"other"))
    source = _source(b"expected")
    with pytest.raises(DatasetError, match="does not match the pinned"):
        download.fetch(source)
    assert not list((tmp_path / "datasets").rglob("*.json*"))


def test_sources_are_pinned_to_a_revision_and_sha256():
    for source in [*longmemeval.SOURCES.values(), locomo.SOURCE]:
        assert source.revision in source.url
        assert len(source.sha256) == 64 and len(source.revision) == 40


# --- LongMemEval ----------------------------------------------------------------

def test_longmemeval_one_episode_per_question(tiny):
    pack = resolve_pack("longmemeval-s")
    assert pack.name == "longmemeval_s" and pack.qa_style == "longmemeval"
    assert pack.dataset["id"] == "longmemeval-s"
    assert pack.dataset["sha256"] == longmemeval.SOURCES["s"].sha256
    assert [s.episode for s in _queries(pack)] == [0, 1, 2]

    writes = [s for s in pack.steps if isinstance(s, WriteStep) and s.episode == 0]
    # sessions are sorted by date: the 03/01 session comes first
    assert writes[0].content == "Any tips for a rainy commute?"
    assert writes[0].timestamp == "2024-03-01T12:00:00+00:00"
    assert writes[3].timestamp == "2024-03-05T18:30:01+00:00"
    first, update, abstain = _queries(pack)
    assert first.timestamp == "2024-03-10T09:00:00+00:00"
    assert (first.category, first.answer, first.abstain) == ("single-session-user", "Green", False)
    assert first.evidence == [writes[2].id]  # the has_answer turn
    assert update.answer == "4" and len(update.evidence) == 2
    assert abstain.abstain and abstain.evidence == []
    assert abstain.answer.startswith("The information provided is not enough")


def test_longmemeval_hides_ground_truth_from_adapters(tiny):
    pack = resolve_pack("longmemeval-oracle")
    for step in pack.steps:
        assert "answer" not in step.session_id
        if isinstance(step, WriteStep):
            assert "answer" not in step.id and step.metadata == {}


def test_longmemeval_official_prompts():
    prompt = longmemeval.judge_prompt("temporal-reasoning", "Q?", "A", "R", False)
    assert "do not penalize off-by-one errors" in prompt
    assert prompt.endswith("Model Response: R\n\nIs the model response correct? Answer yes or no only.")
    assert "Rubric: A" in longmemeval.judge_prompt("single-session-preference", "Q", "A", "R", False)
    assert "unanswerable" in longmemeval.judge_prompt("multi-session", "Q", "A", "R", True)
    assert longmemeval.judge_prompt("multi-session", "Q", "A", "R", False) == (
        longmemeval.judge_prompt("single-session-user", "Q", "A", "R", False)
    )
    reader = longmemeval.reader_prompt("Q?", "CTX", "2024-03-10T09:00:00+00:00")
    assert "History Chats:\n\nCTX\n\nCurrent Date: 2024/03/10 (Sun) 09:00\nQuestion: Q?" in reader


# --- LoCoMo ---------------------------------------------------------------------

def test_locomo_one_episode_per_conversation(tiny):
    pack = resolve_pack("locomo")
    assert pack.name == "locomo" and pack.qa_style == "membench"
    writes = [s for s in pack.steps if isinstance(s, WriteStep)]
    assert writes[0].id == "conv-a/D1:1" and writes[0].role == "Ana"
    assert writes[0].timestamp == "2023-05-08T13:56:00+00:00"
    assert writes[1].content.endswith("[shares an image: a photo of a red kayak]")
    queries = _queries(pack)
    assert [q.category for q in queries] == [
        "single_hop", "multi_hop", "temporal", "adversarial", "open_domain", "single_hop",
    ]
    assert queries[1].evidence == ["conv-a/D1:1", "conv-a/D2:1"]  # split on ';'
    assert queries[2].evidence == ["conv-a/D2:2"] and queries[2].answer == "2023"  # repaired id
    assert queries[3].abstain and queries[3].answer is None
    assert queries[4].evidence == []  # unknown id dropped
    assert queries[0].timestamp == "2023-06-21T10:00:01+00:00"  # last turn + 1 day
    assert {s.episode for s in pack.steps} == {0, 1}
    assert pack.flagged == [pack.steps.index(queries[4])]


def test_locomo_exclusion_list_has_reasons_and_valid_ids():
    entries = yaml.safe_load(locomo.EXCLUSIONS_PATH.read_text())
    assert entries and all(e["reason"] for e in entries)
    assert all(e["id"].split("/")[1].startswith("q") for e in entries)
    assert len(locomo.exclusions()) == len(entries)


def test_exclude_flagged_drops_flagged_queries(tiny):
    pack = resolve_pack("locomo", exclude_flagged=True)
    assert len(_queries(pack)) == 5 and pack.flagged == []
    assert pack.dataset["options"] == {"exclude_flagged": True}
    assert pack.sha256 != resolve_pack("locomo").sha256


# --- subsets --------------------------------------------------------------------

def test_limit_keeps_first_queries_and_drops_empty_episodes(tiny):
    pack = resolve_pack("longmemeval-s", limit=2)
    assert len(_queries(pack)) == 2 and {s.episode for s in pack.steps} == {0, 1}
    assert pack.subset == {"limit": 2}


def test_sample_is_stratified_and_seeded():
    full = resolve_pack("tool-call-memory")
    pack = resolve_pack("tool-call-memory", sample=60, sample_seed=3)
    counts = Counter(q.category for q in _queries(pack))
    assert sum(counts.values()) == 60 and set(counts.values()) == {10}  # 6 equal categories
    again = resolve_pack("tool-call-memory", sample=60, sample_seed=3)
    other = resolve_pack("tool-call-memory", sample=60, sample_seed=4)
    assert pack.sha256 == again.sha256 != other.sha256 != full.sha256
    assert pack.subset == {"sample": 60, "sample_seed": 3}


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"limit": 1, "sample": 1}, "not both"),
    ({"pack_seed": 2}, "synthetic packs only"),
    ({"exclude_flagged": True}, "no exclusion list"),
])
def test_resolve_pack_rejects_bad_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        resolve_pack("mini", **kwargs)


# --- end to end -----------------------------------------------------------------

BACKENDS = ["full-context"] + (["chroma"] if importlib.util.find_spec("chromadb") else [])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("pack", ["longmemeval-s", "locomo"])
def test_dataset_runs_end_to_end(tiny, tmp_path, backend, pack):
    out = tmp_path / "report.json"
    result = runner.invoke(app, [
        "run", "--backend", backend, "--pack", pack, "--no-judge", "--out", str(out),
        "--persist-dir", str(tmp_path / "chroma"),
    ])
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text())
    jsonschema.validate(report, SCHEMA)
    assert report["manifest"]["dataset"]["id"] == pack
    assert report["scores"]["error_rate"] == 0.0
    assert report["scores"]["evidence_recall_at_k"] is not None

    check = runner.invoke(app, ["rescore", "--in", str(out), "--check"])
    assert check.exit_code == 0, check.output


def test_rescore_rebuilds_a_sampled_pack_and_scores_without_flagged(tiny, tmp_path):
    out = tmp_path / "report.json"
    result = runner.invoke(app, [
        "run", "--backend", "full-context", "--pack", "locomo", "--no-judge",
        "--sample", "4", "--seed", "1", "--out", str(out),
    ])
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text())["manifest"]["pack"]["subset"] == {
        "sample": 4, "sample_seed": 1,
    }
    assert runner.invoke(app, ["rescore", "--in", str(out), "--check"]).exit_code == 0

    full = tmp_path / "full.json"
    runner.invoke(app, ["run", "--backend", "full-context", "--pack", "locomo", "--no-judge",
                        "--out", str(full)])
    clean = tmp_path / "clean.json"
    result = runner.invoke(app, ["rescore", "--in", str(full), "--exclude-flagged",
                                 "--out", str(clean)])
    assert result.exit_code == 0, result.output
    data = json.loads(clean.read_text())
    assert sum(t["step_type"] == "query" for t in data["traces"]) == 5
    assert data["manifest"]["dataset"]["options"]["exclude_flagged"] is True
    rerun = runner.invoke(app, ["rescore", "--in", str(clean), "--check"])
    assert rerun.exit_code == 0, rerun.output  # the written report rescores identically


def test_convert_writes_a_loadable_yaml(tiny, tmp_path):
    out = tmp_path / "locomo.yaml"
    result = runner.invoke(app, ["convert", "--pack", "locomo", "--out", str(out)])
    assert result.exit_code == 0, result.output
    pack = load_pack(out)
    assert len(_queries(pack)) == 6 and pack.steps[0].id == "conv-a/D1:1"
