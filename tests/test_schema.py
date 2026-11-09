from datetime import datetime

import pytest
import yaml

from membench.packs.schema import (
    ForgetStep,
    PackValidationError,
    QueryStep,
    WriteStep,
    builtin_pack_path,
    load_pack,
    parse_pack,
)


def test_load_mini_pack():
    pack = load_pack(builtin_pack_path("mini"))

    assert pack.name == "mini"
    assert len(pack.steps) == 8
    assert isinstance(pack.steps[0], WriteStep)
    assert isinstance(pack.steps[2], QueryStep)


def test_query_step_carries_assertion_fields():
    pack = load_pack(builtin_pack_path("mini"))
    query_step = pack.steps[2]

    assert query_step.expect_contains == ["blue"]
    assert query_step.expect_key == "fact_color"


def test_missing_steps_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\n")

    with pytest.raises(PackValidationError):
        load_pack(bad)


def test_unknown_step_type_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\nsteps:\n  - type: nonsense\n")

    with pytest.raises(PackValidationError):
        load_pack(bad)


def test_write_step_missing_field_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "name: bad\nsteps:\n  - type: write\n    session_id: s1\n    key: k1\n"
    )

    with pytest.raises(PackValidationError):
        load_pack(bad)


def test_unknown_builtin_pack_raises():
    with pytest.raises(PackValidationError):
        builtin_pack_path("does-not-exist")


def _write(**extra):
    return {"type": "write", "session_id": "s1", "fact_key": "k", "role": "user",
            "content": "c", **extra}


def test_write_ids_are_unique_and_opaque():
    pack = parse_pack({"steps": [_write(), _write(fact_type="update"), _write(id="mine")]})

    ids = [s.id for s in pack.steps]
    assert len(set(ids)) == 3 and ids[2] == "mine"
    assert all("k" != i for i in ids)  # never the fact key


def test_duplicate_event_id_raises():
    with pytest.raises(PackValidationError, match="duplicate event id"):
        parse_pack({"steps": [_write(id="a"), _write(id="a")]})


def test_legacy_key_field_raises():
    with pytest.raises(PackValidationError, match="fact_key"):
        parse_pack({"steps": [{"type": "write", "session_id": "s1", "key": "k",
                               "role": "user", "content": "c"}]})


def test_forget_step_and_episodes():
    pack = parse_pack({"episodes": [
        {"steps": [_write(id="a"), {"type": "forget", "session_id": "s1", "target": "a"}]},
        {"steps": [_write(id="b")]},
    ]})

    assert [s.episode for s in pack.steps] == [0, 0, 1]
    assert isinstance(pack.steps[1], ForgetStep)
    assert pack.steps[1].target == ["a"]


def test_forget_target_must_be_an_earlier_write_in_the_episode():
    with pytest.raises(PackValidationError, match="forget target"):
        parse_pack({"steps": [{"type": "forget", "session_id": "s1", "target": ["a"]}]})
    with pytest.raises(PackValidationError, match="forget target"):
        parse_pack({"episodes": [
            {"steps": [_write(id="a")]},
            {"steps": [{"type": "forget", "session_id": "s1", "target": "a"}]},
        ]})


def test_empty_episode_raises():
    with pytest.raises(PackValidationError):
        parse_pack({"episodes": [{"steps": []}]})


def _parse(*steps):
    return parse_pack({"name": "p", "steps": list(steps)})


_WRITE = {"type": "write", "session_id": "s", "id": "a", "fact_key": "k", "role": "user",
          "content": "x"}


def test_query_v2_fields_are_parsed():
    pack = _parse(_WRITE, {
        "type": "query", "session_id": "s", "query": "q?", "category": "temporal",
        "answer": "x", "expect_absent": ["old"], "evidence": ["a"],
    })
    query = pack.steps[1]
    assert (query.category, query.answer, query.expect_absent, query.evidence, query.abstain) == (
        "temporal", "x", ["old"], ["a"], False
    )


@pytest.mark.parametrize(("query", "message"), [
    ({"abstain": True, "answer": "x"}, "has no 'answer'"),
    ({"evidence": ["nope"]}, "evidence 'nope' is not an earlier write"),
    ({"category": "forget"}, "must follow a forget step"),
])
def test_query_v2_fields_are_validated(query, message):
    with pytest.raises(PackValidationError, match=message):
        _parse(_WRITE, {"type": "query", "session_id": "s", "query": "q?", **query})


def test_unquoted_yaml_timestamp_becomes_an_iso_string(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text(
        "steps:\n  - {type: write, session_id: s, fact_key: k, role: user, content: x,\n"
        "     timestamp: 2025-01-14T10:02:00Z}\n"
    )
    assert load_pack(path).steps[0].timestamp == "2025-01-14T10:02:00+00:00"


@pytest.mark.parametrize("name", ["mini", "customer_support", "coding_assistant"])
def test_bundled_packs_set_explicit_timestamps_and_categories(name):
    raw = yaml.safe_load(builtin_pack_path(name).read_text())
    steps = [s for e in raw.get("episodes", [raw]) for s in e["steps"]]
    for step in steps:
        if step["type"] in ("write", "query"):
            assert isinstance(step.get("timestamp"), str), step  # never the 1-second default
        if step["type"] == "query":
            assert step.get("category"), step
            assert step.get("abstain") or step.get("answer"), step
    timestamps = [datetime.fromisoformat(s["timestamp"]) for s in steps if "timestamp" in s]
    assert timestamps == sorted(timestamps)
    assert (timestamps[-1] - timestamps[0]).days >= 60  # days to months apart
