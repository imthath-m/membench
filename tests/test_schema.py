import pytest

from membench.packs.schema import (
    PackValidationError,
    QueryStep,
    WriteStep,
    builtin_pack_path,
    load_pack,
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
