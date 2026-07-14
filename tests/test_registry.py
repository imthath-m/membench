import pytest

from membench.adapters.full_context import FullContextAdapter
from membench.core.registry import get_adapter_class, list_backends


def test_list_backends():
    assert list_backends() == ["chroma", "full-context"]


def test_get_adapter_class_full_context():
    assert get_adapter_class("full-context") is FullContextAdapter


def test_get_adapter_class_unknown_raises():
    with pytest.raises(ValueError):
        get_adapter_class("nonexistent-backend")
