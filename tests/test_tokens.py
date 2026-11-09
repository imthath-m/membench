import builtins

import pytest

from membench.core import tokens
from membench.core.tokens import count_tokens, default_tokenizer


def test_whitespace_count_is_labelled_approx():
    assert count_tokens("a b  c") == 3
    assert count_tokens("") == 0
    assert count_tokens.name == "whitespace:approx"


def test_default_tokenizer_falls_back_without_tiktoken(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "tiktoken":
            raise ImportError("No module named 'tiktoken'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    assert default_tokenizer() is count_tokens


def test_tiktoken_counter_counts_o200k_tokens():
    pytest.importorskip("tiktoken")
    try:
        counter = tokens.TiktokenCounter()
    except Exception as exc:  # noqa: BLE001 - vocab not cached and no network
        pytest.skip(f"o200k_base vocab unavailable: {exc}")
    assert counter.name == "tiktoken:o200k_base"
    assert counter("hello world") == 2
    assert counter("") == 0
    assert counter("<|endoftext|>") > 0  # special tokens are counted as text
