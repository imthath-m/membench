import random

import pytest
import yaml

from membench.adapters.full_context import FullContextAdapter
from membench.core.cost import compute_cost, load_prices
from membench.core.manifest import redact, seed_everything
from membench.core.runner import run_pack
from membench.core.scoring.judge import JUDGE_PROMPT_VERSION
from membench.packs.schema import builtin_pack_path, load_pack, parse_pack
from tests.test_scoring_judge_metrics import StubJudge

PRICES = {"as_of": "2026-01-01", "models": {
    "judge-m": {"input": 2.0, "output": 10.0},
    "llm-m": {"input": 0.25, "output": 2.0},
    "embed-m": {"input": 0.02},
}}


# --- manifest ---------------------------------------------------------------

def test_redact_masks_secret_keys_recursively():
    config = {
        "user_id": "u", "top_k": 3, "api_key": "sk-1",
        "mem0_config": {"llm": {"config": {"openai_api_key": "sk-2", "max_tokens": 100}}},
        "headers": [{"auth_token": "t"}], "password": None,
    }
    assert redact(config) == {
        "user_id": "u", "top_k": 3, "api_key": "<redacted>",
        "mem0_config": {"llm": {"config": {"openai_api_key": "<redacted>", "max_tokens": 100}}},
        "headers": [{"auth_token": "<redacted>"}], "password": None,
    }


def test_manifest_records_inputs():
    pack = load_pack(builtin_pack_path("mini"))
    adapter = FullContextAdapter({"api_key": "secret", "region": "eu-west-1"})

    result = run_pack(pack, adapter, "full-context", judge=StubJudge("x"), seed=7, repeat=2)
    manifest = result.manifest

    assert manifest["pack"] == {
        "name": "mini", "sha256": pack.sha256, "subset": None, "qa_style": "membench",
    }
    assert manifest["dataset"] is None
    assert manifest["backend"]["config"]["api_key"] == "<redacted>"
    assert manifest["backend"]["region"] == "eu-west-1"
    assert manifest["judge"] == {
        "model": "stub-model", "base_url": None, "prompt_version": JUDGE_PROMPT_VERSION,
    }
    assert manifest["tokenizer"] == "whitespace:approx"
    assert (manifest["seed"], manifest["repeat"]) == (7, 2)
    assert manifest["started_at"] == result.started_at
    assert manifest["membench"]["git_commit"] is None or len(manifest["membench"]["git_commit"]) == 40


def test_pack_sha256_ignores_formatting_but_not_content():
    raw = yaml.safe_load(builtin_pack_path("mini").read_text())
    reformatted = yaml.safe_load(yaml.safe_dump(raw, default_flow_style=True))
    assert parse_pack(raw).sha256 == parse_pack(reformatted).sha256
    raw["steps"][0]["content"] += "!"
    assert parse_pack(raw).sha256 != parse_pack(reformatted).sha256


def test_seed_everything_is_repeatable():
    seed_everything(3)
    first = random.random()
    seed_everything(3)
    assert random.random() == first


# --- cost -------------------------------------------------------------------

def test_cost_prices_judge_and_backend():
    cost = compute_cost(
        judge_model="judge-m",
        judge_usage={"calls": 2, "cache_hits": 1, "input_tokens": 1_000_000, "output_tokens": 100_000},
        backend_models={"llm_model": "llm-m", "embedding_model": "embed-m"},
        backend_usage={"llm_input_tokens": 2_000_000, "llm_output_tokens": 0,
                       "embedding_tokens": 500_000},
        n_queries=4,
        prices=PRICES,
    )

    assert cost["judge"]["usd"] == pytest.approx(3.0)
    assert cost["backend"]["usd"] == pytest.approx(0.51)
    assert cost["total_usd"] == pytest.approx(3.51)
    assert cost["usd_per_1k_queries"] == pytest.approx(877.5)
    assert cost["price_table_as_of"] == "2026-01-01"


def test_unpriced_model_makes_usd_null():
    cost = compute_cost(
        judge_model="unknown", judge_usage={"input_tokens": 10, "output_tokens": 1},
        backend_models=None, backend_usage=None, n_queries=1, prices=PRICES,
    )
    assert cost["judge"]["usd"] is None and cost["total_usd"] is None
    assert cost["usd_per_1k_queries"] is None


def test_zero_tokens_cost_nothing_even_unpriced():
    cost = compute_cost(
        judge_model="unknown", judge_usage={"input_tokens": 0, "output_tokens": 0},
        backend_models=None, backend_usage=None, n_queries=1, prices=PRICES,
    )
    assert cost["total_usd"] == 0.0


def test_bundled_price_table_is_dated_and_prices_defaults():
    prices = load_prices()
    assert prices["as_of"]
    for model in ("claude-sonnet-5", "gpt-5-mini", "text-embedding-3-small"):
        assert model in prices["models"]
