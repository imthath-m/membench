"""Cost accounting: price API-reported token counts against a dated table.

`membench/prices.yaml` holds USD per 1M tokens per model and an `as_of` date.
The judge's counts come from its client calls this run (cache hits are free),
the backend's from `adapter.usage()` priced against the models its
`describe()` names. A component whose model is missing from the table gets
`usd: None`, and so does the total.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PRICES_PATH = Path(__file__).resolve().parent.parent / "prices.yaml"


def load_prices(path: str | Path = PRICES_PATH) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def _usd(prices: dict, model: str | None, input_tokens: int, output_tokens: int = 0) -> float | None:
    if not input_tokens and not output_tokens:
        return 0.0
    rate = prices["models"].get(model) if model else None
    if rate is None:
        return None
    return (input_tokens * rate.get("input", 0) + output_tokens * rate.get("output", 0)) / 1e6


def _sum(values: list[float | None]) -> float | None:
    return None if any(v is None for v in values) else sum(values, 0.0)


def compute_cost(
    *,
    judge_model: str | None,
    judge_usage: dict[str, int] | None,
    backend_models: dict[str, Any] | None,
    backend_usage: dict[str, int] | None,
    n_queries: int,
    prices: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The report's `cost` section (USD per run and per 1k queries)."""
    prices = prices or load_prices()
    judge = None
    if judge_usage is not None:
        judge = {"model": judge_model, **judge_usage}
        judge["usd"] = _usd(
            prices, judge_model, judge_usage["input_tokens"], judge_usage["output_tokens"]
        )
    backend = None
    if backend_usage is not None:
        models = backend_models or {}
        backend = {
            "llm_model": models.get("llm_model"),
            "embedding_model": models.get("embedding_model"),
            **backend_usage,
        }
        backend["usd"] = _sum([
            _usd(prices, backend["llm_model"], backend_usage.get("llm_input_tokens", 0),
                 backend_usage.get("llm_output_tokens", 0)),
            _usd(prices, backend["embedding_model"], backend_usage.get("embedding_tokens", 0)),
        ])
    total = _sum([part["usd"] for part in (judge, backend) if part is not None])
    return {
        "price_table_as_of": prices["as_of"],
        "judge": judge,
        "reader": None,  # the reader stage arrives in Phase 11
        "backend": backend,
        "total_usd": total,
        "usd_per_1k_queries": total / n_queries * 1000 if total is not None and n_queries else None,
    }
