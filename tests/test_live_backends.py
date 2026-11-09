"""Gated live runs of the hosted / LLM-backed backends against the `mini` pack.

Skipped unless `MEMBENCH_LIVE=1` and the backend's keys are set, so CI and
offline runs never touch the network. `live.yml` runs the same backends
through `membench run`; run one locally with e.g.
`MEMBENCH_LIVE=1 ZEP_API_KEY=... pytest tests/test_live_backends.py -k zep`.
"""

from __future__ import annotations

import os
import uuid

import pytest

from membench.core.registry import get_adapter_class
from membench.core.runner import run_pack
from membench.packs.schema import builtin_pack_path, load_pack

# backend -> env vars it needs (the vendor key, plus the LLM key it extracts with)
LIVE_KEYS = {
    "mem0-platform": ("MEM0_API_KEY",),
    "zep": ("ZEP_API_KEY",),
    "letta": ("LETTA_API_KEY",),
    "letta-agent": ("LETTA_API_KEY",),
    "langmem": ("OPENAI_API_KEY",),
    "cognee": ("LLM_API_KEY",),
}


@pytest.mark.parametrize("backend", sorted(LIVE_KEYS))
def test_live_mini_runs_clean(backend):
    if os.environ.get("MEMBENCH_LIVE") != "1":
        pytest.skip("live test: set MEMBENCH_LIVE=1")
    missing = [key for key in LIVE_KEYS[backend] if not os.environ.get(key)]
    if missing:
        pytest.skip(f"live test: {', '.join(missing)} not set")
    try:
        adapter_cls = get_adapter_class(backend)
        adapter = adapter_cls({"namespace": f"membench-live-{uuid.uuid4().hex[:8]}"})
    except ImportError as exc:
        pytest.skip(str(exc))

    adapter.reset()
    try:
        result = run_pack(load_pack(builtin_pack_path("mini")), adapter, backend_name=backend)
    finally:
        adapter.reset()  # leave nothing behind on the vendor's side

    assert result.scores["error_rate"] == 0.0, [t.error for t in result.traces if t.error]
    assert result.scores["recall_precision"] is not None
