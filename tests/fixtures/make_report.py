"""Regenerate the committed report + judge cache used by the rescore tests.

    python tests/fixtures/make_report.py

Runs the `customer_support` pack on `full-context` with a deterministic fake
judge client (no API key, no network) and writes `report_customer_support.json`
and `judge_cache_customer_support.json` next to this file.
"""

from __future__ import annotations

import json
from pathlib import Path

from membench.adapters.full_context import FullContextAdapter
from membench.core.report.json import write_json
from membench.core.runner import run_pack
from membench.core.scoring.judge import DEFAULT_JUDGE_MODEL, CachedJudge
from membench.packs.schema import builtin_pack_path, load_pack

HERE = Path(__file__).parent
REPORT = HERE / "report_customer_support.json"
CACHE = HERE / "judge_cache_customer_support.json"


class FakeJudgeClient:
    """Passes when the evidence is non-empty; reports fixed token usage."""

    model = DEFAULT_JUDGE_MODEL
    base_url = "https://api.anthropic.com"

    def respond(self, prompt: str) -> tuple[str, dict[str, int]]:
        evidence = prompt.split("EVIDENCE:")[-1].strip()
        verdict = "pass" if evidence else "fail"
        reply = json.dumps({"verdict": verdict, "reason": "fixture"})
        return reply, {"input_tokens": len(prompt.split()), "output_tokens": 12}


def main() -> None:
    CACHE.unlink(missing_ok=True)
    judge = CachedJudge(FakeJudgeClient(), CACHE)
    pack = load_pack(builtin_pack_path("customer_support"))
    result = run_pack(pack, FullContextAdapter(), backend_name="full-context", judge=judge)
    write_json(result, REPORT)


if __name__ == "__main__":
    main()
