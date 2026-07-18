"""Builds the single LLM prompt that turns a transcript into a draft pack.

The prompt asks for the exact YAML shape `membench.packs.schema` validates,
so the reply can be parsed and checked before it's ever written to disk.
"""

from __future__ import annotations

_SCHEMA_INSTRUCTIONS = """\
Output a MemBench scenario pack as YAML with this exact shape:

name: <short slug>
description: <one paragraph>
steps:
  - type: write
    session_id: <string, groups steps into a conversation/session>
    key: <stable id for this fact, reused when it is updated/contradicted>
    role: user | assistant
    content: <the fact as it was actually said>
    fact_type: state | update | contradict
  - type: query
    session_id: <string>
    query: <a question the agent should answer using memory>
    expect_contains: [<substrings that must appear in the recalled context>]
    expect_key: <the key of the fact this query is testing>

Rules:
- Every fact worth remembering is a `write` step with `fact_type: state`
  the first time it's stated.
- If a later message changes that fact, emit another `write` with the
  same `key` and `fact_type: update` (a plain revision) or `fact_type:
  contradict` (an explicit correction of earlier wrong info).
- Every `query` step must reference a real `key` from an earlier `write`
  via `expect_key`, and `expect_contains` must list substrings that
  actually appear in the transcript for the *current* (post-update)
  value of that fact.
- The pack must include at least one `state` write, one `update` write,
  one `contradict` write, and a `query` step after each of them.
- Preserve session boundaries from the transcript in `session_id`.
- Output ONLY the YAML — no prose, no markdown code fences.
"""


def build_generation_prompt(transcript: str, pack_name: str) -> str:
    """Compose the full generation prompt for a given transcript."""
    return (
        f"{_SCHEMA_INSTRUCTIONS}\n"
        f"Use this pack name: {pack_name}\n\n"
        "TRANSCRIPT:\n"
        f"{transcript}\n"
    )
