"""Transcript -> pack pipeline: load transcript, prompt the LLM client for
a draft pack, and validate it through `packs.schema.parse_pack` before it's
ever written to disk — so a generated pack always runs in the runner
unedited.

The LLM call reuses the judge's provider-agnostic client (Phase 3): the
same `AnthropicClient` / `OpenAICompatibleClient` configured by model name,
base URL, and `MEMBENCH_JUDGE_API_KEY` (see `core.scoring.judge.resolve_client`).
No new client is built here.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from membench.generate.prompt import build_generation_prompt
from membench.generate.transcript import load_transcripts
from membench.packs.schema import PackValidationError, parse_pack

_FENCE_RE = re.compile(r"^```(?:yaml)?\s*\n(.*?)\n```$", re.DOTALL)


class GenerationError(ValueError):
    """Raised when the LLM's draft pack fails schema validation."""


def _strip_code_fence(text: str) -> str:
    match = _FENCE_RE.match(text.strip())
    return match.group(1) if match else text.strip()


def generate_pack(transcripts_path: str | Path, client: Any, pack_name: str = "generated") -> dict:
    """Turn a transcript file/dir into a schema-valid pack dict.

    `client` is anything with a `complete(prompt: str) -> str` method — in
    practice an `AnthropicClient` or `OpenAICompatibleClient` from
    `core.scoring.judge`. Raises `GenerationError` if the reply isn't valid
    YAML or fails pack schema validation.
    """
    transcript = load_transcripts(transcripts_path)
    prompt = build_generation_prompt(transcript, pack_name)
    reply = client.complete(prompt)
    yaml_text = _strip_code_fence(reply)

    try:
        raw = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        raise GenerationError(f"LLM did not return valid YAML: {exc}") from exc

    try:
        parse_pack(raw, default_name=pack_name)
    except PackValidationError as exc:
        raise GenerationError(f"generated pack failed schema validation: {exc}") from exc

    return raw
