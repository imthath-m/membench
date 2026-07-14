"""Pluggable, cached LLM judge for the semantic scoring metrics.

The judge answers a single yes/no question — defined by a `rubric` — about a
piece of `evidence` (the recalled context), returning a structured Verdict.
It sits behind the small `Judge` protocol exactly like a backend sits behind
`MemoryAdapter`: configured by three values (model, base URL, API key).

Two thin HTTP clients are provided:
  - `AnthropicClient`  — Anthropic Messages API; the default (Claude).
  - `OpenAICompatibleClient` — any OpenAI-compatible /chat/completions
    endpoint (OpenAI, Gemini's compat surface, most gateways), selected
    automatically when a base URL is supplied.

Both talk their HTTP APIs directly via httpx rather than pulling in a vendor
SDK — the judge makes one structured call, so a heavyweight dependency isn't
warranted. Every call is cached on disk keyed by (judge model, prompt hash),
so scores from different judges are never silently compared and CI replays
without touching the network.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import httpx

DEFAULT_JUDGE_MODEL = "claude-sonnet-5"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
API_KEY_ENV = "MEMBENCH_JUDGE_API_KEY"
_ANTHROPIC_VERSION = "2023-06-01"


@dataclass
class Verdict:
    """A judge's yes/no answer to a rubric, plus a short reason."""

    passed: bool
    reason: str = ""


@runtime_checkable
class Judge(Protocol):
    """Anything that can answer a rubric about evidence."""

    model: str

    def score(self, rubric: str, evidence: str) -> Verdict: ...


def build_prompt(rubric: str, evidence: str) -> str:
    """Deterministic structured prompt shared by every client and the cache key."""
    return (
        "You are a strict evaluator. Decide whether the STATEMENT is true of "
        "the EVIDENCE. Respond with ONLY a JSON object of the form "
        '{"verdict": "pass" | "fail", "reason": "<short reason>"}. '
        'Answer "pass" only if the EVIDENCE clearly supports the STATEMENT.\n\n'
        f"STATEMENT:\n{rubric}\n\nEVIDENCE:\n{evidence}\n"
    )


def parse_verdict(text: str) -> Verdict:
    """Parse a model's JSON reply into a Verdict, tolerating surrounding prose."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"judge reply had no JSON object: {text!r}")
    obj = json.loads(text[start : end + 1])
    verdict = str(obj.get("verdict", "")).strip().lower()
    return Verdict(passed=verdict == "pass", reason=str(obj.get("reason", "")))


class AnthropicClient:
    """Thin Anthropic Messages API judge client (the default, Claude)."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_JUDGE_MODEL,
        base_url: str | None = None,
        http_client: Any | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = (base_url or DEFAULT_ANTHROPIC_BASE_URL).rstrip("/")
        self._http = http_client or httpx.Client(timeout=60)

    def score(self, rubric: str, evidence: str) -> Verdict:
        prompt = build_prompt(rubric, evidence)
        resp = self._http.post(
            f"{self.base_url}/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": _ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 256,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        resp.raise_for_status()
        text = resp.json()["content"][0]["text"]
        return parse_verdict(text)


class OpenAICompatibleClient:
    """Thin client for any OpenAI-compatible /chat/completions endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        http_client: Any | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._http = http_client or httpx.Client(timeout=60)

    def score(self, rubric: str, evidence: str) -> Verdict:
        prompt = build_prompt(rubric, evidence)
        resp = self._http.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        return parse_verdict(text)


class JudgeCache:
    """On-disk JSON cache of verdicts keyed by (judge model, prompt hash)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._data: dict[str, dict] = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text())

    @staticmethod
    def key(model: str, prompt: str) -> str:
        digest = hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()
        return f"{model}:{digest}"

    def get(self, key: str) -> Verdict | None:
        entry = self._data.get(key)
        return Verdict(**entry) if entry is not None else None

    def set(self, key: str, verdict: Verdict) -> None:
        self._data[key] = {"passed": verdict.passed, "reason": verdict.reason}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True))


class CachedJudge:
    """Wraps a client with an on-disk cache; a cache hit makes zero client calls."""

    def __init__(self, client: Judge, cache_path: str | Path) -> None:
        self.client = client
        self.model = client.model
        self.cache = JudgeCache(cache_path)

    def score(self, rubric: str, evidence: str) -> Verdict:
        prompt = build_prompt(rubric, evidence)
        key = JudgeCache.key(self.model, prompt)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        verdict = self.client.score(rubric, evidence)
        self.cache.set(key, verdict)
        return verdict


def build_judge(
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    cache_path: str | Path,
) -> CachedJudge | None:
    """Construct a cached judge, or return None when no API key is available.

    A missing key is not an error — the caller reports judge metrics as null.
    Client selection: an explicit `base_url` picks the OpenAI-compatible path;
    otherwise the default is Anthropic (Claude).
    """
    api_key = api_key or os.environ.get(API_KEY_ENV)
    if not api_key:
        return None
    model = model or DEFAULT_JUDGE_MODEL
    client: Judge
    if base_url:
        client = OpenAICompatibleClient(api_key=api_key, model=model, base_url=base_url)
    else:
        client = AnthropicClient(api_key=api_key, model=model)
    return CachedJudge(client, cache_path)
