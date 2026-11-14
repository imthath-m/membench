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
warranted. Every call is cached on disk keyed by (judge model, prompt version,
prompt hash), with its raw reply and token `usage`, so scores from different
judges or prompts are never silently compared and CI replays without touching
the network (`strict` mode turns a cache miss into an error).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import httpx

DEFAULT_JUDGE_MODEL = "claude-sonnet-5"
# Bump whenever `build_prompt` or `parse_verdict` changes: it is part of the
# cache key and the run manifest, so verdicts from different prompts never mix.
JUDGE_PROMPT_VERSION = "1"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
API_KEY_ENV = "MEMBENCH_JUDGE_API_KEY"
_ANTHROPIC_VERSION = "2023-06-01"
_MAX_ATTEMPTS = 5
_BACKOFF_BASE_S = 1.0


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


class JudgeError(Exception):
    """A judge call gave no usable verdict; the query is recorded, not scored."""


class CacheMissError(Exception):
    """Strict cache: a verdict the run needs is not in the judge cache."""


def record_judge_error(detail: dict[str, Any], metric: str, exc: JudgeError) -> None:
    """Note a judge failure on a query trace's `detail`, keyed by metric."""
    detail.setdefault("judge_error", {})[metric] = str(exc)


def _build_prompt_v1(rubric: str, evidence: str) -> str:
    return (
        "You are a strict evaluator. Decide whether the STATEMENT is true of "
        "the EVIDENCE. Respond with ONLY a JSON object of the form "
        '{"verdict": "pass" | "fail", "reason": "<short reason>"}. '
        'Answer "pass" only if the EVIDENCE clearly supports the STATEMENT.\n\n'
        f"STATEMENT:\n{rubric}\n\nEVIDENCE:\n{evidence}\n"
    )


# CANDIDATE v2 — an UNTESTED hypothesis: never run against any model, no agreement
# numbers exist for it, and it is not the default (`JUDGE_PROMPT_VERSION` stays "1").
# Do not edit this text once any v2 result exists; write a v3 instead. Each rule
# targets a documented failure mode (docs/prompt-tuning.md, "Prompt v2"):
#   - every part / guessing / wording / near-miss / hedge: partial matches
#     (labelling-guidelines.md "Clearly", "Wording", qa_accuracy hedge rule);
#   - amount of evidence: length bias (judge-validation.md "by-length breakdown";
#     labelling-guidelines.md "Do not reward length");
#   - declines / does not know: abstention (labelling-guidelines.md qa_accuracy);
#   - marked as replaced: superseded values (labelling-guidelines.md semantic_recall
#     and contradiction_rate), phrased so a marked update is not a conflict and
#     unmarked conflicting values stay unresolved — consistent with both rubrics;
#   - reason (quoted span) before verdict: evidence-first, against unsupported passes.
# No few-shot examples: examples from validation/sample.jsonl would leak labelled items.
def _build_prompt_v2(rubric: str, evidence: str) -> str:
    return (
        "You are a strict evaluator. Decide only whether the STATEMENT is true of "
        "the EVIDENCE. Use no outside knowledge.\n"
        "Rules:\n"
        "- Pass only if the EVIDENCE clearly supports every part of the STATEMENT. "
        "If you must guess or assume, fail.\n"
        "- Wording and format may differ (paraphrases, synonyms, \"3 hours\" vs "
        "\"three hours\").\n"
        "- Topical overlap, a near-miss value, or a hedge between a right and a wrong "
        "answer is not support: fail.\n"
        "- The amount of EVIDENCE is not support. Long EVIDENCE passes only if it "
        "contains the specific support; short EVIDENCE that states it passes.\n"
        "- EVIDENCE that declines or says it does not know does not support a STATEMENT "
        "that needs a specific answer; it does support a STATEMENT that the answer is "
        "unknown or unavailable.\n"
        "- If the EVIDENCE gives different values for the same thing, a value it clearly "
        "marks as replaced (a later date, \"changed to\", \"now\", \"instead\") is an old "
        "value, not a conflict; unmarked conflicting values leave it unclear which one "
        "holds.\n"
        "First find the shortest EVIDENCE span that decides the STATEMENT. Respond with "
        "ONLY a JSON object of the form "
        '{"reason": "<that span, quoted, or what is missing>", "verdict": "pass" | "fail"}.'
        f"\n\nSTATEMENT:\n{rubric}\n\nEVIDENCE:\n{evidence}\n"
    )


# Prompt text per version. The default (`JUDGE_PROMPT_VERSION`) scores runs; other
# versions are candidates for `judge-tune`. Add one entry per new version.
PROMPT_BUILDERS: dict[str, Callable[[str, str], str]] = {
    "1": _build_prompt_v1,
    "2": _build_prompt_v2,  # candidate, untested (see above)
}


def build_prompt(rubric: str, evidence: str, version: str | None = None) -> str:
    """Deterministic structured prompt shared by every client and the cache key
    (`version` defaults to `JUDGE_PROMPT_VERSION`)."""
    version = version or JUDGE_PROMPT_VERSION
    try:
        builder = PROMPT_BUILDERS[version]
    except KeyError:
        raise ValueError(f"unknown judge prompt version {version!r}; known: "
                         f"{', '.join(sorted(PROMPT_BUILDERS))}") from None
    return builder(rubric, evidence)


def parse_verdict(text: str) -> Verdict:
    """Parse a model's JSON reply into a Verdict, tolerating surrounding prose."""
    if not isinstance(text, str):
        raise TypeError(f"judge reply text is {type(text).__name__}, not str")
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"judge reply had no JSON object: {text!r}")
    obj = json.loads(text[start : end + 1])
    verdict = str(obj.get("verdict", "")).strip().lower()
    return Verdict(passed=verdict == "pass", reason=str(obj.get("reason", "")))


def parse_yes_no(text: str) -> Verdict:
    """Official LongMemEval rule: the reply passes if it contains "yes"."""
    if not isinstance(text, str):
        raise TypeError(f"judge reply text is {type(text).__name__}, not str")
    return Verdict(passed="yes" in text.lower(), reason=text.strip())


def _post_with_retry(http: Any, url: str, **kwargs: Any) -> Any:
    """POST, retrying 429/5xx with exponential backoff (honours `Retry-After`)."""
    for attempt in range(_MAX_ATTEMPTS):
        resp = http.post(url, **kwargs)
        retryable = resp.status_code == 429 or resp.status_code >= 500
        if not retryable or attempt == _MAX_ATTEMPTS - 1:
            break
        try:
            delay = float(resp.headers.get("retry-after"))
        except (TypeError, ValueError):
            delay = _BACKOFF_BASE_S * 2**attempt
        time.sleep(delay)
    resp.raise_for_status()
    return resp


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

    def respond(self, prompt: str, max_tokens: int = 256) -> tuple[str, dict[str, int]]:
        """One completion: (reply text, `usage` as input/output tokens)."""
        resp = _post_with_retry(
            self._http,
            f"{self.base_url}/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": _ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        body = resp.json()
        text = body["content"][0]["text"]
        usage = body.get("usage") or {}
        return text, {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        }

    def complete(self, prompt: str, max_tokens: int = 4096) -> str:
        """Raw text completion — used by the pack generator."""
        return self.respond(prompt, max_tokens=max_tokens)[0]

    def score(self, rubric: str, evidence: str) -> Verdict:
        return parse_verdict(self.respond(build_prompt(rubric, evidence))[0])


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

    def respond(self, prompt: str) -> tuple[str, dict[str, int]]:
        """One completion: (reply text, `usage` as input/output tokens)."""
        resp = _post_with_retry(
            self._http,
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
        body = resp.json()
        text = body["choices"][0]["message"]["content"]
        usage = body.get("usage") or {}
        return text, {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }

    def complete(self, prompt: str) -> str:
        """Raw text completion — used by the pack generator."""
        return self.respond(prompt)[0]

    def score(self, rubric: str, evidence: str) -> Verdict:
        return parse_verdict(self.respond(build_prompt(rubric, evidence))[0])


class JudgeCache:
    """On-disk JSON cache keyed by (judge model, prompt version, prompt hash).

    An entry holds the verdict (`passed`, `reason`) or the `error` it failed
    with, plus the `raw` reply text and the call's token `usage`.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._data: dict[str, dict] = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text())

    @staticmethod
    def key(model: str, prompt: str, version: str | None = None) -> str:
        version = version or JUDGE_PROMPT_VERSION
        digest = hashlib.sha256(f"{model}\n{version}\n{prompt}".encode()).hexdigest()
        return f"{model}:v{version}:{digest}"

    def get(self, key: str) -> dict | None:
        return self._data.get(key)

    def set(self, key: str, entry: dict) -> None:
        self._data[key] = entry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True))


class CachedJudge:
    """Wraps a client with an on-disk cache; a cache hit makes zero client calls.

    A reply that fails to parse (`ValueError`) is retried once; an unexpected
    response shape (`KeyError`, `IndexError`, `TypeError`) is not. Either
    failure raises `JudgeError`; it is cached (with the raw reply) so a strict
    replay reproduces it, but a non-strict run calls the client again.

    `strict=True` never calls the client (it may be None): a cache miss raises
    `CacheMissError`. `usage` totals this run's client calls, tokens and hits.
    """

    def __init__(
        self,
        client: Any | None,
        cache_path: str | Path,
        *,
        strict: bool = False,
        model: str | None = None,
        base_url: str | None = None,
        prompt_version: str | None = None,
    ) -> None:
        self.prompt_version = prompt_version or JUDGE_PROMPT_VERSION
        build_prompt("", "", self.prompt_version)  # fail fast on an unknown version
        self.client = client
        self.model = model or client.model
        self.base_url = base_url or getattr(client, "base_url", None)
        self.strict = strict
        self.cache = JudgeCache(cache_path)
        self.usage = {"calls": 0, "cache_hits": 0, "input_tokens": 0, "output_tokens": 0}

    def score(self, rubric: str, evidence: str) -> Verdict:
        return self.ask(build_prompt(rubric, evidence, self.prompt_version))

    def ask(self, prompt: str, parse: Callable[[str], Verdict] = parse_verdict) -> Verdict:
        """Judge a ready-made prompt (e.g. an official benchmark template)."""
        key = JudgeCache.key(self.model, prompt, self.prompt_version)
        entry = self.cache.get(key)
        if entry is not None and (self.strict or "error" not in entry):
            self.usage["cache_hits"] += 1
        elif self.strict or self.client is None:
            raise CacheMissError(f"no cached verdict for {key} (strict judge cache)")
        else:
            entry = self._call(prompt, parse)
            self.cache.set(key, entry)
        if "error" in entry:
            raise JudgeError(entry["error"])
        return Verdict(passed=entry["passed"], reason=entry["reason"])

    def _call(self, prompt: str, parse: Callable[[str], Verdict]) -> dict:
        """Ask the client (one parse retry); return a cache entry."""
        usage, raw = {"input_tokens": 0, "output_tokens": 0}, None
        for attempt in range(2):
            self.usage["calls"] += 1
            try:
                raw, call_usage = self.client.respond(prompt)
                for name in usage:
                    usage[name] += call_usage.get(name, 0)
                    self.usage[name] += call_usage.get(name, 0)
                verdict = parse(raw)
                return {"passed": verdict.passed, "reason": verdict.reason, "raw": raw,
                        "usage": usage}
            except (KeyError, IndexError, TypeError) as exc:
                error = f"unexpected judge response shape: {type(exc).__name__}: {exc}"
                return {"error": error, "raw": raw, "usage": usage}
            except ValueError as exc:
                if attempt == 1:
                    error = f"judge reply failed to parse twice: {exc}"
                    return {"error": error, "raw": raw, "usage": usage}
        raise AssertionError("unreachable")  # pragma: no cover


def resolve_client(
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> Judge | None:
    """Resolve the provider-agnostic client config, or return None with no key.

    Shared by `build_judge` and `generate` (Phase 6 pack generation) — both
    are configured by the same three values: model, base URL, API key. An
    explicit `base_url` picks the OpenAI-compatible path; otherwise the
    default is Anthropic (Claude).
    """
    api_key = api_key or os.environ.get(API_KEY_ENV)
    if not api_key:
        return None
    model = model or DEFAULT_JUDGE_MODEL
    if base_url:
        return OpenAICompatibleClient(api_key=api_key, model=model, base_url=base_url)
    return AnthropicClient(api_key=api_key, model=model)


def build_judge(
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    cache_path: str | Path,
    strict: bool = False,
    prompt_version: str | None = None,
) -> CachedJudge | None:
    """Construct a cached judge, or return None when no API key is available.

    A missing key is not an error — the caller reports judge metrics as null.
    `strict` builds a cache-only judge (no client, no key needed).
    """
    if strict:
        return CachedJudge(
            None, cache_path, strict=True, model=model or DEFAULT_JUDGE_MODEL,
            base_url=base_url, prompt_version=prompt_version,
        )
    client = resolve_client(model=model, base_url=base_url, api_key=api_key)
    if client is None:
        return None
    return CachedJudge(client, cache_path, prompt_version=prompt_version)
