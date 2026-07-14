import pytest

from membench.core.scoring.judge import (
    AnthropicClient,
    CachedJudge,
    JudgeCache,
    OpenAICompatibleClient,
    Verdict,
    build_judge,
    build_prompt,
    parse_verdict,
)


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeHTTP:
    """Records POSTs and returns a canned JSON body (no real network)."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def post(self, url, headers=None, json=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return FakeResponse(self.payload)


def _anthropic_body(text):
    return {"content": [{"type": "text", "text": text}]}


def _openai_body(text):
    return {"choices": [{"message": {"content": text}}]}


# --- prompt / parsing -------------------------------------------------------

def test_parse_verdict_pass_and_fail():
    assert parse_verdict('{"verdict": "pass", "reason": "ok"}') == Verdict(True, "ok")
    assert parse_verdict('{"verdict": "fail", "reason": "no"}') == Verdict(False, "no")


def test_parse_verdict_tolerates_surrounding_prose():
    assert parse_verdict('Sure! {"verdict":"pass"} done').passed is True


def test_parse_verdict_without_json_raises():
    with pytest.raises(ValueError):
        parse_verdict("no json here")


# --- Anthropic client (mocked HTTP) ----------------------------------------

def test_anthropic_client_posts_messages_and_parses():
    http = FakeHTTP(_anthropic_body('{"verdict": "pass", "reason": "green present"}'))
    client = AnthropicClient(api_key="k", model="claude-sonnet-5", http_client=http)

    verdict = client.score("rubric", "evidence")

    assert verdict == Verdict(True, "green present")
    call = http.calls[0]
    assert call["url"].endswith("/v1/messages")
    assert call["headers"]["x-api-key"] == "k"
    assert call["json"]["temperature"] == 0
    assert call["json"]["model"] == "claude-sonnet-5"


# --- OpenAI-compatible client (mocked HTTP) --------------------------------

def test_openai_compatible_client_posts_chat_completions_and_parses():
    http = FakeHTTP(_openai_body('{"verdict": "fail", "reason": "missing"}'))
    client = OpenAICompatibleClient(
        api_key="k", model="gpt-x", base_url="https://gateway.example/v1", http_client=http
    )

    verdict = client.score("rubric", "evidence")

    assert verdict == Verdict(False, "missing")
    call = http.calls[0]
    assert call["url"] == "https://gateway.example/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer k"
    assert call["json"]["temperature"] == 0


# --- cache ------------------------------------------------------------------

def test_cache_roundtrips_and_second_lookup_skips_client(tmp_path):
    http = FakeHTTP(_anthropic_body('{"verdict": "pass"}'))
    client = AnthropicClient(api_key="k", http_client=http)
    judge = CachedJudge(client, tmp_path / "cache.json")

    first = judge.score("rubric", "evidence")
    assert len(http.calls) == 1

    # New judge over the same cache file, fresh client that would blow up if called.
    reloaded = CachedJudge(AnthropicClient(api_key="k", http_client=FakeHTTP(None)), tmp_path / "cache.json")
    second = reloaded.score("rubric", "evidence")

    assert first == second


def test_cache_key_includes_model():
    prompt = build_prompt("r", "e")
    assert JudgeCache.key("model-a", prompt) != JudgeCache.key("model-b", prompt)


# --- build_judge ------------------------------------------------------------

def test_build_judge_returns_none_without_key(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMBENCH_JUDGE_API_KEY", raising=False)
    assert build_judge(cache_path=tmp_path / "c.json") is None


def test_build_judge_defaults_to_anthropic(tmp_path):
    judge = build_judge(api_key="k", cache_path=tmp_path / "c.json")
    assert isinstance(judge.client, AnthropicClient)


def test_build_judge_uses_openai_compatible_when_base_url_given(tmp_path):
    judge = build_judge(
        api_key="k", base_url="https://gateway.example/v1", cache_path=tmp_path / "c.json"
    )
    assert isinstance(judge.client, OpenAICompatibleClient)


def test_build_judge_reads_key_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMBENCH_JUDGE_API_KEY", "env-key")
    judge = build_judge(cache_path=tmp_path / "c.json")
    assert judge is not None and judge.client.api_key == "env-key"
