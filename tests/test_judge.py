import httpx
import pytest

import membench.core.scoring.judge as judge_module
from membench.core.scoring.judge import (
    AnthropicClient,
    CachedJudge,
    JudgeCache,
    JudgeError,
    OpenAICompatibleClient,
    Verdict,
    build_judge,
    build_prompt,
    parse_verdict,
)


class FakeResponse:
    def __init__(self, payload, status_code=200, headers=None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://example.test")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)

    def json(self):
        return self._payload


class FakeHTTP:
    """Records POSTs and returns a canned JSON body (no real network)."""

    def __init__(self, payload, responses=None):
        self.payload = payload
        self.responses = list(responses or [])  # queued FakeResponses served first
        self.calls = []

    def post(self, url, headers=None, json=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        if self.responses:
            return self.responses.pop(0)
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


# --- retry / backoff --------------------------------------------------------

@pytest.fixture
def sleeps(monkeypatch):
    recorded = []
    monkeypatch.setattr(judge_module.time, "sleep", recorded.append)
    return recorded


def test_anthropic_client_retries_429_honouring_retry_after(sleeps):
    http = FakeHTTP(
        _anthropic_body('{"verdict": "pass"}'),
        responses=[FakeResponse(None, 429, {"retry-after": "7"})],
    )
    client = AnthropicClient(api_key="k", http_client=http)

    assert client.score("rubric", "evidence").passed is True
    assert len(http.calls) == 2
    assert sleeps == [7.0]


def test_openai_client_retries_5xx_with_exponential_backoff(sleeps):
    http = FakeHTTP(
        _openai_body('{"verdict": "pass"}'),
        responses=[FakeResponse(None, 503), FakeResponse(None, 500)],
    )
    client = OpenAICompatibleClient(
        api_key="k", model="gpt-x", base_url="https://gateway.example/v1", http_client=http
    )

    assert client.score("rubric", "evidence").passed is True
    assert len(http.calls) == 3
    assert sleeps == [1.0, 2.0]


def test_retry_gives_up_after_max_attempts(sleeps):
    responses = [FakeResponse(None, 429) for _ in range(judge_module._MAX_ATTEMPTS)]
    http = FakeHTTP(None, responses=responses)
    client = AnthropicClient(api_key="k", http_client=http)

    with pytest.raises(httpx.HTTPStatusError):
        client.score("rubric", "evidence")
    assert len(http.calls) == judge_module._MAX_ATTEMPTS
    assert len(sleeps) == judge_module._MAX_ATTEMPTS - 1


def test_non_retryable_error_raises_immediately(sleeps):
    http = FakeHTTP(None, responses=[FakeResponse(None, 401)])
    client = AnthropicClient(api_key="k", http_client=http)

    with pytest.raises(httpx.HTTPStatusError):
        client.score("rubric", "evidence")
    assert len(http.calls) == 1
    assert sleeps == []


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


# --- malformed replies / unexpected shapes -> JudgeError --------------------

def _cached(tmp_path, http, openai=False):
    if openai:
        client = OpenAICompatibleClient(
            api_key="k", model="gpt-x", base_url="https://gateway.example/v1", http_client=http
        )
    else:
        client = AnthropicClient(api_key="k", http_client=http)
    return CachedJudge(client, tmp_path / "cache.json")


def test_unparseable_reply_is_retried_once_then_succeeds(tmp_path):
    http = FakeHTTP(
        _anthropic_body('{"verdict": "pass"}'),
        responses=[FakeResponse(_anthropic_body("I think it passes."))],
    )

    assert _cached(tmp_path, http).score("r", "e").passed is True
    assert len(http.calls) == 2


@pytest.mark.parametrize("reply", ["no json here", '{"verdict": pass, }'])
def test_unparseable_reply_twice_raises_judge_error_and_is_not_cached(tmp_path, reply):
    http = FakeHTTP(_anthropic_body(reply))
    judge = _cached(tmp_path, http)

    with pytest.raises(JudgeError, match="failed to parse twice"):
        judge.score("r", "e")
    assert len(http.calls) == 2
    assert not (tmp_path / "cache.json").exists()


@pytest.mark.parametrize(
    ("body", "openai"),
    [
        ({"error": "no choices"}, True),  # KeyError
        ({"choices": []}, True),  # IndexError
        ({"choices": [{"message": {"content": None}}]}, True),  # TypeError
        ({"content": []}, False),  # IndexError
        (["not", "a", "dict"], False),  # TypeError
    ],
)
def test_unexpected_response_shape_raises_judge_error_without_retry(tmp_path, body, openai):
    http = FakeHTTP(body)

    with pytest.raises(JudgeError, match="unexpected judge response shape"):
        _cached(tmp_path, http, openai=openai).score("r", "e")
    assert len(http.calls) == 1


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
