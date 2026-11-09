import httpx
import pytest

from membench.core.usage import meter_openai, new_usage


def test_meter_openai_counts_real_sdk_responses():
    openai = pytest.importorskip("openai")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(200, json={
                "object": "list", "model": "text-embedding-3-small",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            })
        return httpx.Response(200, json={
            "id": "c1", "object": "chat.completion", "created": 0, "model": "gpt-5-mini",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "{}"}}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 4, "total_tokens": 34},
        })

    client = openai.OpenAI(
        api_key="sk-test", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    totals = new_usage()
    meter_openai(client, totals)

    client.chat.completions.create(model="gpt-5-mini", messages=[{"role": "user", "content": "hi"}])
    client.embeddings.create(model="text-embedding-3-small", input="hi")

    assert totals == {"llm_input_tokens": 30, "llm_output_tokens": 4, "embedding_tokens": 5}


def test_meter_openai_ignores_none_client():
    totals = new_usage()
    meter_openai(None, totals)
    assert totals == new_usage()
