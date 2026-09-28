import json

import httpx
import pytest
from pydantic import BaseModel

from src.config import LLMConfig
from src.providers.llm import GroqProvider, get_llm_provider
from src.providers.llm.groq import _parse_duration_seconds, extract_json_block


class Person(BaseModel):
    name: str
    age: int


def _completion(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def make_provider(handler, **kwargs) -> tuple[GroqProvider, list[dict]]:
    calls: list[dict] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        return handler(body, len(calls))

    provider = GroqProvider(api_key="gsk_test", transport=httpx.MockTransport(wrapped), **kwargs)
    provider._sleep = lambda _seconds: None
    return provider, calls


def test_generate_json_uses_json_mode_and_parses():
    provider, calls = make_provider(lambda _body, _n: _completion('{"name": "Ada", "age": 36}'))
    result = provider.generate_json("Extract the person", Person)
    assert isinstance(result, Person) and result.name == "Ada"
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert calls[0]["model"] == "llama-3.3-70b-versatile"
    assert calls[0]["messages"][0]["role"] == "system"


def test_rate_limit_is_retried_with_retry_after():
    def handler(body, n):
        if n == 1:
            return httpx.Response(429, headers={"retry-after": "2"}, json={"error": {"message": "rate limit"}})
        return _completion("pong")

    provider, calls = make_provider(handler)
    slept: list[float] = []
    provider._sleep = slept.append
    assert provider.generate_text("ping") == "pong"
    assert slept == [2.0]
    assert len(calls) == 2


def test_decommissioned_model_falls_back():
    def handler(body, n):
        if body["model"] == "old-model":
            return httpx.Response(400, json={"error": {"code": "model_decommissioned", "message": "decommissioned"}})
        return _completion("ok from fallback")

    provider, calls = make_provider(handler, model="old-model", fallback_models=["llama-3.1-8b-instant"])
    assert provider.generate_text("hi") == "ok from fallback"
    assert [c["model"] for c in calls] == ["old-model", "llama-3.1-8b-instant"]
    # the dead model is remembered and skipped next time
    provider.generate_text("again")
    assert calls[-1]["model"] == "llama-3.1-8b-instant"


def test_long_rate_limit_switches_model_instead_of_waiting():
    def handler(body, n):
        if body["model"] == "big":
            return httpx.Response(429, headers={"retry-after": "3600"}, json={"error": {"message": "daily tokens"}})
        return _completion("small model answer")

    provider, calls = make_provider(handler, model="big", fallback_models=["small"])
    assert provider.generate_text("hi") == "small model answer"


def test_json_validate_failed_retries_without_json_mode():
    def handler(body, n):
        if "response_format" in body:
            return httpx.Response(400, json={"error": {"code": "json_validate_failed", "message": "json_validate_failed"}})
        return _completion('Sure! ```json\n{"name": "Lin", "age": 29}\n```')

    provider, calls = make_provider(handler)
    result = provider.generate_json("x", Person)
    assert result.name == "Lin"
    assert "response_format" not in calls[-1]


def test_invalid_json_triggers_repair_round():
    def handler(body, n):
        return _completion('{"name": "Bo"}') if n == 1 else _completion('{"name": "Bo", "age": 41}')

    provider, calls = make_provider(handler)
    result = provider.generate_json("x", Person)
    assert result.age == 41
    assert "failed validation" in calls[1]["messages"][-1]["content"]


def test_missing_key_raises_clear_error():
    provider = GroqProvider(api_key="")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        provider.generate_text("hi")


def test_factory_reads_env_key_and_fast_model(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_env")
    cfg = LLMConfig(provider="groq", model="llama-3.3-70b-versatile", fast_model="llama-3.1-8b-instant")
    main = get_llm_provider(cfg)
    fast = get_llm_provider(cfg, fast=True)
    assert isinstance(main, GroqProvider) and main.api_key == "gsk_env"
    assert fast.model == "llama-3.1-8b-instant"


def test_helpers():
    assert _parse_duration_seconds("2m59.5s") == pytest.approx(179.5)
    assert _parse_duration_seconds("7.66s") == pytest.approx(7.66)
    assert _parse_duration_seconds("12") == 12.0
    assert extract_json_block('<think>hmm</think> here {"a": 1} done') == '{"a": 1}'
