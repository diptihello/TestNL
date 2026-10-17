"""Groq and fake client behavior, tested without network access."""

from __future__ import annotations

from types import SimpleNamespace

import groq
import httpx
import pytest

from zomato_reco.config import load_settings
from zomato_reco.llm.client import (
    FakeLLMClient,
    GroqClient,
    LLMAuthError,
    LLMRateLimited,
    LLMResponseError,
    LLMTimeout,
    LLMUnavailable,
    _safe_provider_error,
    make_client,
)
from zomato_reco.llm.prompts import SYSTEM_PROMPT


def _settings(tmp_path, **overrides):
    return load_settings(_env_file=tmp_path / "missing.env", **overrides)


def test_fake_client_builds_recommendations_from_candidate_ids() -> None:
    client = FakeLLMClient()
    response = client.rank(
        SYSTEM_PROMPT,
        'CANDIDATES\n{"id":"r_1","name":"One"}\n{"id":"r_2","name":"Two"}\n',
        100,
    )
    assert '"id":"r_1"' in response.content
    assert '"id":"r_2"' in response.content
    assert response.total_tokens == response.prompt_tokens + response.completion_tokens


def test_fake_client_can_raise_and_return_malformed_content() -> None:
    with pytest.raises(LLMTimeout):
        FakeLLMClient(scripted={"error": LLMTimeout()}).rank(SYSTEM_PROMPT, "x", 10)
    response = FakeLLMClient(scripted={"malformed": True}).rank(SYSTEM_PROMPT, "x", 10)
    assert response.content == "{not valid json"


def test_provider_error_classes_are_safely_mapped() -> None:
    request = httpx.Request("POST", "https://api.groq.com")
    timeout = _safe_provider_error(groq.APITimeoutError(request=request))
    assert isinstance(timeout, LLMTimeout)
    rate_response = httpx.Response(
        429, headers={"retry-after": "12"}, request=request
    )
    rate = _safe_provider_error(
        groq.RateLimitError("provider detail", response=rate_response, body=None)
    )
    assert isinstance(rate, LLMRateLimited)
    assert rate.retry_after == 12
    auth_response = httpx.Response(401, request=request)
    auth = _safe_provider_error(
        groq.AuthenticationError("provider detail", response=auth_response, body=None)
    )
    assert isinstance(auth, LLMAuthError)
    unavailable = _safe_provider_error(groq.APIConnectionError(request=request))
    assert isinstance(unavailable, LLMUnavailable)
    assert "provider detail" not in str(rate)


def test_client_selection_prefers_fake_without_key_and_in_demo_mode(tmp_path) -> None:
    assert isinstance(make_client(_settings(tmp_path)), FakeLLMClient)
    configured = _settings(
        tmp_path, groq_api_key="gsk_" + "a" * 40, demo_mode=True
    )
    assert isinstance(make_client(configured), FakeLLMClient)


def test_groq_client_is_created_without_retries(tmp_path, monkeypatch) -> None:
    observed = {}

    class StubGroq:
        def __init__(self, **kwargs):
            observed.update(kwargs)

    monkeypatch.setattr(groq, "Groq", StubGroq)
    settings = _settings(tmp_path, groq_api_key="gsk_" + "a" * 40)
    client = GroqClient(settings)
    assert client is not None
    assert observed["timeout"] == settings.llm_timeout_seconds
    assert observed["max_retries"] == 0


def test_finish_reason_length_raises_response_error(tmp_path, monkeypatch) -> None:
    observed = {}

    def create(**request):
        observed.update(request)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="length",
                    message=SimpleNamespace(content='{"truncated":'),
                )
            ],
            usage=None,
        )

    class StubGroq:
        def __init__(self, **kwargs):
            observed["client"] = kwargs
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )

    monkeypatch.setattr(groq, "Groq", StubGroq)
    client = GroqClient(_settings(tmp_path, groq_api_key="gsk_" + "a" * 40))
    with pytest.raises(LLMResponseError, match="cut off"):
        client.rank(SYSTEM_PROMPT, "prompt", 5)
    assert observed["response_format"] == {"type": "json_object"}
    assert observed["temperature"] == 0.2
    assert observed["max_tokens"] == 5


def test_missing_usage_uses_local_estimates(tmp_path, monkeypatch) -> None:
    observed = {}

    def create(**request):
        observed.update(request)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content='{"ok":true}'),
                )
            ],
            usage=None,
        )

    class StubGroq:
        def __init__(self, **kwargs):
            assert kwargs["max_retries"] == 0
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )

    monkeypatch.setattr(groq, "Groq", StubGroq)
    client = GroqClient(_settings(tmp_path, groq_api_key="gsk_" + "a" * 40))
    response = client.rank(SYSTEM_PROMPT, "prompt", 20)
    assert response.prompt_tokens > 0
    assert response.completion_tokens > 0
    assert response.total_tokens == response.prompt_tokens + response.completion_tokens
    assert observed["response_format"] == {"type": "json_object"}
