"""Groq API client and deterministic fake implementation."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from zomato_reco.config import Settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base class for safe, user-presentable LLM failures."""


class LLMTimeout(LLMError):
    """The provider did not respond before the configured timeout."""


class LLMRateLimited(LLMError):
    """The provider rejected a request due to rate limits."""

    def __init__(self, message: str = "The AI service is rate limited.", retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class LLMAuthError(LLMError):
    """The provider rejected the configured credentials."""


class LLMUnavailable(LLMError):
    """The provider could not be reached or is unavailable."""


class RawLLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: int
    model: str
    finish_reason: str | None = None


class ConnectionStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    model: str
    latency_ms: int
    tokens_used: int
    message: str


class LLMResponseError(LLMError):
    """The provider returned an empty or incomplete response."""


def _is_provider_error(exc: Exception) -> bool:
    import groq
    import httpx

    return isinstance(exc, (groq.APIError, httpx.HTTPError))


class LLMClient(Protocol):
    def test_connection(self) -> ConnectionStatus: ...

    def rank(
        self, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> RawLLMResponse: ...


def _retry_after(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _safe_provider_error(exc: Exception) -> LLMError:
    """Map SDK/transport errors without copying provider text into messages."""
    import groq
    import httpx

    if isinstance(exc, (groq.APITimeoutError, httpx.TimeoutException)):
        return LLMTimeout("The AI service timed out.")
    if isinstance(exc, groq.RateLimitError):
        return LLMRateLimited(retry_after=_retry_after(exc))
    if isinstance(exc, groq.AuthenticationError):
        return LLMAuthError("The AI service rejected the configured API key.")
    if isinstance(exc, (groq.APIConnectionError, httpx.NetworkError)):
        return LLMUnavailable("The AI service could not be reached.")
    if isinstance(exc, groq.APIStatusError) and exc.status_code >= 500:
        return LLMUnavailable("The AI service is temporarily unavailable.")
    return LLMError("The AI request failed.")


class GroqClient:
    """A no-retry Groq wrapper with bounded timeouts and safe errors."""

    def __init__(self, settings: Settings) -> None:
        from groq import Groq

        self._settings = settings
        self._client = Groq(
            api_key=settings.groq_api_key.get_secret_value(),
            timeout=settings.llm_timeout_seconds,
            max_retries=0,
        )

    def test_connection(self) -> ConnectionStatus:
        from zomato_reco.llm.prompts import estimate_tokens

        prompt = "Reply with OK"
        started = time.monotonic()
        try:
            response = self._client.chat.completions.create(
                model=self._settings.groq_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
            )
        except Exception as exc:
            if not _is_provider_error(exc):
                raise
            mapped = _safe_provider_error(exc)
            logger.warning("llm_connection_failed", extra={"error_class": type(mapped).__name__})
            raise mapped from None

        latency = int((time.monotonic() - started) * 1000)
        usage = getattr(response, "usage", None)
        tokens_used = getattr(usage, "total_tokens", None)
        if tokens_used is None:
            tokens_used = estimate_tokens(prompt) + 5
            logger.warning("llm_usage_missing", extra={"operation": "test_connection"})
        return ConnectionStatus(
            ok=True,
            model=self._settings.groq_model,
            latency_ms=latency,
            tokens_used=int(tokens_used),
            message="Connection successful.",
        )

    def rank(
        self, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> RawLLMResponse:
        from zomato_reco.llm.prompts import estimate_tokens

        started = time.monotonic()
        try:
            response = self._client.chat.completions.create(
                model=self._settings.groq_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            if not _is_provider_error(exc):
                raise
            mapped = _safe_provider_error(exc)
            logger.warning("llm_request_failed", extra={"error_class": type(mapped).__name__})
            raise mapped from None

        latency = int((time.monotonic() - started) * 1000)
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMResponseError("The AI service returned no response.")
        choice = choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            raise LLMResponseError("The AI response was cut off.")
        content = getattr(getattr(choice, "message", None), "content", None)
        if not isinstance(content, str) or not content.strip():
            raise LLMResponseError("The AI service returned an empty response.")

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        usage_missing = prompt_tokens is None or completion_tokens is None
        if usage_missing:
            prompt_tokens = estimate_tokens(system_prompt + "\n" + user_prompt)
            completion_tokens = estimate_tokens(content)
            logger.warning("llm_usage_missing", extra={"operation": "rank"})
        prompt_tokens = int(prompt_tokens)
        completion_tokens = int(completion_tokens)
        total_tokens = None if usage_missing else getattr(usage, "total_tokens", None)
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens
        return RawLLMResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=int(total_tokens),
            latency_ms=latency,
            model=self._settings.groq_model,
            finish_reason=getattr(choice, "finish_reason", None),
        )


@dataclass
class FakeLLMClient:
    """Offline client that returns valid output for candidate ids in the prompt."""

    scripted: dict[str, Any] | None = None
    model: str = "fake-model"

    def _raise_scripted_error(self) -> None:
        error = (self.scripted or {}).get("error")
        if error is None:
            return
        if isinstance(error, Exception):
            raise error
        if isinstance(error, type) and issubclass(error, Exception):
            raise error()
        raise TypeError("scripted error must be an exception or exception type")

    def test_connection(self) -> ConnectionStatus:
        self._raise_scripted_error()
        return ConnectionStatus(
            ok=True, model=self.model, latency_ms=0, tokens_used=5,
            message="Demo connection successful.",
        )

    def rank(
        self, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> RawLLMResponse:
        del system_prompt
        del max_tokens
        self._raise_scripted_error()
        started = time.monotonic()
        scripted = self.scripted or {}
        if "content" in scripted:
            content = str(scripted["content"])
        elif scripted.get("malformed"):
            content = "{not valid json"
        else:
            candidate_ids: list[str] = []
            for line in user_prompt.splitlines():
                if not line.startswith("{"):
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict) and isinstance(item.get("id"), str):
                    candidate_ids.append(item["id"])
            if not candidate_ids:
                raise LLMResponseError("The demo prompt contains no candidates.")
            content = json.dumps(
                {
                    "summary": "These options match the structured search preferences.",
                    "recommendations": [
                        {
                            "id": candidate_id,
                            "rank": index,
                            "explanation": "A strong match based on the available restaurant data.",
                            "matched": ["available restaurant data"],
                        }
                        for index, candidate_id in enumerate(candidate_ids, start=1)
                    ],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        from zomato_reco.llm.prompts import estimate_tokens

        prompt_tokens = estimate_tokens(user_prompt)
        completion_tokens = estimate_tokens(content)
        return RawLLMResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=int((time.monotonic() - started) * 1000),
            model=self.model,
            finish_reason="stop",
        )


def make_client(settings: Settings) -> LLMClient:
    """Use the fake in demo or no-key mode; otherwise construct Groq."""
    if settings.demo_mode or not settings.groq_api_key.get_secret_value():
        return FakeLLMClient(model=settings.groq_model)
    return GroqClient(settings)
