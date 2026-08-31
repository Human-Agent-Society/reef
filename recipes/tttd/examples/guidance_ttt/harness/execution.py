"""External execution-model adapters for the Guidance-TTT harness."""

from __future__ import annotations

import http.client
import json
import os
import time
import urllib.error
import urllib.request
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Protocol

from .state import LLMRequest, LLMResponse


@dataclass(frozen=True)
class ExecutionBackend:
    """Serializable executor settings. API-key values are deliberately excluded."""

    name: str
    model: str
    base_url: str
    api_key_env: str | None = None
    temperature: float = 1.0
    max_tokens: int | None = None
    timeout_s: float = 1_200.0
    concurrency: int = 16
    max_retries: int = 3
    retry_backoff_s: float = 2.0
    request_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.model.strip() or not self.base_url.strip():
            raise ValueError("executor name, model, and base_url must be non-empty")
        if self.temperature < 0:
            raise ValueError("executor temperature must be non-negative")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("executor max_tokens must be positive when set")
        if self.timeout_s <= 0 or self.concurrency < 1 or self.max_retries < 0 or self.retry_backoff_s < 0:
            raise ValueError("invalid executor timeout, concurrency, or retry settings")

    def safe_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_s": self.timeout_s,
            "concurrency": self.concurrency,
            "max_retries": self.max_retries,
            "retry_backoff_s": self.retry_backoff_s,
            "request_options": dict(self.request_options),
        }


def gpt_oss_120b_backend(
    *,
    base_url: str = "http://127.0.0.1:8000/v1",
    concurrency: int = 16,
    reasoning_effort: str = "high",
) -> ExecutionBackend:
    if reasoning_effort not in {"low", "medium", "high"}:
        raise ValueError("GPT-OSS reasoning_effort must be 'low', 'medium', or 'high'")
    return ExecutionBackend(
        name="gpt_oss_120b",
        model="openai/gpt-oss-120b",
        base_url=base_url,
        temperature=0.0,
        concurrency=concurrency,
        max_retries=0,
        request_options={"reasoning_effort": reasoning_effort},
    )


def openrouter_glm_5_2_backend(*, concurrency: int = 16) -> ExecutionBackend:
    return ExecutionBackend(
        name="openrouter_glm_5_2",
        model="z-ai/glm-5.2",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        temperature=1.0,
        concurrency=concurrency,
        max_retries=6,
        request_options={"reasoning": {"effort": "high", "exclude": False}},
    )


class ExecutionClient(Protocol):
    backend: ExecutionBackend

    def complete(self, request: LLMRequest) -> LLMResponse: ...


class OpenAICompatibleExecutionClient:
    """Small synchronous client suitable for the harness thread pool."""

    def __init__(self, backend: ExecutionBackend) -> None:
        self.backend = backend
        if backend.api_key_env:
            self._api_key = os.environ.get(backend.api_key_env, "").strip()
            if not self._api_key:
                raise ValueError(f"executor credential environment variable {backend.api_key_env!r} is not set")
        else:
            # OpenAI-compatible local servers usually require a syntactically
            # present bearer value even when authentication is disabled.
            self._api_key = "local-executor"

    def complete(self, request: LLMRequest) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "temperature": request.temperature,
            **self.backend.request_options,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        data, attempts, elapsed_s = self._post_json(payload)
        choices = data.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise RuntimeError("executor response has no choice")
        choice = choices[0]
        message = choice.get("message") or {}
        text = message.get("content") or choice.get("text") or ""
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("executor response has no final content")
        reasoning = (
            message.get("reasoning_content")
            or message.get("reasoning")
            or choice.get("reasoning_content")
            or choice.get("reasoning")
            or ""
        )
        metadata = {
            "provider": "openai_compatible",
            "backend": self.backend.name,
            "api_attempts": attempts,
            "api_elapsed_s": elapsed_s,
            **request.metadata,
        }
        if data.get("model"):
            metadata["api_response_model"] = data["model"]
        if data.get("provider"):
            metadata["api_response_provider"] = data["provider"]
        return LLMResponse(
            text=text,
            model=str(data.get("model") or request.model),
            finish_reason=str(choice.get("finish_reason") or ""),
            reasoning=str(reasoning),
            usage=dict(data.get("usage") or {}),
            metadata=metadata,
        )

    def _post_json(self, payload: dict[str, Any]) -> tuple[dict[str, Any], int, float]:
        started_at = time.monotonic()
        endpoint = self.backend.base_url.rstrip("/") + "/chat/completions"
        for attempt in range(self.backend.max_retries + 1):
            request = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode(),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.backend.timeout_s) as response:
                    data = json.loads(response.read().decode())
                if not isinstance(data, dict):
                    raise ValueError("executor returned a non-object JSON response")
                if data.get("error"):
                    raise _RetryableExecutorResponse("executor returned an API error object")
                return data, attempt + 1, time.monotonic() - started_at
            except urllib.error.HTTPError as exc:
                retryable = exc.code in {408, 429, 500, 502, 503, 504}
                if not retryable or attempt >= self.backend.max_retries:
                    body = exc.read().decode(errors="replace")[:1_000]
                    raise RuntimeError(f"executor request failed with HTTP {exc.code}: {body}") from exc
                self._backoff(attempt, exc.headers.get("Retry-After"))
            except (
                http.client.HTTPException,
                urllib.error.URLError,
                OSError,
                UnicodeError,
                ValueError,
                _RetryableExecutorResponse,
            ) as exc:
                if attempt >= self.backend.max_retries:
                    raise RuntimeError(
                        f"executor request failed after {attempt + 1} attempts: {type(exc).__name__}: {exc}"
                    ) from exc
                self._backoff(attempt)
        raise AssertionError("unreachable")

    def _backoff(self, attempt: int, retry_after: str | None = None) -> None:
        delay = self.backend.retry_backoff_s * (2**attempt)
        if retry_after is not None:
            with suppress(ValueError):
                delay = float(retry_after)
        time.sleep(max(0.0, delay))


class _RetryableExecutorResponse(RuntimeError):
    pass
