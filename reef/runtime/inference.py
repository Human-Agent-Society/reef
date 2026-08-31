"""Inference backend: execute one request against one artifact handle.

``InferenceRuntime`` (``base.py``) owns the lifecycle and *composes* an
``InferenceBackend``; this module is the backend contract, its HTTP
implementation, and the streaming wrapper the service forwards.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any, Protocol

from reef.artifact.artifact import Artifact
from reef.core.errors import ReefError


class UpstreamStatusError(ReefError):
    """An upstream service answered an inference request with an error status.

    Carries ``status`` so the service can hand the caller the upstream's own
    4xx and message rather than collapsing both into an opaque 500. That
    distinction is load-bearing: agents correct themselves from these bodies
    (shrinking ``max_tokens`` when the engine reports a context overflow, for
    instance), and an opaque 500 leaves them retrying the identical request
    until they give up.
    """

    def __init__(self, message: str, *, status: int) -> None:
        super().__init__(message)
        self.status = status


class InferenceBackend(ABC):
    """Execute inference for a selected artifact without implicitly materializing it."""

    @abstractmethod
    async def inference(
        self,
        artifact: Artifact,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Return the provider response for one native request payload."""

    async def inference_stream(
        self,
        artifact: Artifact,
        path: str,
        payload: dict[str, Any],
    ) -> InferenceStream:
        """Return a stream for one native request payload.

        Custom backends that only implement buffered inference keep working: the
        default implementation exposes their JSON response as one chunk. HTTP
        backends override this method to preserve provider-native streaming.
        """
        value = await self.inference(artifact, path, payload)

        async def chunks() -> AsyncIterator[bytes]:
            yield json.dumps(value, ensure_ascii=False).encode()

        return InferenceStream(
            status=200,
            headers={"Content-Type": "application/json"},
            chunks=chunks(),
        )


class InferenceBackendFactory(Protocol):
    """Construct a deployment-selected backend for one serving endpoint."""

    def __call__(
        self,
        upstream_url: str,
        *,
        model_path: str,
        timeout_s: float,
        **config: Any,
    ) -> InferenceBackend: ...


class InferenceStream:
    """One open provider response whose bytes can be forwarded incrementally."""

    def __init__(
        self,
        *,
        status: int,
        headers: Mapping[str, str],
        chunks: AsyncIterator[bytes],
        close: Callable[[], Awaitable[None]] | None = None,
        record_response: Mapping[str, Any] | None = None,
        record_response_pending: bool = False,
    ) -> None:
        self.status = status
        self.headers = dict(headers)
        self.chunks = chunks
        self._close = close
        self._closed = False
        self.record_response = None if record_response is None else dict(record_response)
        # Some custom backends can only construct their exact, provider-neutral
        # recording response after the upstream stream reaches its terminal
        # event. RequestService keeps durable admission open for those streams
        # and validates the completed capture before accepting the record.
        self.record_response_pending = record_response_pending

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._close is not None:
            await self._close()


class RequestHeadersFactory(Protocol):
    """Produce request headers for one artifact-bound upstream call."""

    def __call__(self, artifact: Artifact, path: str) -> Mapping[str, str]: ...


def artifact_identity_headers(artifact: Artifact) -> dict[str, str]:
    """Reef identity headers for one selected, optionally materialized artifact."""
    headers = {"x-reef-artifact-version": artifact.ref.version}
    if artifact.local_path is not None:
        headers["x-reef-artifact-path"] = str(artifact.local_path)
    return headers


def default_artifact_request_headers(artifact: Artifact, path: str = "") -> Mapping[str, str]:
    """The default :data:`RequestHeadersFactory`: identity headers for every path."""
    return artifact_identity_headers(artifact)


def provider_request_headers(api_key: str) -> RequestHeadersFactory:
    """Build a RequestHeadersFactory that adds provider-native auth.

    Reef artifact identity headers are always included. For Anthropic
    (/v1/messages) the api key is sent as x-api-key + anthropic-version;
    for OpenAI-compatible routes it is sent as a Bearer token.
    """
    if not api_key:
        raise ValueError("api_key must be non-empty")

    def headers_for(artifact: Artifact, path: str) -> Mapping[str, str]:
        headers = artifact_identity_headers(artifact)
        if path == "/v1/messages":
            headers["x-api-key"] = api_key
            headers["anthropic-version"] = "2023-06-01"
        else:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    return headers_for


class HttpInferenceBackend(InferenceBackend):
    """POST native inference requests to an HTTP provider.

    The request headers are produced by a callable, so callers can inject
    any combination of reef artifact identity headers, provider auth,
    or custom headers without subclassing.
    """

    def __init__(
        self,
        upstream_url: str,
        *,
        request_headers: RequestHeadersFactory = default_artifact_request_headers,
        timeout_s: float = 300.0,
        error_label: str = "inference upstream",
    ) -> None:
        upstream = upstream_url.rstrip("/")
        if not upstream:
            raise ValueError("upstream_url must be non-empty")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._upstream_url = upstream
        self._request_headers = request_headers
        self._timeout_s = timeout_s
        self._error_label = error_label

    async def inference(
        self,
        artifact: Artifact,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        from aiohttp import ClientSession, ClientTimeout

        async with (
            ClientSession(timeout=ClientTimeout(total=self._timeout_s)) as session,
            session.post(
                f"{self._upstream_url}{path}",
                json=payload,
                headers=dict(self._request_headers(artifact, path)),
            ) as response,
        ):
            body = await response.text()
            if response.status >= 400:
                raise UpstreamStatusError(
                    f"{self._error_label} returned {response.status}: {body[:400]}",
                    status=response.status,
                )
            value = await response.json()
        if not isinstance(value, dict):
            raise TypeError(f"{self._error_label} response must be a JSON object")
        return value

    async def inference_stream(
        self,
        artifact: Artifact,
        path: str,
        payload: dict[str, Any],
    ) -> InferenceStream:
        from aiohttp import ClientSession, ClientTimeout

        session = ClientSession(
            timeout=ClientTimeout(total=self._timeout_s),
            auto_decompress=False,
        )
        try:
            response = await session.post(
                f"{self._upstream_url}{path}",
                json=payload,
                headers=dict(self._request_headers(artifact, path)),
            )
        except Exception:
            await session.close()
            raise

        if response.status >= 400:
            try:
                body = (await response.read()).decode(errors="replace")
            finally:
                response.close()
                await session.close()
            raise UpstreamStatusError(
                f"{self._error_label} returned {response.status}: {body[:400]}",
                status=response.status,
            )

        excluded_headers = {
            "connection",
            "content-length",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailer",
            "transfer-encoding",
            "upgrade",
        }
        response_headers = {
            name: value for name, value in response.headers.items() if name.lower() not in excluded_headers
        }

        async def close() -> None:
            response.close()
            await session.close()

        return InferenceStream(
            status=response.status,
            headers=response_headers,
            chunks=response.content.iter_any(),
            close=close,
        )


def build_http_inference_backend(
    upstream_url: str,
    *,
    model_path: str,
    timeout_s: float,
    **config: Any,
) -> InferenceBackend:
    """Default factory; ``model_path`` is reserved for tokenizer-aware backends."""

    if config:
        raise ValueError(f"default HTTP inference backend does not accept config keys: {sorted(config)}")
    return HttpInferenceBackend(upstream_url, timeout_s=timeout_s)


__all__ = [
    "HttpInferenceBackend",
    "InferenceBackend",
    "InferenceBackendFactory",
    "InferenceStream",
    "RequestHeadersFactory",
    "UpstreamStatusError",
    "artifact_identity_headers",
    "build_http_inference_backend",
    "default_artifact_request_headers",
    "provider_request_headers",
]
