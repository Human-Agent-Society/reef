"""HTTP request execution and inference proxy runtime construction.

This protocol adapter supports external providers and managed inference
endpoints without importing a native model framework. Request payloads and
streamed responses retain the provider's format.
"""

from __future__ import annotations

import asyncio
import email.utils
import json
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC
from typing import Any

import aiohttp
from aiohttp import ClientTimeout, TCPConnector

from reef.artifact.artifact import Artifact
from reef.core.config import config_option
from reef.core.errors import UpstreamLimit, UpstreamStatusError
from reef.core.served_models import ServedArtifact
from reef.inference.interfaces import (
    FULL_WEIGHT_NAME_PREFIX,
    InferenceArtifact,
    InferenceHandler,
    InferenceRuntime,
    InferenceStream,
)
from reef.inference.provider_protocol import provider_headers
from reef.inference.registry import RuntimeFactory, RuntimeRegistry, config_secret, config_string
from reef.inference.residency import ADAPTER_NAME_PREFIX

# The inference service binds its connector only while it handles a request.
inference_connector = ContextVar[tuple[asyncio.AbstractEventLoop, TCPConnector] | None](
    "inference_connector", default=None
)


def provider_session(timeout_s: float, *, auto_decompress: bool = True) -> aiohttp.ClientSession:
    scoped = inference_connector.get()
    connector = (
        scoped[1] if scoped is not None and scoped[0] is asyncio.get_running_loop() and not scoped[1].closed else None
    )
    # A pooled provider session must not replay provider cookies on redirect.
    return aiohttp.ClientSession(
        timeout=ClientTimeout(total=timeout_s),
        connector=connector,
        connector_owner=connector is None,
        cookie_jar=aiohttp.DummyCookieJar() if connector is not None else None,
        auto_decompress=auto_decompress,
    )


#: Hop-by-hop headers that describe one connection and must not be forwarded.
HOP_BY_HOP_HEADERS = frozenset(
    {
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
)

#: Provider API dialects the proxy can describe to the training side.
PROVIDER_APIS = ("openai", "responses", "anthropic")

#: The request header with the engine-side name of the weights that a request selects. A remote inference
#: component that keeps each request on its weights admits the request only while it serves that name.
SERVED_MODEL_HEADER = "x-reef-served-model"

#: The ``error.metadata.limit_source`` of OpenRouter's temporary limit on the requests of a key in flight.
OPENROUTER_IN_FLIGHT_LIMIT = "openrouter_in_flight_budget"
#: The ``error.metadata.limit_source`` values of OpenRouter's credits and key limits, which the same request cannot
#: pass again on its own.
OPENROUTER_PAYMENT_LIMITS = ("openrouter_credits", "openrouter_key_limit")
#: The ``error.metadata.reason`` of a request that costs more than OpenRouter's budget allows, alone or with
#: ``limit_source`` ``openrouter_credits``.
OPENROUTER_OVER_BUDGET = "weight_exceeds_budget"
#: The ``error.metadata.limit_source`` values of OpenRouter's 403 for a spending cap: a guardrail's, a workspace's or
#: the key's budget.
OPENROUTER_BUDGET_LIMITS = ("openrouter_guardrail_budget", "openrouter_workspace_budget", "openrouter_key_limit")
#: The error ``code`` and ``type`` with which OpenAI answers 429 for an account whose quota ran out.
OPENAI_QUOTA_CODE = "insufficient_quota"

# -- Provider refusals --------------------------------------------------------


def retry_after_seconds(value: str | None, now: float) -> float | None:
    """The wait in seconds that the ``Retry-After`` header value ``value`` asks for at the epoch time ``now``.

    ``value`` is a number of whole seconds or an HTTP date (RFC 9110). A date that has passed asks for no wait (0).
    None for no value and for any other value, also a number or a date out of range.
    """
    if value is None:
        return None
    text = value.strip()
    if text.isascii() and text.isdigit():
        seconds = float(text)
        return seconds if math.isfinite(seconds) else None
    try:
        moment = email.utils.parsedate_to_datetime(text)
        # An HTTP date is in GMT; a date that names no zone reads as one.
        epoch_seconds = (moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)).timestamp()
    except (TypeError, ValueError, OverflowError):
        return None
    return max(0.0, epoch_seconds - now)


def upstream_limit(status: int, body: str, *, retry_after: float | None) -> UpstreamLimit | None:
    """The limit that refused a request with ``status``, the error body ``body`` and the ``Retry-After`` wait
    ``retry_after``. It reads structured fields only, never a message.

    - A 429 is ``RATE``; one whose error ``code`` or ``type`` is ``OPENAI_QUOTA_CODE`` is ``PAYMENT``.
    - A 402 whose ``error.metadata.limit_source`` is ``OPENROUTER_IN_FLIGHT_LIMIT`` is ``RATE``.
    - A 402 whose ``limit_source`` is in ``OPENROUTER_PAYMENT_LIMITS``, or whose ``error.metadata.reason`` is
      ``OPENROUTER_OVER_BUDGET``, is ``PAYMENT``, also with a ``Retry-After``.
    - Any other 402 is ``RATE`` with a ``Retry-After`` wait, else ``PAYMENT``, whatever its body holds.
    - A 403 whose ``limit_source`` is in ``OPENROUTER_BUDGET_LIMITS`` is ``PAYMENT``, also with a ``Retry-After``;
      any other 403 is None.
    - Any other status is None.
    """
    if status not in (402, 403, 429):
        return None
    try:
        reply = json.loads(body)
    except ValueError:
        reply = None
    error = reply.get("error") if isinstance(reply, dict) else None
    error = error if isinstance(error, dict) else {}
    if status == 429:
        quota = OPENAI_QUOTA_CODE in (error.get("code"), error.get("type"))
        return UpstreamLimit.PAYMENT if quota else UpstreamLimit.RATE
    metadata = error.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    source = metadata.get("limit_source")
    source = source if isinstance(source, str) else None
    if status == 403:
        return UpstreamLimit.PAYMENT if source in OPENROUTER_BUDGET_LIMITS else None
    if source == OPENROUTER_IN_FLIGHT_LIMIT:
        return UpstreamLimit.RATE
    if source in OPENROUTER_PAYMENT_LIMITS or metadata.get("reason") == OPENROUTER_OVER_BUDGET or retry_after is None:
        return UpstreamLimit.PAYMENT
    return UpstreamLimit.RATE


# -- Request headers ----------------------------------------------------------


class RequestHeadersFactory(ABC):
    """Produce request headers for one artifact-bound upstream call."""

    @abstractmethod
    def headers(self, artifact: InferenceArtifact, path: str) -> Mapping[str, str]:
        """Return identity and authentication headers for the selected route."""

    def external_headers(self, path: str) -> Mapping[str, str]:
        """Headers for an external model call with no Reef artifact identity."""
        return {}


def artifact_identity_headers(artifact: InferenceArtifact) -> dict[str, str]:
    """Reef identity headers for one selected, optionally materialized artifact.

    The artifact ID is always present, and the local path when the artifact has one. ``SERVED_MODEL_HEADER``
    carries ``served_model`` when that is a full-weight or adapter name that Reef gave the engines; a provider's
    own model name stays in the request body only.
    """
    headers = {"x-reef-artifact-id": artifact.artifact_id}
    if artifact.served_model is not None and artifact.served_model.startswith(
        (FULL_WEIGHT_NAME_PREFIX, ADAPTER_NAME_PREFIX)
    ):
        headers[SERVED_MODEL_HEADER] = artifact.served_model
    if artifact.local_path is not None:
        headers["x-reef-artifact-path"] = str(artifact.local_path)
    return headers


class ArtifactRequestHeaders(RequestHeadersFactory):
    """Attach the selected Reef artifact identity."""

    def headers(self, artifact: InferenceArtifact, path: str) -> Mapping[str, str]:
        return artifact_identity_headers(artifact)


class ProviderRequestHeaders(RequestHeadersFactory):
    """Attach artifact identity and provider authentication.

    Values: the object holds ``api_key``, a non-empty provider credential.
    Purpose: an :class:`HttpInferenceHandler` that calls a provider asks it
    for the headers of each call. ``headers(artifact, path)`` returns the
    identity headers of ``artifact`` followed by
    :func:`~reef.inference.provider_protocol.provider_headers` for ``path``.
    """

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("api_key must be non-empty")
        self._api_key = api_key

    def headers(self, artifact: InferenceArtifact, path: str) -> Mapping[str, str]:
        return {**artifact_identity_headers(artifact), **self.external_headers(path)}

    def external_headers(self, path: str) -> Mapping[str, str]:
        return provider_headers(path, self._api_key)


def provider_request_headers(api_key: str) -> RequestHeadersFactory:
    """Build provider-native authentication with Reef artifact identity."""
    return ProviderRequestHeaders(api_key)


# -- HTTP handler -------------------------------------------------------------


class HttpInferenceHandler(InferenceHandler):
    """POST native inference requests to an HTTP provider.

    The handler holds a non-empty upstream URL string after removing trailing slashes, a
    ``RequestHeadersFactory``, a positive timeout in seconds, and an error label string. The constructor
    uses ``ArtifactRequestHeaders`` when no factory is supplied. The factory supplies artifact identity,
    provider authentication, or deployment-specific headers. The error label identifies the upstream in
    errors reported to the caller.

    ``inference`` returns the provider's JSON object. A provider status of 400 or more raises
    ``UpstreamStatusError`` with that status, the provider's ``Retry-After`` wait (``retry_after_seconds``) and the
    limit that its reply names (``upstream_limit``); the handler sends each request once. A successful status
    whose body is not one JSON object raises it with status 502, since the provider failed and not the caller.
    ``inference_stream`` owns its provider session until it returns: on any failure or cancellation before then it
    closes the session and re-raises. The returned ``InferenceStream`` then owns the session, and its ``close``
    releases it.
    In the inference service, sessions borrow the service connector.
    """

    def __init__(
        self,
        upstream_url: str,
        *,
        request_headers: RequestHeadersFactory | None = None,
        timeout_s: float = 300.0,
        error_label: str = "inference upstream",
    ) -> None:
        upstream = upstream_url.rstrip("/")
        if not upstream:
            raise ValueError("upstream_url must be non-empty")
        if not timeout_s > 0:
            raise ValueError("timeout_s must be positive")
        self._upstream_url = upstream
        self._request_headers = request_headers if request_headers is not None else ArtifactRequestHeaders()
        self._timeout_s = timeout_s
        self._error_label = error_label

    @classmethod
    def from_config(
        cls, upstream_url: str, *, model_path: str, timeout_s: float, **config: Any
    ) -> HttpInferenceHandler:
        """Build a plain HTTP handler; model_path is for tokenizer-aware subclasses."""
        if config:
            raise ValueError(f"default HTTP inference handler does not accept config keys: {sorted(config)}")
        return cls(upstream_url, timeout_s=timeout_s)

    def post_arguments(self, artifact: InferenceArtifact | None, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """The keyword arguments of one ``session.post`` call: the upstream URL of ``path``, ``payload`` as JSON,
        and the headers for ``artifact``, or only the provider's headers without one."""
        headers = (
            self._request_headers.external_headers(path)
            if artifact is None
            else self._request_headers.headers(artifact, path)
        )
        return {
            "url": f"{self._upstream_url}{path}",
            "json": payload,
            "headers": dict(headers),
        }

    def _status_error(self, status: int, body: str, retry_after: str | None = None) -> UpstreamStatusError:
        # Only the wait of the header leaves the reply: no other provider header reaches the error.
        wait_seconds = retry_after_seconds(retry_after, time.time())
        return UpstreamStatusError(
            f"{self._error_label} returned {status}: {body[:400]}",
            status=status,
            retry_after_seconds=wait_seconds,
            limit=upstream_limit(status, body, retry_after=wait_seconds),
        )

    async def inference(self, artifact: InferenceArtifact, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await self.post_json(path, payload, artifact=artifact)

    async def post_json(
        self, path: str, payload: dict[str, Any], *, artifact: InferenceArtifact | None = None
    ) -> dict[str, Any]:
        """POST to the provider. External models omit the Reef artifact."""
        from aiohttp import ContentTypeError

        async with (
            provider_session(self._timeout_s) as session,
            session.post(**self.post_arguments(artifact, path, payload)) as response,
        ):
            if response.status >= 400:
                # An error body that is not UTF-8 keeps the provider's status, as a stream's error body does.
                body = (await response.read()).decode(errors="replace")
                raise self._status_error(response.status, body, response.headers.get("Retry-After"))
            try:
                value = await response.json()
            except (ContentTypeError, ValueError) as exc:
                # A successful status with an unreadable body is the provider's failure, not the caller's.
                raise UpstreamStatusError(f"{self._error_label} returned a body that is not JSON", status=502) from exc
        if not isinstance(value, dict):
            raise UpstreamStatusError(f"{self._error_label} response must be a JSON object", status=502)
        return value

    async def inference_stream(
        self, artifact: InferenceArtifact, path: str, payload: dict[str, Any]
    ) -> InferenceStream:
        return await self.post_stream(path, payload, artifact=artifact)

    async def post_stream(
        self, path: str, payload: dict[str, Any], *, artifact: InferenceArtifact | None = None
    ) -> InferenceStream:
        """Open the provider stream. Its owner closes it after consuming or cancelling it."""
        # The session outlives this call: the stream owns it until closed.
        session = provider_session(self._timeout_s, auto_decompress=False)
        try:
            response = await session.post(**self.post_arguments(artifact, path, payload))
        except BaseException:
            # Cancelled while waiting for the provider's headers, too: nothing else owns this session yet.
            await session.close()
            raise

        async def close() -> None:
            response.close()
            await session.close()

        if response.status >= 400:
            retry_after = response.headers.get("Retry-After")
            try:
                body = (await response.read()).decode(errors="replace")
            finally:
                await close()
            raise self._status_error(response.status, body, retry_after)
        return InferenceStream(
            status=response.status,
            headers={
                name: value for name, value in response.headers.items() if name.lower() not in HOP_BY_HOP_HEADERS
            },
            chunks=response.content.iter_any(),
            close=close,
        )


# -- Proxy runtime ------------------------------------------------------------


class InferenceProxyRuntime(InferenceRuntime):
    """Inference runtime that wraps an HTTP inference handler.

    Returned by the ``inference_proxy`` runtime type for no-update recipes.
    Holds an HttpInferenceHandler with provider-native auth
    and exposes it via inference_handler, which forwards each request to the
    provider in its native format. It binds no weights: nothing ties a Reef
    artifact to what the provider serves under ``model_path``. So
    ``served_artifact`` gives no route for any artifact, and
    ``activate_checkpoint`` refuses every snapshot whose weights the runtime
    would load. A snapshot without a weight component, such as a record-only or
    harness snapshot, routes to the provider's model (``endpoint_model_name``).
    """

    def __init__(
        self,
        *,
        model_path: str = "",
        base_url: str,
        api_key: str | None = None,
        api: str = "openai",
        inference_timeout_s: float = 300.0,
    ) -> None:
        super().__init__(base_url=base_url, inference_timeout_s=inference_timeout_s)
        if api not in PROVIDER_APIS:
            raise ValueError(f"inference proxy api must be one of {PROVIDER_APIS}, got {api!r}")
        self._model_path = model_path
        self._api_key = api_key
        self._api = api
        self._inference_handler = HttpInferenceHandler(
            self.base_url,
            request_headers=ProviderRequestHeaders(api_key) if api_key else None,
            timeout_s=self.inference_timeout_s,
            error_label="inference provider",
        )

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def provider_key(self) -> str | None:
        return self._api_key

    @property
    def api(self) -> str:
        """The provider's API dialect (``openai``, ``responses``, or
        ``anthropic``). The proxy forwards whatever path a client calls; this
        tells the training side which dialect to speak when it calls the model
        itself."""
        return self._api

    @property
    def inference_handler(self) -> InferenceHandler:
        return self._inference_handler

    def activate_checkpoint(self, artifact: Artifact, *, scenario: str | None = None) -> str:
        """Refuse the snapshot with ``ValueError``: the proxy loads no weights, so its provider's model would answer
        under the snapshot's artifact ID. The refusal does not change on a retry, so a startup fails for good."""
        raise ValueError(
            f"an inference proxy loads no weights, so it cannot serve the weights of artifact {artifact.artifact_id}; "
            "serve weight snapshots with a runtime that loads them"
        )

    def served_artifact(self, artifact_id: str, scenario: str) -> ServedArtifact | None:
        """None for every artifact: the proxy binds no artifact's weights to the provider's model."""
        return None

    def endpoint_model_name(self, scenario: str) -> str | None:
        """The provider's model (``served_model_name``, by default ``model_path``) for a snapshot without weights; None
        when the proxy names no model."""
        return self.served_model_name(scenario) or None


@dataclass(frozen=True)
class InferenceProxySettings:
    """Connection configuration owned by the inference proxy adapter."""

    base_url: str = config_option("", help="Inference provider base URL.")
    api: str = config_option("openai", help="Provider API format.")
    api_key: str | None = config_option(None, help="Provider credential.")
    api_key_env: str | None = config_option(None, help="Environment variable containing the provider credential.")
    timeout_s: float = config_option(300.0, help="Inference request timeout in seconds.")

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("runtime.base_url must be a non-empty string")
        if self.timeout_s <= 0:
            raise ValueError("runtime.timeout_s must be positive")


class InferenceProxyRuntimeFactory(RuntimeFactory):
    """Build an :class:`InferenceProxyRuntime` from a runtime config section."""

    kind = "inference_proxy"

    def config_type(self) -> type:
        return InferenceProxySettings

    def __call__(
        self,
        config: Mapping[str, Any],
        model_path: str,
        recipe_config: Mapping[str, Any],
        environ: Mapping[str, str],
    ) -> InferenceRuntime:
        return InferenceProxyRuntime(
            model_path=model_path,
            base_url=config_string(config, "base_url"),
            api_key=config_secret(config, environ, "api_key", "api_key_env"),
            api=config["api"],
            inference_timeout_s=config["timeout_s"],
        )


def resolve_proxy_runtime(
    values: Mapping[str, str],
    runtime: InferenceRuntime | None,
) -> InferenceRuntime | None:
    """Prefer an injected runtime, else build a proxy from environment values.

    Construction goes through the runtime registry's ``inference_proxy`` kind,
    the same factory the YAML config path uses.
    """
    if runtime is not None:
        return runtime
    base_url = _env_value(values, "REEF_UPSTREAM_URL")
    if base_url is None:
        return None
    config: dict[str, object] = {"type": "inference_proxy", "base_url": base_url}
    # REEF_API_KEY is the pre-rename spelling; drop it once no deployment sets it.
    api_key = _env_value(values, "REEF_UPSTREAM_API_KEY") or _env_value(values, "REEF_API_KEY")
    if api_key is not None:
        config["api_key"] = api_key
    timeout_raw = _env_value(values, "REEF_INFERENCE_TIMEOUT_S")
    if timeout_raw is not None:
        config["timeout_s"] = float(timeout_raw)
    built = RuntimeRegistry().build(config, model_path=_env_value(values, "REEF_MODEL_PATH") or "", environ=values)
    if not isinstance(built, InferenceRuntime):
        raise TypeError("inference proxy factory must return an InferenceRuntime")
    return built


def _env_value(values: Mapping[str, str], name: str) -> str | None:
    return values.get(name, "").strip() or None


__all__ = [
    "PROVIDER_APIS",
    "SERVED_MODEL_HEADER",
    "ArtifactRequestHeaders",
    "HttpInferenceHandler",
    "InferenceProxyRuntime",
    "InferenceProxyRuntimeFactory",
    "InferenceProxySettings",
    "ProviderRequestHeaders",
    "RequestHeadersFactory",
    "artifact_identity_headers",
    "provider_request_headers",
    "resolve_proxy_runtime",
    "retry_after_seconds",
    "upstream_limit",
]
