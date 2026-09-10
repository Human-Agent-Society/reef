"""Resolve a harness scenario's model configuration through its trusted platform.

The platform returns a version-scoped proxy credential, not the user's provider
key. A failed resolution is never treated as permission to use managed models.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from reef.artifact.artifact import Artifact
from reef.core.errors import ReefError
from reef.runtime.adapters.inference_proxy import InferenceProxyRuntime
from reef.runtime.base import InferenceRuntime
from reef.runtime.inference import InferenceBackend, InferenceStream


class ProviderResolutionError(ReefError):
    """The configured platform could not authorize a model binding."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


@dataclass(frozen=True)
class ProviderSnapshot:
    mode: str
    version: str
    runtime: InferenceRuntime


@dataclass(frozen=True)
class ScenarioProviderResolver:
    """A deployment-scoped control-plane client configured by the operator."""

    url: str
    token: str = field(repr=False)

    def __post_init__(self) -> None:
        parsed = urlparse(self.url)
        if parsed.scheme not in ("https", "http") or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("REEF_BYOK_RESOLVER_URL must be an HTTP(S) URL without credentials")
        if not self.token:
            raise ValueError("REEF_BYOK_RESOLVER_TOKEN is required")

    def resolve(self, scenario: str, managed: InferenceRuntime | None, version: str | None = None) -> ProviderSnapshot:
        payload = {"scenario": scenario}
        if version is not None:
            payload["version"] = version
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode(),
            headers={"content-type": "application/json", "authorization": f"Bearer {self.token}"},
            method="POST",
        )
        try:
            with urllib.request.build_opener(_NoRedirect()).open(request, timeout=15) as response:
                answer = json.load(response)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise ProviderResolutionError("Cannot resolve the scenario model provider; no managed fallback") from exc
        if not isinstance(answer, Mapping) or not isinstance(answer.get("version"), str):
            raise ProviderResolutionError("Invalid scenario provider response")
        if version is not None and answer["version"] != version:
            raise ProviderResolutionError("Scenario provider configuration changed")
        if answer.get("mode") == "managed":
            if managed is None:
                raise ProviderResolutionError("No managed model is configured for this scenario")
            return ProviderSnapshot("managed", answer["version"], managed)
        if answer.get("mode") != "byok" or answer.get("api") not in ("openai", "anthropic"):
            raise ProviderResolutionError("Invalid scenario provider mode or protocol")
        for name in ("base_url", "api_key", "model"):
            if not isinstance(answer.get(name), str) or not answer[name]:
                raise ProviderResolutionError("Incomplete scenario provider binding")
        # Only the trusted platform can receive the scoped credential.
        base = urlparse(answer["base_url"])
        origin = urlparse(self.url)
        if (base.scheme, base.netloc) != (origin.scheme, origin.netloc) or base.username or base.password:
            raise ProviderResolutionError("Provider proxy must use the resolver's origin")
        runtime = InferenceProxyRuntime(
            base_url=answer["base_url"], api_key=answer["api_key"], model_path=answer["model"], api=answer["api"]
        )
        return ProviderSnapshot("byok", answer["version"], runtime)


class _ProviderInferenceBackend(InferenceBackend):
    def __init__(self, runtime: ScenarioProviderRuntime) -> None:
        self._runtime = runtime

    async def inference(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        backend = await asyncio.to_thread(self._runtime.request_backend)
        return await backend.inference(artifact, path, payload)

    async def inference_stream(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> InferenceStream:
        backend = await asyncio.to_thread(self._runtime.request_backend)
        return await backend.inference_stream(artifact, path, payload)


class ScenarioProviderRuntime(InferenceRuntime):
    """One scenario's live configuration; each request or step freezes a snapshot."""

    def __init__(self, resolver: ScenarioProviderResolver, scenario: str, managed: InferenceRuntime | None) -> None:
        super().__init__(base_url=resolver.url)
        self.resolver = resolver
        self.scenario = scenario
        self.managed = managed
        self._backend = _ProviderInferenceBackend(self)

    def snapshot(self, version: str | None = None) -> ProviderSnapshot:
        return self.resolver.resolve(self.scenario, self.managed, version)

    def request_backend(self, version: str | None = None) -> InferenceBackend:
        return self.snapshot(version).runtime.inference_backend

    @property
    def inference_backend(self) -> InferenceBackend:
        return self._backend
