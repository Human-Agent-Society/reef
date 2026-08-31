"""Runtime for proxy deployments that do not train weights."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.runtime.base import InferenceRuntime
from reef.runtime.inference import (
    HttpInferenceBackend,
    InferenceBackend,
    RequestHeadersFactory,
    default_artifact_request_headers,
    provider_request_headers,
)
from reef.runtime.registry import RuntimeFactory, config_secret, config_string, register_runtime_kind

#: Provider API dialects the proxy can describe to the training side.
PROVIDER_APIS = ("openai", "anthropic")


class InferenceProxyRuntime(InferenceRuntime):
    """Inference runtime that wraps an HTTP inference backend.

    Returned by the ``inference_proxy`` runtime type for no-update recipes.
    Holds an HttpInferenceBackend with provider-native auth
    and exposes it via inference_backend. Does not implement training
    lifecycle methods.
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
        super().__init__(
            base_url=base_url,
            inference_timeout_s=inference_timeout_s,
        )
        if api not in PROVIDER_APIS:
            raise ValueError(f"inference proxy api must be one of {PROVIDER_APIS}, got {api!r}")
        self._model_path = model_path
        self._api_key = api_key
        self._api = api
        request_headers: RequestHeadersFactory = default_artifact_request_headers
        if api_key:
            request_headers = provider_request_headers(api_key)
        self._inference_backend = HttpInferenceBackend(
            self.base_url,
            request_headers=request_headers,
            timeout_s=self.inference_timeout_s,
            error_label="inference provider",
        )

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def api_key(self) -> str | None:
        return self._api_key

    @property
    def api(self) -> str:
        """The provider's API dialect (``openai`` or ``anthropic``). The proxy
        forwards whatever path a client calls; this tells the training side
        which dialect to speak when it calls the model itself."""
        return self._api

    @property
    def inference_backend(self) -> InferenceBackend:
        return self._inference_backend


@register_runtime_kind
class InferenceProxyRuntimeFactory(RuntimeFactory):
    """Build an :class:`InferenceProxyRuntime` from a runtime config section."""

    kind = "inference_proxy"

    def __call__(
        self,
        config: Mapping[str, Any],
        model_path: str,
        recipe_config: Mapping[str, Any],
        environ: Mapping[str, str],
    ) -> InferenceRuntime:
        api_key = config_secret(config, environ, "api_key", "api_key_env")
        return InferenceProxyRuntime(
            model_path=model_path,
            base_url=config_string(config, "base_url"),
            api_key=api_key,
            api=str(config.get("api", "openai")),
            inference_timeout_s=float(config.get("timeout_s", 300.0)),
        )
