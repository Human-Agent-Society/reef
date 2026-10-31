"""How a separate inference service reaches the engine of each ``served_models`` row: as control reaches it.

``DeploymentEngineClients`` holds the deployment's serving runtime, built from the same configuration as control's
and never opened, so it changes no engine. A row at a URL that the runtime serves uses the runtime's own handler for
that URL (``InferenceRuntime.handler_for``), with its dialect and its credentials. A row at another URL is a
scenario's own model (its model config): it uses a provider proxy with the deployment's provider key, as control's
runtime of that scenario does.
"""

from __future__ import annotations

from reef.core.served_models import ServedArtifact
from reef.inference.http import InferenceProxyRuntime
from reef.runtime.interfaces import EngineClients, InferenceHandler, InferenceRuntime


class DeploymentEngineClients(EngineClients):
    """The engines of a separate inference service.

    Values: ``runtime`` is the deployment's serving runtime, or None for a deployment that serves only scenarios'
    own models; ``provider_key`` is the deployment's provider key, or None; ``timeout_seconds`` bounds each call
    of a proxy handler; ``proxies`` holds the proxy handler of each other URL, made on first use.

    Purpose: ``reef.service.components`` gives it to the replica of the inference service.
    """

    def __init__(self, runtime: InferenceRuntime | None, *, provider_key: str | None, timeout_seconds: float) -> None:
        self.runtime = runtime
        self.provider_key = provider_key
        self.timeout_seconds = timeout_seconds
        self.proxies: dict[str, InferenceHandler] = {}

    def handler(self, served: ServedArtifact) -> InferenceHandler:
        if self.runtime is not None:
            own = self.runtime.handler_for(served.url)
            if own is not None:
                return own
        found = self.proxies.get(served.url)
        if found is None:
            found = InferenceProxyRuntime(
                base_url=served.url, api_key=self.provider_key, inference_timeout_s=self.timeout_seconds
            ).inference_handler
            self.proxies[served.url] = found
        return found


__all__ = ["DeploymentEngineClients"]
