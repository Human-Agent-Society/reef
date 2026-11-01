"""How a separate inference service picks the handler of each ``served_models`` row."""

from __future__ import annotations

from reef.core.served_models import ServedArtifact
from reef.inference.http import InferenceProxyRuntime
from reef.runtime.interfaces import InferenceHandler
from reef.service.engine_clients import DeploymentEngineClients


class EndpointsRuntime(InferenceProxyRuntime):
    """A runtime that serves artifacts at several URLs, each with its own handler."""

    def __init__(self, urls: tuple[str, ...]) -> None:
        super().__init__(base_url="https://engine.example/base", inference_timeout_s=5)
        self.handlers = {
            url: InferenceProxyRuntime(base_url=url, api_key="engine-key").inference_handler for url in urls
        }

    def handler_for(self, url: str) -> InferenceHandler | None:
        return self.handlers.get(url)


def test_each_url_that_the_runtime_serves_uses_its_handler_and_other_urls_a_proxy() -> None:
    runtime = EndpointsRuntime(("https://engine.example/release-1",))
    clients = DeploymentEngineClients(runtime, provider_key=None, timeout_seconds=5)

    served = ServedArtifact("a1", "https://engine.example/release-1", "model")
    assert clients.handler(served) is runtime.handlers["https://engine.example/release-1"]

    elsewhere = ServedArtifact("a2", "https://provider.example/v1", "model")
    proxy = clients.handler(elsewhere)
    assert proxy not in runtime.handlers.values()
    assert clients.handler(elsewhere) is proxy


def test_by_default_only_the_base_url_uses_the_runtime_handler() -> None:
    runtime = InferenceProxyRuntime(base_url="https://engine.example/base", inference_timeout_s=5)
    assert runtime.handler_for("https://engine.example/base") is not None
    assert runtime.handler_for("https://engine.example/other") is None
