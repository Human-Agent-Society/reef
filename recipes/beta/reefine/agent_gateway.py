"""The loopback gateway Reefine's agent and its trial runs reach models through.

Reefine's gateway is the agent gateway (:mod:`reef.harness.gateway`), which
relays the model routes to the served model on the step's budget and serves
the agent's tools, with what a candidate harness reaches in its trials too:

- the multimodal routes (images, embeddings, speech, decisions) go to the
  recipe's multimodal provider, and the step record keeps their status only;
- ``GET /models?modality=`` lists the provider's models, fetched with its key.

The agent and every candidate harness it tries hold its one URL and nothing
else, so a candidate's own extensions reach the gateway as they would reach
Reef, with no credential of their own.
"""

from __future__ import annotations

import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler
from typing import Any

from recipes.beta.reefine.multimodal import MultimodalProvider
from reef.harness.gateway import AgentGateway, AgentTool, ModelCalls, refuse, relay
from reef.harness.rollouts.model_binding import ModelBinding
from reef.runtime.interfaces import MULTIMODAL_ROUTES

#: How much of a provider's refusal a trial reports back to the agent.
MAX_ERROR_CHARS = 600


class ReefineGateway(AgentGateway):
    """The agent gateway with the multimodal provider's routes, for one Reefine agent run.

    Values: the gateway's ``binding`` is the served model, and ``provider``
    is the multimodal gateway, ``None`` for none. ``provider_calls`` holds
    each multimodal call in call order.
    Purpose: Reefine's agent run starts one and stops it when the agent ends.
    """

    def __init__(
        self,
        served: ModelBinding,
        calls: ModelCalls,
        tools: Sequence[AgentTool],
        provider: MultimodalProvider | None,
        *,
        host: str = "127.0.0.1",
    ) -> None:
        super().__init__(served, calls, tools, host=host)
        self.provider = provider
        self.lock = threading.Lock()
        self.provider_calls: list[dict[str, Any]] = []

    def provider_calls_since(self, index: int) -> list[dict[str, Any]]:
        """The provider calls made after the first ``index``: what one trial's harness asked of the provider."""
        with self.lock:
            return list(self.provider_calls[index:])

    def provider_call_count(self) -> int:
        with self.lock:
            return len(self.provider_calls)

    def post(self, client: BaseHTTPRequestHandler, route: str, payload: dict[str, Any]) -> None:
        if route in MULTIMODAL_ROUTES:
            self.relay_provider_call(client, route, payload)
        else:
            super().post(client, route, payload)

    def get(self, client: BaseHTTPRequestHandler, route: str, query: str) -> None:
        if route != "/models":
            super().get(client, route, query)
            return
        modality = urllib.parse.parse_qs(query).get("modality", ["text"])[0]
        self.relay_models(client, modality)

    def relay_models(self, client: BaseHTTPRequestHandler, modality: str) -> None:
        """The provider's models of one output modality, so the agent picks a real one without holding the key."""
        if self.provider is None:
            refuse(client, 501, "the Reef deployment has no multimodal provider")
            return
        relay_models_request(client, self.provider.models_url(modality), self.provider.api_key)

    def relay_provider_call(self, client: BaseHTTPRequestHandler, route: str, payload: dict[str, Any]) -> None:
        if self.provider is None:
            refuse(client, 501, f"{route} needs a multimodal provider key on the Reef deployment")
            return
        upstream_path = self.provider.upstream_path(route)
        if upstream_path is None:
            refuse(client, 501, f"the deployment's {self.provider.preset.name} provider serves no {route}")
            return
        try:
            self.calls.spend()
        except RuntimeError as exc:
            refuse(client, 429, str(exc))
            return
        headers = {
            "Content-Type": "application/json",
            "Accept-Encoding": "identity",
            "Authorization": f"Bearer {self.provider.api_key}",
        }
        url = f"{self.provider.base_url}{upstream_path}"
        started = time.monotonic()
        status, body, _ = relay(client, url, headers, payload, timeout_s=300.0)
        call: dict[str, Any] = {"path": route, "model": payload.get("model"), "status": status}
        if status >= 400:
            call["error"] = body.decode("utf-8", errors="replace")[:MAX_ERROR_CHARS]
        with self.lock:
            self.provider_calls.append(call)
        seconds = round(time.monotonic() - started, 3)
        # The step record keeps what was asked of the provider and how it answered, never the media itself.
        self.calls.record({"source": "agent", **call, "seconds": seconds})
        outcome = f"{status}" if status < 400 else f"{status}: {call['error'][:160]}"
        self.calls.note("provider", f"{route} {call['model']} → {outcome} ({seconds:g} s)", failed=status >= 400)


def relay_models_request(client: BaseHTTPRequestHandler, url: str, api_key: str) -> None:
    """Relay the provider's model list, fetched with its key, whole."""
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(request, timeout=60) as upstream:
            status, body = upstream.status, upstream.read()
    except urllib.error.HTTPError as error:
        status, body = error.code, error.read()
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        refuse(client, 502, f"the provider did not answer: {error}")
        return
    client.send_response(status)
    client.send_header("Content-Type", "application/json")
    client.send_header("Content-Length", str(len(body)))
    client.end_headers()
    client.wfile.write(body)


__all__ = ["ReefineGateway"]
