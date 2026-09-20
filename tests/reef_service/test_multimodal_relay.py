"""Multimodal calls: relayed by the scenario's recipe to its gateway, unrecorded; reefine configures the gateway."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from reef.artifact import InMemoryRepositoryBackend
from reef.dispatcher import Dispatcher
from reef.inference.http import InferenceProxyRuntime
from reef.inference.model_config import ModelConfig
from reef.recipe import Recipe
from reef.recipe.errors import RecipeConfigError
from reef.recipe.reefine.multimodal import PRESETS, MultimodalProvider, MultimodalSettings, ProviderRelay
from reef.runtime.interfaces import MultimodalRelay
from reef.service.app import create_app
from reef.storage.sqlite import SQLiteScenarioStorage

AUDIO_BYTES = bytes(range(256)) * 64


@dataclass(frozen=True, kw_only=True)
class RelayRecipe(Recipe):
    """The record-only recipe, offering ``relay`` for multimodal calls."""

    relay: MultimodalRelay | None = field(default=None)

    @property
    def multimodal_relay(self) -> MultimodalRelay | None:
        return self.relay


def gateway_app(received: list[dict]) -> web.Application:
    """A multimodal gateway at both presets' paths, remembering each request."""

    async def handle(request: web.Request) -> web.StreamResponse:
        received.append(
            {
                "path": request.path,
                "payload": await request.json(),
                "authorization": request.headers.get("Authorization"),
                "reef_headers": sorted(name.lower() for name in request.headers if name.lower().startswith("x-reef")),
            }
        )
        if request.path == "/v1/audio/speech":
            return web.Response(body=AUDIO_BYTES, content_type="audio/mpeg")
        if request.path == "/v1/embeddings":
            return web.json_response({"error": {"message": "Model not/real does not exist"}}, status=400)
        return web.json_response({"ok": request.path, "usage": {"cost": 0.04}})

    app = web.Application()
    for path in ("/v1/images", "/v1/images/generations", "/v1/embeddings", "/v1/audio/speech", "/alpha/decisions"):
        app.router.add_post(path, handle)
    return app


async def reef_client(tmp_path, relay: MultimodalRelay | None) -> tuple[TestClient, Dispatcher]:
    initial = tmp_path / "initial"
    initial.mkdir(exist_ok=True)
    dispatcher = Dispatcher(
        RelayRecipe(relay=relay),
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(),
    )
    client = TestClient(TestServer(create_app(dispatcher)))
    await client.start_server()
    return client, dispatcher


def provider_at(server: TestServer, preset: str) -> MultimodalProvider:
    return MultimodalProvider(PRESETS[preset], str(server.make_url("")).rstrip("/"), "gateway-key")


@pytest.mark.unit
def test_the_recipes_relay_reaches_the_gateway_and_records_nothing(tmp_path) -> None:
    async def run() -> None:
        received: list[dict] = []
        gateway = TestServer(gateway_app(received))
        await gateway.start_server()
        client, dispatcher = await reef_client(tmp_path, ProviderRelay(provider_at(gateway, "openrouter")))
        headers = {"x-reef-scenario": "media"}
        try:
            speech = await client.post("/v1/audio/speech", headers=headers, json={"model": "tts", "input": "hi"})
            assert speech.status == 200 and speech.headers["Content-Type"] == "audio/mpeg"
            assert await speech.read() == AUDIO_BYTES
            assert "x-reef-agent-record-id" not in speech.headers
            decision = await client.post("/v1/decisions", headers=headers, json={"model": "jev"})
            assert (await decision.json())["ok"] == "/alpha/decisions"
            refused = await client.post("/v1/embeddings", headers=headers, json={"model": "not/real", "input": "x"})
            assert refused.status == 400 and "does not exist" in await refused.text()
            streamed = await client.post("/v1/images", headers=headers, json={"model": "img", "stream": True})
            assert streamed.status == 400

            assert [request["path"] for request in received] == [
                "/v1/audio/speech",
                "/alpha/decisions",
                "/v1/embeddings",
            ]
            assert {request["authorization"] for request in received} == {"Bearer gateway-key"}
            assert all(request["reef_headers"] == [] for request in received)
            assert not dispatcher.get_or_create_scenario("media").records.replay("media")
        finally:
            await client.close()
            await gateway.close()

    asyncio.run(run())


@pytest.mark.unit
def test_an_openai_compatible_gateway_gets_its_paths_and_no_decisions(tmp_path) -> None:
    async def run() -> None:
        received: list[dict] = []
        gateway = TestServer(gateway_app(received))
        await gateway.start_server()
        client, _ = await reef_client(tmp_path, ProviderRelay(provider_at(gateway, "openai-compatible")))
        headers = {"x-reef-scenario": "media"}
        try:
            image = await client.post("/v1/images", headers=headers, json={"model": "img", "prompt": "reef"})
            assert (await image.json())["ok"] == "/v1/images/generations"
            decision = await client.post("/v1/decisions", headers=headers, json={"model": "jev"})
            assert decision.status == 501 and "serves no /v1/decisions" in await decision.text()
        finally:
            await client.close()
            await gateway.close()

    asyncio.run(run())


@pytest.mark.unit
def test_a_recipe_without_a_relay_serves_no_multimodal_call(tmp_path) -> None:
    async def run() -> None:
        client, _ = await reef_client(tmp_path, None)
        try:
            response = await client.post("/v1/images", headers={"x-reef-scenario": "media"}, json={"model": "img"})
            assert response.status == 501 and "relays no multimodal calls" in await response.text()
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_multimodal_settings_read_the_preset_address_and_key() -> None:
    assert MultimodalSettings.from_config(None, {}) is None
    settings = MultimodalSettings.from_config({"api_key": " sk-or "}, {})
    assert (settings.preset.name, settings.base_url, settings.api_key) == (
        "openrouter",
        "https://openrouter.ai/api",
        "sk-or",
    )
    assert "sk-or" not in repr(settings)
    # A variable instead, as evolution.models take one; a literal wins.
    assert MultimodalSettings.from_config({"api_key_env": "MM_KEY"}, {"MM_KEY": "sk-env"}).api_key == "sk-env"
    assert (
        MultimodalSettings.from_config({"api_key": "sk-lit", "api_key_env": "MM_KEY"}, {"MM_KEY": "x"}).api_key
        == "sk-lit"
    )
    # No key of its own (the profile's ${REEF_MULTIMODAL_API_KEY} unset is empty): the upstream's, only when the
    # upstream is the same gateway.
    keyless = MultimodalSettings.from_config({"api_key": ""}, {})
    assert keyless.provider("https://openrouter.ai/api/", "upstream").api_key == "upstream"
    assert keyless.provider("http://127.0.0.1:11434", "ollama") is None
    compatible = MultimodalSettings.from_config({"preset": "openai-compatible", "url": "https://gw.example/"}, {})
    assert compatible.base_url == "https://gw.example"
    with pytest.raises(ValueError, match="needs url"):
        MultimodalSettings.from_config({"preset": "openai-compatible"}, {})
    with pytest.raises(ValueError, match="unknown multimodal preset"):
        MultimodalSettings.from_config({"preset": "nope"}, {})


@pytest.mark.unit
def test_reefine_hands_its_gateway_to_the_relay_and_the_agent() -> None:
    from reef.recipe.reefine import ReefineRecipe
    from reef.recipe.reefine.agent import AgentProposer

    settings = {"evolution": {"tasks": ["[health] reply reef-ok"], "multimodal": {"preset": "openrouter"}}}
    kwargs = ReefineRecipe._recipe_kwargs(settings, {})
    upstream = InferenceProxyRuntime(base_url="https://openrouter.ai/api", api_key="sk-upstream", model_path="m")
    recipe = ReefineRecipe(**kwargs, runtime=upstream)
    relay = recipe.multimodal_relay
    assert isinstance(relay, ProviderRelay) and relay.provider.api_key == "sk-upstream"
    assert isinstance(recipe.propose, AgentProposer) and recipe.propose.provider == relay.provider
    # A scenario on its own model (BYOK) still relays on the deployment's key, which the platform bills.
    own = InferenceProxyRuntime(base_url="https://openrouter.ai/api", api_key="sk-byok", model_path="m")
    scenario_recipe = recipe.with_model_config(ModelConfig(runtime=own))
    assert scenario_recipe.multimodal_relay.provider.api_key == "sk-upstream"
    assert scenario_recipe.propose.provider.api_key == "sk-upstream"
    elsewhere = InferenceProxyRuntime(base_url="http://127.0.0.1:11434", api_key="ollama", model_path="m")
    assert ReefineRecipe(**kwargs, runtime=elsewhere).multimodal_relay is None
    with pytest.raises(RecipeConfigError, match=r"evolution\.multimodal"):
        ReefineRecipe._recipe_kwargs({"evolution": {**settings["evolution"], "multimodal": "openrouter"}}, {})
