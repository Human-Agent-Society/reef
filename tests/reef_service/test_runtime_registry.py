from __future__ import annotations

import asyncio

import pytest
from aiohttp.test_utils import TestClient, TestServer

from reef.artifact import Artifact, InMemoryRepositoryBackend, LiveWeightArtifactRef
from reef.dispatcher import Dispatcher
from reef.recipe import Recipe, RecipeRegistry
from reef.recipe.registry import build_named_recipe
from reef.runtime import InferenceProxyRuntime, InferenceRuntime, RuntimeConfigError, RuntimeRegistry
from reef.runtime.inference import InferenceBackend
from reef.service.app import create_app


@pytest.mark.unit
def test_runtime_repository_builds_all_available_runtime_types() -> None:
    repository = RuntimeRegistry()
    proxy = repository.build(
        {"type": "inference_proxy", "base_url": "http://provider", "api_key": "secret"},
        model_path="Qwen/Qwen3.5-27B",
    )

    assert isinstance(proxy, InferenceProxyRuntime)
    assert repository.names == ("inference_proxy", "ray_training")


@pytest.mark.unit
def test_runtime_repository_rejects_unknown_recipe_runtime() -> None:
    with pytest.raises(RuntimeConfigError, match="unknown runtime type 'missing'"):
        RuntimeRegistry().build({"type": "missing"}, model_path="Qwen/Qwen3.5-27B")


@pytest.mark.unit
def test_runtime_repository_preserves_optional_recipe_context() -> None:
    contexts = []

    def factory(config, model_path, recipe_config, environ):
        del config, environ
        contexts.append(recipe_config)
        return InferenceProxyRuntime(model_path=model_path, base_url="http://provider")

    repository = RuntimeRegistry({"capture": factory})
    repository.build({"type": "capture"}, model_path="model", environ={})
    configured = {"data": {"temperature": 0.5}}
    repository.build({"type": "capture"}, model_path="model", recipe_config=configured, environ={})

    assert contexts[0] == {}
    assert contexts[1] is configured


@pytest.mark.unit
def test_recipe_rejects_unknown_runtime_before_scenario_is_created(tmp_path) -> None:
    recipes = tmp_path / "recipes"
    recipes.mkdir()
    (recipes / "qwen.yaml").write_text(
        """implementation: recipe
runtime:
  type: missing
  base_url: http://runtime
model:
  path: Qwen/Qwen3.5-27B
"""
    )

    with pytest.raises(RuntimeConfigError, match="unknown runtime type 'missing'"):
        build_named_recipe("qwen", config_directory=recipes)


@pytest.mark.unit
def test_recipe_resolution_does_not_probe_runtime(tmp_path, monkeypatch) -> None:
    recipes = tmp_path / "recipes"
    recipes.mkdir()
    (recipes / "qwen.yaml").write_text(
        """implementation: recipe
runtime:
  type: inference_proxy
  base_url: http://provider
  api_key: secret
model:
  path: Qwen/Qwen3.5-27B
"""
    )

    def unexpected_probe(*args, **kwargs):
        del args, kwargs
        raise AssertionError("recipe resolution must not probe the runtime")

    monkeypatch.setattr("urllib.request.urlopen", unexpected_probe)
    initial = tmp_path / "initial"
    initial.mkdir()
    dispatcher = Dispatcher(
        RecipeRegistry({"qwen": build_named_recipe("qwen", config_directory=recipes)}),
        InMemoryRepositoryBackend.factory(initial),
    )

    scenario = dispatcher.get_or_create_scenario("math", "qwen")

    assert scenario.recipe == "qwen"


@pytest.mark.unit
def test_http_app_routes_scenarios_to_runtime_specific_inference_services(tmp_path) -> None:
    async def run() -> None:
        class RoutingRuntime(InferenceRuntime):
            def __init__(self, recipe: str) -> None:
                super().__init__(base_url="http://recipe-runtime")
                self.recipe = recipe

            @property
            def inference_backend(self) -> InferenceBackend:
                class _Backend(InferenceBackend):
                    async def inference(self, artifact, path, payload):
                        del artifact, path, payload
                        return {"runtime": recipe}

                return _Backend()

        initial = tmp_path / "initial"
        initial.mkdir()
        dispatcher = Dispatcher(
            RecipeRegistry(
                recipes={
                    "qwen": Recipe(name="qwen", runtime=RoutingRuntime("qwen")),
                    "gemma": Recipe(name="gemma", runtime=RoutingRuntime("gemma")),
                },
            ),
            InMemoryRepositoryBackend.factory(initial),
        )

        for scenario, recipe in (("math", "qwen"), ("code", "gemma")):
            dispatcher.get_or_create_scenario(scenario, recipe)

        client = TestClient(TestServer(create_app(dispatcher)))
        await client.start_server()
        try:
            for scenario, recipe in (("math", "qwen"), ("code", "gemma")):
                response = await client.post(
                    "/v1/chat/completions",
                    headers={"x-reef-scenario": scenario},
                    json={"messages": []},
                )
                assert response.status == 200
                assert await response.json() == {"runtime": recipe}
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_recipe_runtime_advertises_itself_and_proxies_provider_api() -> None:
    async def run() -> None:
        received = {}

        async def upstream(request):
            received["path"] = request.path
            received["payload"] = await request.json()
            received["version"] = request.headers["x-reef-artifact-version"]
            received["authorization"] = request.headers["Authorization"]
            return __import__("aiohttp").web.json_response({"provider": "ok"})

        from aiohttp import web

        upstream_app = web.Application()
        upstream_app.router.add_post("/v1/chat/completions", upstream)
        upstream_server = TestServer(upstream_app)
        await upstream_server.start_server()
        runtime = InferenceProxyRuntime(
            model_path="Qwen/Qwen3.5-27B",
            base_url=str(upstream_server.make_url("")).rstrip("/"),
            api_key="secret",
        )
        try:
            response = await runtime.inference_backend.inference(
                Artifact(LiveWeightArtifactRef("id", "version-1", None, "weight-1"), None),
                "/v1/chat/completions",
                {"messages": [{"role": "user", "content": "hi"}]},
            )
            assert response == {"provider": "ok"}
            assert received == {
                "path": "/v1/chat/completions",
                "payload": {"messages": [{"role": "user", "content": "hi"}]},
                "version": "version-1",
                "authorization": "Bearer secret",
            }

            stream = await runtime.inference_backend.inference_stream(
                Artifact(LiveWeightArtifactRef("id", "version-1", None, "weight-1"), None),
                "/v1/chat/completions",
                {"messages": [], "stream": True},
            )
            try:
                streamed_body = b"".join([chunk async for chunk in stream.chunks])
            finally:
                await stream.close()
            assert stream.status == 200
            assert stream.headers["Content-Type"].startswith("application/json")
            assert streamed_body == b'{"provider": "ok"}'
            assert received["payload"] == {"messages": [], "stream": True}
        finally:
            await upstream_server.close()

    asyncio.run(run())


@pytest.mark.unit
def test_recipe_runtime_proxies_anthropic_auth_and_version() -> None:
    async def run() -> None:
        received = {}

        async def upstream(request):
            received["payload"] = await request.json()
            received["api_key"] = request.headers["x-api-key"]
            received["version"] = request.headers["anthropic-version"]
            return __import__("aiohttp").web.json_response({"content": []})

        from aiohttp import web

        upstream_app = web.Application()
        upstream_app.router.add_post("/v1/messages", upstream)
        upstream_server = TestServer(upstream_app)
        await upstream_server.start_server()
        runtime = InferenceProxyRuntime(
            model_path="claude-compatible",
            base_url=str(upstream_server.make_url("")).rstrip("/"),
            api_key="anthropic-secret",
        )
        try:
            response = await runtime.inference_backend.inference(
                Artifact(LiveWeightArtifactRef("id", "rev", None, "weight-1"), None),
                "/v1/messages",
                {"model": "claude-compatible", "messages": []},
            )
            assert response == {"content": []}
            assert received == {
                "payload": {"model": "claude-compatible", "messages": []},
                "api_key": "anthropic-secret",
                "version": "2023-06-01",
            }
        finally:
            await upstream_server.close()

    asyncio.run(run())


@pytest.mark.unit
def test_inference_proxy_runtime_inherits_runtime_base() -> None:
    runtime = InferenceProxyRuntime(
        model_path="Qwen/Qwen3.5-27B",
        base_url="http://provider",
        api_key="secret",
    )

    assert isinstance(runtime, InferenceRuntime)
    assert runtime.model_path == "Qwen/Qwen3.5-27B"
    assert runtime.base_url == "http://provider"
    assert runtime.inference_timeout_s == 300.0

    with pytest.raises(ValueError, match="base_url must be non-empty"):
        InferenceProxyRuntime(
            model_path="Qwen/Qwen3.5-27B",
            base_url="",
            api_key="secret",
        )
    with pytest.raises(ValueError, match="inference_timeout_s must be positive"):
        InferenceProxyRuntime(
            model_path="Qwen/Qwen3.5-27B",
            base_url="http://provider",
            api_key="secret",
            inference_timeout_s=0,
        )
