"""An application that embeds Reef: an injected inference handler, dispatcher access and an artifact backend."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from tests.support.artifact_backend_plugin import MemoryArtifactBackend

from reef.artifact import GitLFSRepositoryBackendFactory, RepositoryBackendContext, git_lfs
from reef.runtime.interfaces import InferenceHandler
from reef.service import assembly
from reef.service.app import get_dispatcher
from reef.service.deploy.deployment_config import translate_layout
from reef.service.deploy.service_config import ServiceConfig, service_config_from_mapping
from reef.storage.sqlite import SQLiteScenarioStorage

MEMORY_BACKEND = "tests.support.artifact_backend_plugin:MemoryArtifactBackend"
UNREACHABLE_BACKEND = "tests.support.artifact_backend_plugin:UnreachableArtifactBackend"


class EchoHandler(InferenceHandler):
    def __init__(self) -> None:
        self.paths: list[str] = []

    async def inference(self, artifact, path, payload):
        self.paths.append(path)
        return {"model": payload["model"]}


def _settings(tmp_path: Path, **overrides) -> ServiceConfig:
    values = {
        "recipe": "recipe",
        "agent_record_dir": str(tmp_path / "records"),
        "artifact_work_dir": str(tmp_path / "work"),
        "artifact_cache_dir": str(tmp_path / "cache"),
        "artifact_backend": {"module": MEMORY_BACKEND, "config": {"label": "memory"}},
    }
    values.update(overrides)
    return ServiceConfig(**values)


@pytest.fixture
def no_git_lfs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if the service builds the Git LFS backend or checks its tools."""

    def refuse(*args, **kwargs):
        raise AssertionError("a plugin artifact backend must not build Git LFS")

    monkeypatch.setattr(GitLFSRepositoryBackendFactory, "from_config", refuse)
    monkeypatch.setattr(git_lfs, "_check_tools", refuse)


@pytest.mark.unit
def test_build_app_serves_through_an_injected_handler_and_exposes_its_dispatcher(tmp_path, no_git_lfs) -> None:
    handler = EchoHandler()
    app = assembly.build_app(_settings(tmp_path, tokens=("secret",)), inference_handler=handler)
    dispatcher = get_dispatcher(app)

    async def scenario_names(request: web.Request) -> web.Response:
        return web.json_response(get_dispatcher(request.app).list_scenarios())

    app.router.add_get("/extension/scenarios", scenario_names)

    async def run() -> None:
        authorization = {"Authorization": "Bearer secret"}
        async with TestClient(TestServer(app)) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={**authorization, "x-reef-scenario": "math"},
                json={"model": "reef", "messages": []},
            )
            assert response.status == 200
            assert (await response.json())["model"] == "reef"
            # The service tokens also protect the routes that the application adds.
            assert (await client.get("/extension/scenarios")).status == 401
            extension = await client.get("/extension/scenarios", headers=authorization)
            assert "math" in await extension.text()

    asyncio.run(run())
    assert handler.paths == ["/v1/chat/completions"]
    # The app owns the dispatcher it built and closed it on cleanup.
    assert dispatcher.local_workers.closed.is_set()


@pytest.mark.unit
def test_a_plugin_artifact_backend_gets_its_config_and_the_deployment_context(tmp_path, no_git_lfs) -> None:
    dispatcher = assembly.build_dispatcher(_settings(tmp_path), environ={})
    try:
        factory = dispatcher.registry._backend_factory
        assert isinstance(factory, MemoryArtifactBackend)
        assert factory.label == "memory"
        assert factory.context == RepositoryBackendContext(work_dir=tmp_path / "work", cache_dir=tmp_path / "cache")
        scenario = dispatcher.get_or_create_scenario("math")
        assert scenario.repository.current_artifact is not None
        assert factory.list_registrations() == ("math",)
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_service_config_reads_the_artifact_backend_from_the_storage_section() -> None:
    backend = {"module": MEMORY_BACKEND, "config": {"label": "memory"}}
    layout = translate_layout({"recipe": {"implementation": "recipe"}, "storage": {"artifact_backend": backend}})

    assert service_config_from_mapping(layout).artifact_backend == backend
    assert service_config_from_mapping({"reef": {"recipe": "recipe"}}).artifact_backend == "git_lfs"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"artifact_backend": "s3"}, "git_lfs or 'package.module:ClassName'"),
        ({"artifact_backend": "reef.storage.sqlite:SQLiteScenarioStorage"}, "ConfigurableRepositoryBackendFactory"),
        ({"artifact_backend": MEMORY_BACKEND, "artifact_repository": "/srv/artifacts.git"}, "requires"),
    ],
)
def test_service_config_refuses_an_unusable_artifact_backend(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        ServiceConfig(recipe="recipe", **overrides)


@pytest.mark.unit
def test_a_failed_artifact_backend_names_itself_and_releases_the_storage(tmp_path, monkeypatch, no_git_lfs) -> None:
    closed = []

    class RecordedStorage(SQLiteScenarioStorage):
        def close(self) -> None:
            closed.append(True)
            super().close()

    monkeypatch.setattr(assembly, "SQLiteScenarioStorage", RecordedStorage)

    with pytest.raises(ConnectionError, match="unreachable") as raised:
        assembly.build_app(_settings(tmp_path, artifact_backend=UNREACHABLE_BACKEND), inference_handler=EchoHandler())

    assert raised.value.__notes__ == ["while building storage.artifact_backend UnreachableArtifactBackend"]
    assert closed == [True]


@pytest.mark.unit
def test_a_failed_app_closes_the_dispatcher_and_keeps_the_original_error(tmp_path, monkeypatch, no_git_lfs) -> None:
    dispatcher = assembly.build_dispatcher(_settings(tmp_path), environ={})
    closed = []

    def close() -> None:
        closed.append(True)
        raise RuntimeError("close failed")

    def create_app(*args, **kwargs):
        raise ValueError("bad route")

    monkeypatch.setattr(dispatcher, "close", close)
    monkeypatch.setattr(assembly, "build_dispatcher", lambda *args, **kwargs: dispatcher)
    monkeypatch.setattr(assembly, "create_app", create_app)

    with pytest.raises(ValueError, match="bad route"):
        assembly.build_app(_settings(tmp_path))
    assert closed == [True]
