from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from aiohttp.test_utils import TestClient, TestServer

from reef.artifact import ArtifactRef, GitLFSRepositoryBackend, InMemoryRepositoryBackend
from reef.core import RequestType
from reef.dispatcher import Dispatcher, build_default_dispatcher
from reef.inference.http import InferenceProxyRuntime
from reef.recipe import Recipe
from reef.runtime.interfaces import InferenceHandler
from reef.service.app import RequestService, create_app
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sqlite import SQLiteScenarioStorage


class StubInferenceHandler(InferenceHandler):
    async def inference(self, artifact, path, payload):
        del artifact, path, payload
        return {"ok": True}


@pytest.mark.unit
@pytest.mark.parametrize(
    "registered, headers",
    [
        (False, {}),
        # A deployment serves one recipe, so requests cannot choose one: a stray
        # ``x-reef-recipe`` is an ordinary unknown header and changes nothing.
        (False, {"x-reef-recipe": "different"}),
        # A registered scenario keeps its binding.
        (True, {}),
    ],
    ids=["new", "stray-recipe-header", "registered"],
)
def test_a_report_creates_or_reuses_the_scenario(registered: bool, headers: dict) -> None:
    async def run() -> None:
        dispatcher = build_default_dispatcher(
            scenario_storage=SQLiteScenarioStorage(), coordination=memory_coordination()
        )
        if registered:
            RequestService(dispatcher).accept(
                {"x-reef-scenario": "scenario"}, {"score": 1.0}, request_type=RequestType.REPORT
            )
        client = TestClient(TestServer(create_app(dispatcher)))
        await client.start_server()
        try:
            response = await client.post(
                "/reef/report", headers={"x-reef-scenario": "scenario", **headers}, json={"score": 1.0}
            )

            assert response.status == 200
            assert dispatcher.registry.get_optional("scenario") is not None
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
@pytest.mark.parametrize("headers", [{}, {"x-reef-recipe": "different"}], ids=["plain", "stray-recipe-header"])
def test_inference_answers_a_created_scenario_and_never_creates_one(tmp_path, headers: dict) -> None:
    async def run() -> None:
        initial = tmp_path / "initial"
        initial.mkdir()
        dispatcher = Dispatcher(
            Recipe(runtime=InferenceProxyRuntime(base_url="http://engine.test", model_path="reef")),
            InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
            scenario_storage=SQLiteScenarioStorage(),
            coordination=memory_coordination(),
        )
        client = TestClient(TestServer(create_app(dispatcher, inference_handler=StubInferenceHandler())))
        await client.start_server()
        payload = {"model": "reef", "messages": []}
        try:
            unknown = await client.post(
                "/v1/chat/completions", headers={"x-reef-scenario": "scenario", **headers}, json=payload
            )
            assert unknown.status == 404
            assert dispatcher.registry.get_optional("scenario") is None

            dispatcher.get_or_create_scenario("scenario")
            response = await client.post(
                "/v1/chat/completions", headers={"x-reef-scenario": "scenario", **headers}, json=payload
            )
            assert response.status == 200, await response.text()
            assert await response.json() == {"ok": True}
        finally:
            await client.close()
            dispatcher.close()

    asyncio.run(run())


@pytest.mark.integration
def test_request_recovers_durable_scenario_without_resubmission(
    tmp_path,
    fake_git_lfs: None,
) -> None:
    remote = tmp_path / "artifacts.git"
    first_factory = GitLFSRepositoryBackend.factory(
        remote,
        work_dir=tmp_path / "first-work",
        cache_dir=tmp_path / "first-cache",
    )

    first = RequestService(
        build_default_dispatcher(
            backend_factory=first_factory, scenario_storage=SQLiteScenarioStorage(), coordination=memory_coordination()
        )
    )
    first.accept({"x-reef-scenario": "durable"}, {"score": 1.0}, request_type=RequestType.REPORT)

    restarted_factory = GitLFSRepositoryBackend.factory(
        remote,
        work_dir=tmp_path / "restarted-work",
        cache_dir=tmp_path / "restarted-cache",
    )
    restarted_dispatcher = build_default_dispatcher(
        backend_factory=restarted_factory, scenario_storage=SQLiteScenarioStorage(), coordination=memory_coordination()
    )
    restarted = RequestService(restarted_dispatcher)
    restarted.accept(
        {"x-reef-scenario": "durable"},
        {"score": 0.5},
        request_type=RequestType.REPORT,
    )

    assert restarted_dispatcher.registry.get_optional("durable") is not None


@pytest.mark.unit
def test_different_scenarios_do_not_share_a_creation_lock(monkeypatch) -> None:
    dispatcher = build_default_dispatcher(scenario_storage=SQLiteScenarioStorage(), coordination=memory_coordination())
    load_or_create = dispatcher.registry._scenario_factory.load_or_create
    entered = Barrier(2)

    def load_or_create_together(scenario, release_id):
        entered.wait(timeout=2)
        return load_or_create(scenario, release_id)

    monkeypatch.setattr(dispatcher.registry._scenario_factory, "load_or_create", load_or_create_together)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(dispatcher.get_or_create_scenario, "first"),
            executor.submit(dispatcher.get_or_create_scenario, "second"),
        )
        scenarios = [future.result() for future in futures]

    assert {scenario.name for scenario in scenarios} == {"first", "second"}


@pytest.mark.unit
def test_create_freezes_head_selector_at_the_resolved_release(monkeypatch, tmp_path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    backend_factory = InMemoryRepositoryBackend.factory(
        initial,
        root=tmp_path / "repository",
    )
    backend = backend_factory("moving-latest")
    initial_release_id = backend.resolve_release("head").release_id
    resolve_release = backend.resolve_release
    head_calls = 0

    def moving_head(release_id=None):
        nonlocal head_calls
        if release_id == "head":
            head_calls += 1
            if head_calls > 1:
                return ArtifactRef("moved", "moved-release", initial_release_id)
        return resolve_release(release_id)

    monkeypatch.setattr(backend, "resolve_release", moving_head)
    dispatcher = build_default_dispatcher(
        backend_factory=backend_factory, scenario_storage=SQLiteScenarioStorage(), coordination=memory_coordination()
    )

    created = dispatcher.get_or_create_scenario(
        "moving-latest",
        release_id="head",
    )

    assert created.repository.base_artifact.release_id == initial_release_id
    assert head_calls == 1
