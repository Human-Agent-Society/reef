from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from aiohttp.test_utils import TestClient, TestServer

from reef.artifact import ArtifactRef, GitLFSRepositoryBackend, InMemoryRepositoryBackend
from reef.core import RequestType
from reef.dispatcher import build_default_dispatcher
from reef.runtime.inference import InferenceBackend
from reef.service.app import RequestService, create_app


class StubInferenceBackend(InferenceBackend):
    async def inference(self, artifact, path, payload):
        del artifact, path, payload
        return {"ok": True}


def _payload_for(path: str) -> dict:
    if path == "/reef/report":
        return {"score": 1.0}
    return {"model": "reef", "messages": []}


@pytest.mark.unit
@pytest.mark.parametrize(
    "path",
    ["/v1/chat/completions", "/reef/report"],
)
def test_new_scenario_is_created_for_every_request_type(path: str) -> None:
    async def run() -> None:
        dispatcher = build_default_dispatcher()
        client = TestClient(
            TestServer(
                create_app(
                    dispatcher,
                    inference_backend=StubInferenceBackend(),
                )
            )
        )
        await client.start_server()
        try:
            response = await client.post(
                path,
                headers={"x-reef-scenario": "new-scenario"},
                json=_payload_for(path),
            )

            assert response.status == 200
            assert dispatcher.has_loaded("new-scenario")
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
@pytest.mark.parametrize(
    "path",
    ["/v1/chat/completions", "/reef/report"],
)
def test_recipe_header_is_not_part_of_the_protocol(path: str) -> None:
    """A deployment serves one recipe, so requests cannot choose one: a stray
    ``x-reef-recipe`` is an ordinary unknown header and changes nothing."""

    async def run() -> None:
        dispatcher = build_default_dispatcher()
        client = TestClient(
            TestServer(
                create_app(
                    dispatcher,
                    inference_backend=StubInferenceBackend(),
                )
            )
        )
        await client.start_server()
        try:
            response = await client.post(
                path,
                headers={"x-reef-scenario": "new-scenario", "x-reef-recipe": "different"},
                json=_payload_for(path),
            )

            assert response.status == 200
            assert dispatcher.has_loaded("new-scenario")
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
@pytest.mark.parametrize(
    "path",
    ["/v1/chat/completions", "/reef/report"],
)
def test_registered_scenario_keeps_its_binding_for_every_request_type(path: str) -> None:
    async def run() -> None:
        dispatcher = build_default_dispatcher()
        RequestService(dispatcher).accept(
            {"x-reef-scenario": "registered"},
            {"score": 1.0},
            request_type=RequestType.REPORT,
        )
        client = TestClient(
            TestServer(
                create_app(
                    dispatcher,
                    inference_backend=StubInferenceBackend(),
                )
            )
        )
        await client.start_server()
        try:
            response = await client.post(
                path,
                headers={"x-reef-scenario": "registered"},
                json=_payload_for(path),
            )

            assert response.status == 200
            assert dispatcher.has_loaded("registered")
        finally:
            await client.close()

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

    first = RequestService(build_default_dispatcher(backend_factory=first_factory))
    first.accept({"x-reef-scenario": "durable"}, {"score": 1.0}, request_type=RequestType.REPORT)

    restarted_factory = GitLFSRepositoryBackend.factory(
        remote,
        work_dir=tmp_path / "restarted-work",
        cache_dir=tmp_path / "restarted-cache",
    )
    restarted_dispatcher = build_default_dispatcher(backend_factory=restarted_factory)
    restarted = RequestService(restarted_dispatcher)
    restarted.accept(
        {"x-reef-scenario": "durable"},
        {"score": 0.5},
        request_type=RequestType.REPORT,
    )

    assert restarted_dispatcher.has_loaded("durable")


@pytest.mark.unit
def test_different_scenarios_do_not_share_a_creation_lock(monkeypatch) -> None:
    dispatcher = build_default_dispatcher()
    load_or_create = dispatcher._registry._scenario_factory.load_or_create
    entered = Barrier(2)

    def load_or_create_together(scenario, release_id):
        entered.wait(timeout=2)
        return load_or_create(scenario, release_id)

    monkeypatch.setattr(dispatcher._registry._scenario_factory, "load_or_create", load_or_create_together)

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
    dispatcher = build_default_dispatcher(backend_factory=backend_factory)

    created = dispatcher.get_or_create_scenario(
        "moving-latest",
        release_id="head",
    )

    assert created.repository.base_artifact.release_id == initial_release_id
    assert head_calls == 1
