from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from reef.artifact import ArtifactRef, GitLFSRepositoryBackend, InMemoryRepositoryBackend
from reef.artifact.git_client import GitClient
from reef.core import ReefError, RequestType
from reef.dispatcher import Dispatcher, build_default_dispatcher
from reef.recipe import Recipe, RecipeRegistry, ScenarioRecipeConflict
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
def test_new_scenario_binds_the_served_recipe_for_every_request_type(path: str) -> None:
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
            assert dispatcher.get_or_create_scenario("new-scenario").recipe == "recipe"
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
            assert dispatcher.get_or_create_scenario("new-scenario").recipe == "recipe"
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
            assert dispatcher.get_or_create_scenario("registered").recipe == "recipe"
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.integration
@pytest.mark.parametrize(
    "path",
    ["/v1/chat/completions", "/reef/report"],
)
def test_scenario_bound_to_another_recipe_conflicts_for_every_request_type(
    path: str,
    tmp_path,
    fake_git_lfs: None,
) -> None:
    """A durable scenario carries its recipe; a deployment serving a different
    one cannot adopt it."""
    remote = tmp_path / "artifacts.git"
    RequestService(
        build_default_dispatcher(
            backend_factory=GitLFSRepositoryBackend.factory(
                remote,
                work_dir=tmp_path / "first-work",
                cache_dir=tmp_path / "first-cache",
            )
        )
    ).accept({"x-reef-scenario": "registered"}, {"score": 1.0}, request_type=RequestType.REPORT)

    other = Dispatcher(
        RecipeRegistry({"other": Recipe(name="other")}),
        GitLFSRepositoryBackend.factory(
            remote,
            work_dir=tmp_path / "other-work",
            cache_dir=tmp_path / "other-cache",
        ),
    )

    async def run() -> None:
        client = TestClient(TestServer(create_app(other, inference_backend=StubInferenceBackend())))
        await client.start_server()
        try:
            response = await client.post(
                path,
                headers={"x-reef-scenario": "registered"},
                json=_payload_for(path),
            )

            assert response.status == 409
            assert "already bound to recipe 'recipe', not 'other'" in await response.text()
            assert not other.has_loaded("registered")
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.integration
def test_request_recovers_durable_recipe_without_resubmission(
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

    assert restarted_dispatcher.get_or_create_scenario("durable").recipe == "recipe"


@pytest.mark.unit
def test_rejected_recipeless_creation_leaves_no_scenario_state(tmp_path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    backend = InMemoryRepositoryBackend.factory(
        initial,
        root=tmp_path / "repository",
    )
    agent_record_dir = tmp_path / "agent-record"
    # A multi-recipe registry serves no single recipe, so creation must name one.
    dispatcher = Dispatcher(
        RecipeRegistry({"recipe": Recipe(), "other": Recipe(name="other")}),
        backend,
        agent_record_dir=agent_record_dir,
    )

    with pytest.raises(ReefError, match="no served recipe"):
        RequestService(dispatcher).accept(
            {"x-reef-scenario": "rejected"},
            {"score": 1.0},
            request_type=RequestType.REPORT,
        )

    assert not dispatcher.has_loaded("rejected")
    assert not backend.has_registration("rejected")
    assert not (tmp_path / "repository").exists()
    assert list(agent_record_dir.iterdir()) == []


@pytest.mark.unit
def test_different_scenarios_do_not_share_a_creation_lock(monkeypatch) -> None:
    dispatcher = build_default_dispatcher()
    load_or_create = dispatcher._registry._scenario_factory.load_or_create
    entered = Barrier(2)

    def load_or_create_together(scenario, recipe, artifact_version):
        entered.wait(timeout=2)
        return load_or_create(scenario, recipe, artifact_version)

    monkeypatch.setattr(dispatcher._registry._scenario_factory, "load_or_create", load_or_create_together)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(dispatcher.get_or_create_scenario, "first", "recipe"),
            executor.submit(dispatcher.get_or_create_scenario, "second", "recipe"),
        )
        scenarios = [future.result() for future in futures]

    assert {scenario.name for scenario in scenarios} == {"first", "second"}


@pytest.mark.unit
def test_create_freezes_latest_selector_at_the_resolved_version(monkeypatch, tmp_path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    backend_factory = InMemoryRepositoryBackend.factory(
        initial,
        root=tmp_path / "repository",
    )
    backend = backend_factory("moving-latest")
    initial_version = backend.resolve_version("latest").version
    resolve_version = backend.resolve_version
    latest_calls = 0

    def moving_latest(version=None):
        nonlocal latest_calls
        if version == "latest":
            latest_calls += 1
            if latest_calls > 1:
                return ArtifactRef("moved", "moved-version", initial_version)
        return resolve_version(version)

    monkeypatch.setattr(backend, "resolve_version", moving_latest)
    dispatcher = build_default_dispatcher(backend_factory=backend_factory)

    created = dispatcher.get_or_create_scenario(
        "moving-latest",
        "recipe",
        "latest",
    )

    assert created.repository.base_artifact.version == initial_version
    assert latest_calls == 1


@pytest.mark.integration
def test_concurrent_first_requests_bind_exactly_one_recipe(
    tmp_path,
    fake_git_lfs: None,
    monkeypatch,
) -> None:
    remote = tmp_path / "artifacts.git"
    seed_factory = GitLFSRepositoryBackend.factory(
        remote,
        work_dir=tmp_path / "seed-work",
        cache_dir=tmp_path / "seed-cache",
    )
    seed_factory("seed")
    backend_factories = (
        GitLFSRepositoryBackend.factory(
            remote,
            work_dir=tmp_path / "first-work",
            cache_dir=tmp_path / "first-cache",
        ),
        GitLFSRepositoryBackend.factory(
            remote,
            work_dir=tmp_path / "second-work",
            cache_dir=tmp_path / "second-cache",
        ),
    )

    def build_dispatcher(index: int) -> Dispatcher:
        return Dispatcher(
            RecipeRegistry(
                {
                    "recipe-a": Recipe(name="recipe-a"),
                    "recipe-b": Recipe(name="recipe-b"),
                },
            ),
            backend_factories[index],
        )

    dispatchers = (build_dispatcher(0), build_dispatcher(1))
    push_barrier = Barrier(2)
    registration_push_lock = Lock()
    git_run = GitClient.run

    def synchronize_registration_push(self, command, *, cwd=None, source_error=False):
        if command[:3] == ("git", "push", "origin") and any(
            "refs/reef/scenarios/" in argument for argument in command
        ):
            push_barrier.wait(timeout=5)
            with registration_push_lock:
                return git_run(self, command, cwd=cwd, source_error=source_error)
        return git_run(self, command, cwd=cwd, source_error=source_error)

    monkeypatch.setattr(GitClient, "run", synchronize_registration_push)

    def create(index: int, recipe: str):
        return dispatchers[index].get_or_create_scenario("race", recipe).recipe

    outcomes: list[str] = []
    errors: list[BaseException] = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(create, 0, "recipe-a"),
            executor.submit(create, 1, "recipe-b"),
        )
        for future in futures:
            try:
                outcomes.append(future.result())
            except BaseException as exc:
                errors.append(exc)

    assert len(outcomes) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], ScenarioRecipeConflict)

    fresh_factory = GitLFSRepositoryBackend.factory(
        remote,
        work_dir=tmp_path / "fresh-work",
        cache_dir=tmp_path / "fresh-cache",
    )
    recovered = Dispatcher(
        RecipeRegistry(
            {
                "recipe-a": Recipe(name="recipe-a"),
                "recipe-b": Recipe(name="recipe-b"),
            },
        ),
        fresh_factory,
    ).get_or_create_scenario("race")
    assert recovered.recipe == outcomes[0]
    assert backend_factories[0]("race").current() == backend_factories[1]("race").current()
