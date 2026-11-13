from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.support.runtime_stubs import StubTrainingComponent

from reef.artifact import (
    ArtifactPublicationError,
    GitLFSRepositoryBackend,
    GitLFSRepositoryBackendFactory,
    InMemoryRepositoryBackend,
)
from reef.core import ScenarioNotFoundError
from reef.core.errors import ScenarioDeletedError
from reef.dispatcher import Dispatcher, build_default_dispatcher
from reef.recipe import Recipe
from reef.scenario.admission import ScenarioLimits
from reef.service.app import create_app
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sqlite import SQLiteScenarioStorage


def _dispatcher(tmp_path, *, allow_implicit_creation: bool = True) -> Dispatcher:
    return Dispatcher(
        Recipe(),
        InMemoryRepositoryBackend.factory(directory=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "local",
        allow_implicit_creation=allow_implicit_creation,
        scenario_storage=SQLiteScenarioStorage(None),
        coordination=memory_coordination(),
    )


def test_get_or_create_scenario_returns_same_instance(tmp_path) -> None:
    dispatcher = _dispatcher(tmp_path)
    first = dispatcher.get_or_create_scenario("code-repair")
    assert first is not None
    again = dispatcher.get_or_create_scenario("code-repair")
    assert again is first


def test_implicit_creation_off_returns_none_for_unknown_scenarios(tmp_path) -> None:
    dispatcher = _dispatcher(tmp_path, allow_implicit_creation=False)
    assert dispatcher.get_or_create_scenario("typo-name") is None


def test_implicit_creation_off_still_resolves_created_scenarios(tmp_path) -> None:
    dispatcher = _dispatcher(tmp_path, allow_implicit_creation=False)
    dispatcher.get_or_create_scenario("code-repair", allow_implicit_creation=True)
    assert dispatcher.get_or_create_scenario("code-repair").name == "code-repair"


def test_implicit_creation_on_keeps_current_behavior(tmp_path) -> None:
    dispatcher = _dispatcher(tmp_path)
    assert dispatcher.get_or_create_scenario("fresh").name == "fresh"


def test_list_scenarios_shows_loaded_bindings(tmp_path) -> None:
    dispatcher = _dispatcher(tmp_path)
    dispatcher.get_or_create_scenario("a")
    dispatcher.get_or_create_scenario("b")
    rows = dispatcher.list_scenarios()
    names = [row["scenario"] for row in rows]
    assert names == ["a", "b"]
    assert all(row["loaded"] and row["components"] for row in rows)
    assert all("release_id" not in row and "content_id" not in row for row in rows)
    assert all("recipe" not in row for row in rows)


def test_list_scenarios_empty_when_nothing_exists(tmp_path) -> None:
    assert _dispatcher(tmp_path).list_scenarios() == ()


def test_scenario_contract_returns_processor_and_request_types(tmp_path) -> None:
    dispatcher = _dispatcher(tmp_path)
    dispatcher.get_or_create_scenario("math")
    contract = dispatcher.scenario_contract("math")
    assert contract["scenario"] == "math"
    assert "recipe" not in contract
    assert contract["processor"] == "NoJobData"
    assert contract["required_request_types"] == ["inference", "report"]
    assert contract["parameters_schema"] == {}, "a recipe without a training component declares no schema"


def test_scenario_contract_serves_the_training_components_schema_by_sample_kind(tmp_path) -> None:
    class Declaring(StubTrainingComponent):
        def parameters_schema(self, kind):
            return {"title": kind} if kind == "task" else None

    dispatcher = _dispatcher(tmp_path)
    scenario = dispatcher.get_or_create_scenario("math")
    scenario._binding = replace(scenario._binding, training_component=Declaring())
    assert dispatcher.scenario_contract("math")["parameters_schema"] == {"task": {"title": "task"}}


def test_scenario_contract_raises_for_unknown_scenario(tmp_path) -> None:
    with pytest.raises(ScenarioNotFoundError):
        _dispatcher(tmp_path).scenario_contract("nope")


def test_a_deleted_scenario_leaves_the_registry_and_its_id_is_never_created_again(tmp_path) -> None:
    """A deleted scenario leaves the registry and the listing, and its lock slot goes. Its ID is not used again: an
    explicit creation and an implicit one, as an incoming record makes, are refused."""
    value = _dispatcher(tmp_path)
    assert value.get_or_create_scenario("delivery") is not None

    result = value.delete_scenario("delivery")

    assert result["scenario"] == "delivery"
    assert not value.has_scenario("delivery")
    assert "delivery" not in {row["scenario"] for row in value.list_scenarios()}
    assert value.get_or_create_scenario("delivery", allow_implicit_creation=False) is None
    with pytest.raises(ScenarioNotFoundError):
        value.delete_scenario("delivery")
    with pytest.raises(ScenarioDeletedError, match="create a scenario with a new ID"):
        value.get_or_create_scenario("delivery", allow_implicit_creation=True)
    with pytest.raises(ScenarioDeletedError):
        value.create_scenario("delivery")
    assert not value.has_scenario("delivery")
    assert value.create_scenario("delivery-2")[1]


class FailingDiscard(InMemoryRepositoryBackend):
    """An artifact store that does not answer the first discard of all content."""

    failures = 1

    def discard_all_content(self) -> None:
        if type(self).failures:
            type(self).failures -= 1
            raise ArtifactPublicationError("the artifact store did not answer")
        super().discard_all_content()


def test_a_delete_refuses_before_it_changes_anything_and_a_failed_delete_is_finished_by_the_same_call(
    tmp_path, monkeypatch
) -> None:
    factory = FailingDiscard.factory(directory=tmp_path / "artifacts")
    value = Dispatcher(
        Recipe(),
        factory,
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
        coordination=memory_coordination(),
    )
    created = value.get_or_create_scenario("delivery")
    assert created is not None
    factory("delivery").store_content("job:records", tmp_path / "artifacts")
    # A backend that cannot archive a registration, such as a remote Git repository, refuses first.
    monkeypatch.setattr(type(factory), "archives_registrations", False)
    with pytest.raises(NotImplementedError, match="cannot archive a scenario"):
        value.delete_scenario("delivery")
    assert value.get_or_create_scenario("delivery", allow_implicit_creation=False) is created
    monkeypatch.undo()
    assert not GitLFSRepositoryBackendFactory(
        GitLFSRepositoryBackend,
        "https://example.invalid/artifacts.git",
        work_dir=tmp_path / "w",
        cache_dir=tmp_path / "c",
    ).archives_registrations

    # The store does not answer the discard: the scenario keeps its storage and its registration.
    with pytest.raises(ArtifactPublicationError, match="did not answer"):
        value.delete_scenario("delivery")
    assert value.has_scenario("delivery")
    assert factory("delivery").materialize_content("job:records").is_dir()
    # The same call again finishes the delete, and no content stays.
    assert value.delete_scenario("delivery")["scenario"] == "delivery"
    assert not value.has_scenario("delivery")
    value.close()


def test_delete_scenario_refuses_names_that_are_paths() -> None:
    value = build_default_dispatcher(scenario_storage=SQLiteScenarioStorage(), coordination=memory_coordination())
    for name in ("", ".", "..", "a/b"):
        with pytest.raises(ScenarioNotFoundError):
            value.delete_scenario(name)


def test_delete_scenario_archives_its_records_and_commit_log(tmp_path) -> None:
    """With a record directory, the scenario's SQLite store and commit history move under ``archived``; the
    directory keeps nothing of the name at the top level, so a restart does not find it."""
    record_dir = tmp_path / "records"
    value = Dispatcher(
        Recipe(),
        InMemoryRepositoryBackend.factory(directory=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "local",
        scenario_storage=SQLiteScenarioStorage(record_dir),
        coordination=memory_coordination(),
    )
    assert value.get_or_create_scenario("delivery") is not None
    before = {path.name for path in record_dir.iterdir() if path.is_file()}
    assert any(name.endswith(".sqlite3") for name in before)

    result = value.delete_scenario("delivery")

    assert not [path for path in record_dir.iterdir() if path.is_file()]
    archived = list((record_dir / "archived").iterdir())
    assert len(archived) == 1 and any(path.name.endswith(".sqlite3") for path in archived[0].iterdir())
    assert all(item.startswith(str(record_dir / "archived")) for item in result["archived"])
    value.close()


@pytest.mark.unit
def test_delete_scenario_over_http() -> None:
    async def run() -> None:
        client = TestClient(
            TestServer(
                create_app(
                    build_default_dispatcher(
                        scenario_storage=SQLiteScenarioStorage(), coordination=memory_coordination()
                    )
                )
            )
        )
        await client.start_server()
        try:
            created = await client.post("/reef/scenarios", json={"name": "delivery"})
            assert created.status == 201
            deleted = await client.delete("/reef/scenarios/delivery")
            assert deleted.status == 200
            assert (await deleted.json())["scenario"] == "delivery"
            assert await (await client.get("/reef/scenarios")).json() == {"scenarios": []}
            assert (await client.delete("/reef/scenarios/delivery")).status == 404
            assert (await client.get("/reef/scenarios/delivery/artifacts")).status == 404
            again = await client.post("/reef/scenarios", json={"name": "delivery"})
            assert again.status == 409 and "was deleted" in await again.text()
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_scenario_limits_answer_409_and_503_over_http(tmp_path) -> None:
    dispatcher = Dispatcher(
        Recipe(),
        InMemoryRepositoryBackend.factory(directory=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "local",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
        scenario_admission=ScenarioLimits(max_loaded_scenarios=1, max_scenarios=2),
        coordination=memory_coordination(),
    )

    async def run() -> None:
        client = TestClient(TestServer(create_app(dispatcher)))
        await client.start_server()
        try:
            assert (await client.post("/reef/scenarios", json={"name": "a"})).status == 201
            busy = dispatcher.get_or_create_scenario("a")
            assert busy is not None
            # A running record write keeps the loaded scenario busy, so it cannot be unloaded.
            measurement = busy.operations.start("ingest/write")
            full = await client.post("/reef/scenarios", json={"name": "b"})
            assert full.status == 503
            assert full.headers["Retry-After"] == "30"
            measurement.finish(succeeded=True)
            assert (await client.post("/reef/scenarios", json={"name": "b"})).status == 201
            status = await (await client.get("/reef/status")).json()
            assert status["loaded_scenarios"] == 1
            refused = await client.post("/reef/scenarios", json={"name": "c"})
            assert refused.status == 409
            assert "reef.max_scenarios" in await refused.text()
        finally:
            await client.close()

    asyncio.run(run())
