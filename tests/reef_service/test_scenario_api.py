from __future__ import annotations

import asyncio

import pytest
from aiohttp.test_utils import TestClient, TestServer

from reef.artifact import InMemoryRepositoryBackend
from reef.core import UnknownScenario
from reef.dispatcher import Dispatcher, build_default_dispatcher
from reef.recipe import Recipe
from reef.service.app import create_app
from reef.storage.scenario import SQLiteScenarioStorage


def _dispatcher(tmp_path, *, allow_implicit_creation: bool = True) -> Dispatcher:
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir(exist_ok=True)
    return Dispatcher(
        Recipe(),
        InMemoryRepositoryBackend.factory(bootstrap, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "local",
        agent_record_dir=None,
        allow_implicit_creation=allow_implicit_creation,
        scenario_storage=SQLiteScenarioStorage(None),
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
    assert all(row["loaded"] and row["release_id"] for row in rows)
    assert all("recipe" not in row for row in rows)


def test_list_scenarios_empty_when_nothing_exists(tmp_path) -> None:
    assert _dispatcher(tmp_path).list_scenarios() == ()


def test_scenario_contract_returns_processor_and_request_types(tmp_path) -> None:
    dispatcher = _dispatcher(tmp_path)
    dispatcher.get_or_create_scenario("math")
    contract = dispatcher.scenario_contract("math")
    assert contract["scenario"] == "math"
    assert "recipe" not in contract
    assert contract["processor"] == "DataProcessor"
    assert contract["required_request_types"] == ["inference", "report"]


def test_scenario_contract_raises_for_unknown_scenario(tmp_path) -> None:
    with pytest.raises(UnknownScenario):
        _dispatcher(tmp_path).scenario_contract("nope")


def test_delete_scenario_forgets_it_and_frees_the_name(tmp_path) -> None:
    """A deleted scenario leaves the registry and the listing, its lock slot goes, and the name creates fresh."""
    value = _dispatcher(tmp_path)
    created = value.get_or_create_scenario("delivery")
    assert created is not None
    first_release = created.repository.require_current_artifact().release_id

    result = value.delete_scenario("delivery")

    assert result["scenario"] == "delivery"
    assert not value.has_scenario("delivery")
    assert "delivery" not in {row["scenario"] for row in value.list_scenarios()}
    assert value.get_or_create_scenario("delivery", allow_implicit_creation=False) is None
    with pytest.raises(UnknownScenario):
        value.delete_scenario("delivery")
    # The name is free: a new scenario under it starts from the base, not the dropped chain.
    again = value.get_or_create_scenario("delivery")
    assert again is not None and again is not created
    assert again.repository.require_current_artifact().release_id == first_release or again.scenario_step == 0


def test_delete_scenario_refuses_names_that_are_paths() -> None:
    value = build_default_dispatcher(scenario_storage=SQLiteScenarioStorage())
    for name in ("", ".", "..", "a/b"):
        with pytest.raises(UnknownScenario):
            value.delete_scenario(name)


def test_delete_scenario_archives_its_records_and_commit_log(tmp_path) -> None:
    """With a record directory, the scenario's SQLite store and commit log move under ``archived``; the
    directory keeps nothing of the name at the top level, so a restart does not find it."""
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    record_dir = tmp_path / "records"
    value = Dispatcher(
        Recipe(),
        InMemoryRepositoryBackend.factory(bootstrap, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "local",
        agent_record_dir=record_dir,
        scenario_storage=SQLiteScenarioStorage(record_dir),
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
        client = TestClient(TestServer(create_app(build_default_dispatcher(scenario_storage=SQLiteScenarioStorage()))))
        await client.start_server()
        try:
            created = await client.post("/reef/scenarios", json={"name": "delivery"})
            assert created.status == 201
            deleted = await client.delete("/reef/scenarios/delivery")
            assert deleted.status == 200
            assert (await deleted.json())["scenario"] == "delivery"
            assert await (await client.get("/reef/scenarios")).json() == {"scenarios": []}
            assert (await client.delete("/reef/scenarios/delivery")).status == 404
            assert (await client.get("/reef/scenarios/delivery/releases")).status == 404
        finally:
            await client.close()

    asyncio.run(run())
