from __future__ import annotations

import pytest

from reef.artifact import InMemoryRepositoryBackend
from reef.core import UnknownScenario
from reef.dispatcher import Dispatcher
from reef.recipe import Recipe


def _dispatcher(tmp_path, *, allow_implicit_creation: bool = True) -> Dispatcher:
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir(exist_ok=True)
    return Dispatcher(
        Recipe(),
        InMemoryRepositoryBackend.factory(bootstrap, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "local",
        agent_record_dir=None,
        allow_implicit_creation=allow_implicit_creation,
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
