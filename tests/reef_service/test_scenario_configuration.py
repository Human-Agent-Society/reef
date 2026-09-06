"""Scenario configuration is selected at creation and survives recovery."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from aiohttp.test_utils import TestClient, TestServer

from reef.artifact import ArtifactConflict
from reef.dispatcher import Dispatcher
from reef.recipe.errors import RecipeConfigError
from reef.scenario.configuration import ScenarioConfig
from reef.scenario.snapshot import SCENARIO_SNAPSHOT_METADATA_KEY
from reef.service.app import create_app

from .test_harness_proposals import _dispatcher, _recipe
from .test_manual_training import instruction


def propose(nodes, samples, models, *, requests=()):
    return None


def test_creation_config_selects_mode_and_is_immutable(tmp_path):
    dispatcher = _dispatcher(tmp_path, _recipe(tmp_path, propose))
    config = {"data": {"training_mode": "manual", "batch_size": 20}}
    try:
        scenario = dispatcher.get_or_create_scenario("s", config=config)
        assert scenario.trainer.training_mode == "manual"
        assert scenario.configuration["data"]["batch_size"] == 20
        config["data"]["batch_size"] = 3
        copy = scenario.configuration
        copy["data"]["training_mode"] = "auto"
        assert scenario.configuration["data"]["batch_size"] == 20
        assert scenario.trainer.training_mode == "manual"
        assert dispatcher.get_or_create_scenario("s", config={"data": {"batch_size": 20}}) is scenario
        with pytest.raises(ArtifactConflict, match="fixed at creation"):
            dispatcher.get_or_create_scenario("s", config=config)
        assert dispatcher.get_or_create_scenario("s") is scenario
    finally:
        dispatcher.close()


def test_configuration_value_detaches_and_freezes_nested_input():
    values = {"data": {"values": [1, {"nested": 2}]}}
    config = ScenarioConfig(values)
    values["data"]["values"][1]["nested"] = 3
    assert config.to_dict()["data"]["values"][1]["nested"] == 2
    with pytest.raises(TypeError):
        config.values["data"]["values"][1]["nested"] = 4


def test_config_survives_training_checkpoint_and_changed_deployment_defaults(tmp_path):
    recipe = _recipe(tmp_path, propose)
    first = _dispatcher(tmp_path, recipe)
    factory = first._registry._backend_factory
    try:
        scenario = first.get_or_create_scenario("s", config={"data": {"training_mode": "manual", "batch_size": 20}})
        expected = scenario.configuration
        scenario.records.append(instruction("change"))
        result = scenario.prepare_training_step()
        assert result is not None
        scenario.commit(result)
        assert scenario.repository.backend.metadata()[SCENARIO_SNAPSHOT_METADATA_KEY]["config"] == expected
    finally:
        first.close()
    recovered = Dispatcher(replace(recipe, batch_size=99), factory, agent_record_dir=tmp_path / "agent-record")
    try:
        scenario = recovered.get_or_create_scenario("s")
        assert scenario.configuration == expected
        assert scenario.trainer.training_mode == "manual"
        assert scenario.scenario_step == 1
        assert scenario.prepare_training_step() is None
        with pytest.raises(ArtifactConflict):
            recovered.get_or_create_scenario("s", config={"data": {"batch_size": 99}})
    finally:
        recovered.close()


def test_concurrent_creators_cannot_change_winning_config(tmp_path):
    dispatcher = _dispatcher(tmp_path, _recipe(tmp_path, propose))

    def create(size):
        try:
            return dispatcher.get_or_create_scenario("s", config={"data": {"batch_size": size}})
        except ArtifactConflict:
            return None

    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            results = list(workers.map(create, (2, 3)))
        assert sum(result is not None for result in results) == 1
        winner = next(result for result in results if result is not None)
        assert dispatcher.get_or_create_scenario("s").configuration == winner.configuration
    finally:
        dispatcher.close()


@pytest.mark.parametrize("config", [{"data": {"batch_size": 0}}, {"data": {"unknown": 2}}, {"data": []}, {"typo": {}}])
def test_invalid_creation_config_does_not_register_a_scenario(tmp_path, config):
    dispatcher = _dispatcher(tmp_path, _recipe(tmp_path, propose))
    try:
        with pytest.raises(ValueError):
            dispatcher.get_or_create_scenario("invalid", config=config)
        assert not dispatcher.has_scenario("invalid")
    finally:
        dispatcher.close()


def test_manual_proposer_capability_is_checked_before_registration(tmp_path):
    dispatcher = _dispatcher(tmp_path, _recipe(tmp_path, lambda n, s, m: None))
    try:
        with pytest.raises(RecipeConfigError, match="proposer"):
            dispatcher.get_or_create_scenario("invalid", config={"data": {"training_mode": "manual"}})
        assert not dispatcher.has_scenario("invalid")
    finally:
        dispatcher.close()


def test_http_creation_config_is_readable_but_not_updatable(tmp_path):
    dispatcher = _dispatcher(tmp_path, _recipe(tmp_path, propose))

    async def run():
        client = TestClient(TestServer(create_app(dispatcher)))
        await client.start_server()
        try:
            body = {"name": "s", "config": {"data": {"training_mode": "manual", "batch_size": 20}}}
            response = await client.post("/reef/scenarios", json=body)
            assert response.status == 201, await response.text()
            created = await response.json()
            response = await client.get("/reef/scenarios/s/config")
            assert response.status == 200
            assert await response.json() == created["config"]
            response = await client.post("/reef/scenarios", json=body)
            assert response.status == 200
            body["config"]["data"]["batch_size"] = 2
            response = await client.post("/reef/scenarios", json=body)
            assert response.status == 409
            response = await client.post("/reef/scenarios/s/config/updates", json={"data": {"batch_size": 3}})
            assert response.status == 404
            response = await client.get("/reef/scenarios/missing/config")
            assert response.status == 404
            assert not dispatcher.has_scenario("missing")
        finally:
            await client.close()

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()
