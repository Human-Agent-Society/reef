"""A scenario's own record-to-sample config: set over HTTP, applied by loading the scenario again."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Self

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.support.harness_service import harness_dispatcher

from reef.core import AgentRecord
from reef.core.config import config_option
from reef.dispatcher.samples import LocalSampleProducers
from reef.pipeline_context import PipelineContext
from reef.recipe import Recipe
from reef.record2sample import Record2Sample, Record2SampleSettings, SampleResult
from reef.record2sample.service import Selection
from reef.record2sample.sets import RecordSet
from reef.scenario.live_objects import LiveObjects
from reef.service.app import create_app
from reef.train.pipeline import PipelineSettings, PipelineSpec

STAGE = f"{__name__}:StalenessSamples"


@dataclass(frozen=True)
class StateSettings(Record2SampleSettings):
    state_root: str = config_option("", help="The directory under which the stage keeps each scenario's state.")


class StalenessSamples(Record2Sample):
    """A stage that reads ``max_staleness`` and keeps each scenario's state under ``state_root``. It makes no
    sample."""

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, context: PipelineContext, environ: Mapping[str, str]) -> Self:
        return cls(max_staleness=StateSettings.parse(config, plugin_class=cls).max_staleness)

    @classmethod
    def scenario_state_dirs(cls, config: Mapping[str, Any], scenario: str) -> tuple[Path, ...]:
        root = StateSettings.parse(config, plugin_class=cls).state_root
        return (Path(root) / scenario,) if root else ()

    def push(self, record: AgentRecord) -> None:
        return

    def make_samples(self, record_set: RecordSet) -> SampleResult:
        raise TypeError("StalenessSamples forms no record set")


class NoProgress(Selection):
    def progress(self, records_read: int, samples_created: int) -> None:
        return


@pytest.fixture
def dispatcher(tmp_path):
    section = {"record2sample": {"module": STAGE, "config": {"max_staleness": 1}}}
    recipe = Recipe(pipeline_stages=PipelineSettings.from_setting(section, spec=PipelineSpec(), environ={}))
    opened = harness_dispatcher(tmp_path, recipe, LiveObjects())
    opened.get_or_create_scenario("s")
    yield opened
    opened.close()


def staleness(dispatcher) -> int:
    return dispatcher.registry.require("s").trainer.pipeline.record2sample.max_staleness


def test_a_scenario_builds_its_stage_with_its_own_keys_over_the_deployment_config(dispatcher):
    url = "/reef/scenarios/s/update"

    async def run():
        async with TestClient(TestServer(create_app(dispatcher))) as client:
            response = await client.post(url, json={"record2sample_config": {"records": {"max_staleness": 3}}})
            assert response.status == 200
            assert await response.json() == {
                "scenario": "s",
                "record2sample_config": {"records": {"max_staleness": 3}},
            }
            assert staleness(dispatcher) == 3
            settings = await (await client.get("/reef/scenarios/s/settings")).json()
            assert settings["record2sample_config"] == {"records": {"max_staleness": 3}}

            # A refused config stores nothing, and the loaded instance stays.
            loaded = dispatcher.registry.require("s")
            for refused in (
                {"records": {"max_staleness": -1}},
                {"records": {"group_size": 2}},
                {"judge": {"max_staleness": 2}},
                {"records": 3},
                [],
            ):
                response = await client.post(url, json={"record2sample_config": refused})
                assert response.status == 400, refused
                assert dispatcher.registry.require("s") is loaded
            assert dispatcher.stored_settings("s").record2sample_config == {"records": {"max_staleness": 3}}

            # An empty mapping goes back to the deployment's config.
            response = await client.post(url, json={"record2sample_config": {}})
            assert response.status == 200
            assert staleness(dispatcher) == 1

    asyncio.run(run())


def test_a_samples_process_loads_the_scenario_again_when_control_changed_its_config(dispatcher):
    # Control in another process stored a new config; this process loaded the scenario before.
    changed = replace(dispatcher.stored_settings("s"), record2sample_config={"records": {"max_staleness": 2}})
    dispatcher.storage.write_settings("s", changed)
    assert staleness(dispatcher) == 1

    LocalSampleProducers(dispatcher.registry, None).produce("s", "records", NoProgress("selection-1"))

    assert staleness(dispatcher) == 2


def test_deleting_a_scenario_archives_the_state_that_its_own_config_names(dispatcher, tmp_path):
    state = tmp_path / "scenario-state"
    dispatcher.set_record2sample_config("s", {"records": {"state_root": str(state)}})
    (state / "s").mkdir(parents=True)

    dispatcher.delete_scenario("s")

    assert not (state / "s").exists()
    assert [path.name.startswith("s-") for path in (state / "archived").iterdir()] == [True]
