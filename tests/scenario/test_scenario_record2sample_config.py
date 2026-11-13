"""A scenario's own record-to-sample config: set over HTTP, applied by loading the scenario again."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Self

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.support.harness_service import harness_dispatcher

from reef.core import AgentRecord
from reef.core.config import config_option
from reef.core.plugins import PluginSettings
from reef.dispatcher.samples import LocalSampleProducers
from reef.pipeline_context import PipelineContext
from reef.recipe import Recipe
from reef.record2sample import Record2Sample, SampleResult
from reef.record2sample.service import Selection
from reef.record2sample.sets import RecordSet
from reef.scenario.live_objects import LiveObjects
from reef.service.app import create_app
from reef.train.pipeline import PipelineSettings, PipelineSpec

STAGE = f"{__name__}:WindowSamples"


@dataclass(frozen=True)
class StateSettings(PluginSettings):
    """``window``, a non-negative int."""

    window: int = config_option(0, help="A number that the stage keeps.")

    def __post_init__(self) -> None:
        if self.window < 0:
            raise ValueError("window must be non-negative")


class WindowSamples(Record2Sample):
    """A stage that keeps its ``window``. It makes no sample."""

    def __init__(self, window: int) -> None:
        super().__init__()
        self.window = window

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, context: PipelineContext, environ: Mapping[str, str]) -> Self:
        return cls(StateSettings.parse(config, plugin_class=cls).window)

    def push(self, record: AgentRecord) -> None:
        return

    def make_samples(self, record_set: RecordSet) -> SampleResult:
        raise TypeError("WindowSamples forms no record set")


class NoProgress(Selection):
    def progress(self, records_read: int, samples_created: int) -> None:
        return


@pytest.fixture
def dispatcher(tmp_path):
    section = {"record2sample": {"module": STAGE, "config": {"window": 1}}}
    recipe = Recipe(pipeline_stages=PipelineSettings.from_setting(section, spec=PipelineSpec(), environ={}))
    opened = harness_dispatcher(tmp_path, recipe, LiveObjects())
    opened.get_or_create_scenario("s")
    yield opened
    opened.close()


def window(dispatcher) -> int:
    stage = dispatcher.registry.require("s").trainer.pipeline.record2sample
    assert isinstance(stage, WindowSamples)
    return stage.window


def test_a_scenario_builds_its_stage_with_its_own_keys_over_the_deployment_config(dispatcher):
    url = "/reef/scenarios/s/update"

    async def run():
        async with TestClient(TestServer(create_app(dispatcher))) as client:
            response = await client.post(url, json={"record2sample_config": {"records": {"window": 3}}})
            assert response.status == 200
            assert await response.json() == {
                "scenario": "s",
                "record2sample_config": {"records": {"window": 3}},
            }
            assert window(dispatcher) == 3
            settings = await (await client.get("/reef/scenarios/s/settings")).json()
            assert settings["record2sample_config"] == {"records": {"window": 3}}

            # A refused config stores nothing, and the loaded instance stays.
            loaded = dispatcher.registry.require("s")
            for refused in (
                {"records": {"window": -1}},
                {"records": {"max_staleness": 1}},
                {"records": {"group_size": 2}},
                {"judge": {"window": 2}},
                {"records": 3},
                [],
            ):
                response = await client.post(url, json={"record2sample_config": refused})
                assert response.status == 400, refused
                assert dispatcher.registry.require("s") is loaded
            assert dispatcher.stored_settings("s").record2sample_config == {"records": {"window": 3}}

            # An empty mapping goes back to the deployment's config.
            response = await client.post(url, json={"record2sample_config": {}})
            assert response.status == 200
            assert window(dispatcher) == 1

    asyncio.run(run())


def test_a_samples_process_loads_the_scenario_again_when_control_changed_its_config(dispatcher):
    # Control in another process stored a new config; this process loaded the scenario before.
    changed = replace(dispatcher.stored_settings("s"), record2sample_config={"records": {"window": 2}})
    dispatcher.storage.write_settings("s", changed)
    assert window(dispatcher) == 1

    LocalSampleProducers(dispatcher.registry, None).produce("s", "records", NoProgress("selection-1"))

    assert window(dispatcher) == 2
