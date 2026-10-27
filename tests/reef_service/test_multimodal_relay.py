"""Multimodal calls: the service relays them through the scenario's recipe, and refuses them without a relay.

The relay of a configured gateway is tested with the recipe that ships one
(``tests/recipes/reefine/test_reefine_multimodal_relay.py``).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest
from aiohttp.test_utils import TestClient, TestServer

from reef.artifact import InMemoryRepositoryBackend
from reef.dispatcher import Dispatcher
from reef.recipe import Recipe
from reef.runtime.interfaces import MultimodalRelay
from reef.service.app import create_app
from reef.storage.sqlite import SQLiteScenarioStorage


@dataclass(frozen=True, kw_only=True)
class RelayRecipe(Recipe):
    """The record-only recipe, offering ``relay`` for multimodal calls."""

    relay: MultimodalRelay | None = field(default=None)

    @property
    def multimodal_relay(self) -> MultimodalRelay | None:
        return self.relay


async def reef_client(tmp_path, relay: MultimodalRelay | None) -> tuple[TestClient, Dispatcher]:
    initial = tmp_path / "initial"
    initial.mkdir(exist_ok=True)
    dispatcher = Dispatcher(
        RelayRecipe(relay=relay),
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(),
    )
    client = TestClient(TestServer(create_app(dispatcher)))
    await client.start_server()
    return client, dispatcher


@pytest.mark.unit
def test_a_recipe_without_a_relay_serves_no_multimodal_call(tmp_path) -> None:
    async def run() -> None:
        client, _ = await reef_client(tmp_path, None)
        try:
            response = await client.post("/v1/images", headers={"x-reef-scenario": "media"}, json={"model": "img"})
            assert response.status == 501 and "relays no multimodal calls" in await response.text()
        finally:
            await client.close()

    asyncio.run(run())
