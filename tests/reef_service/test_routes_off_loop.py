"""Route handlers that wait for a scenario's lock do so off the event loop."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from reef.service.app import create_app

from .test_component_trainers import _dispatcher


@pytest.mark.unit
def test_a_catalog_read_waiting_for_the_scenario_lock_leaves_the_loop_free(tmp_path: Path) -> None:
    """A commit holds the scenario's lock for as long as it takes; other routes answer meanwhile."""
    dispatcher, _ = _dispatcher(tmp_path)
    held = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with dispatcher._registry.lock_for("agent"):
            held.set()
            release.wait(10)

    holder = threading.Thread(target=hold, daemon=True)
    try:
        assert dispatcher.get_or_create_scenario("agent") is not None
        holder.start()
        assert held.wait(5)

        async def run() -> None:
            client = TestClient(TestServer(create_app(dispatcher)))
            await client.start_server()
            try:
                catalog = asyncio.ensure_future(client.get("/reef/scenarios/agent/releases"))
                await asyncio.sleep(0.2)
                assert not catalog.done()
                health = await asyncio.wait_for(client.get("/healthz"), 2)
                assert health.status == 200
                release.set()
                assert (await catalog).status == 200
            finally:
                await client.close()

        asyncio.run(run())
    finally:
        release.set()
        holder.join(5)
        dispatcher.close()
