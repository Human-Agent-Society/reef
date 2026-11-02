"""Wait for inference startup before testing requests that need an engine."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from urllib.parse import quote

from aiohttp.test_utils import TestClient


async def wait_for_inference(
    client: TestClient,
    scenario: str,
    *,
    headers: Mapping[str, str] | None = None,
) -> None:
    """Wait for the current artifact to become ready, without sending inference requests."""
    async with asyncio.timeout(10):
        while True:
            response = await client.get(
                f"/reef/scenarios/{quote(scenario, safe='')}/inference/status", headers=headers
            )
            assert response.status == 200, await response.text()
            status = await response.json()
            if status["state"] == "ready":
                return
            assert status["state"] in ("pending", "starting"), status
            await asyncio.sleep(0.01)
