"""Shared setup for the harness route tests: a ``delivery`` scenario, a started client, and one evaluated step."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.support.harness_jobs import changed_rules, harness_recipe, rebuilt_recipe
from tests.support.harness_service import EchoInference, harness_dispatcher

from reef.dispatcher import Dispatcher
from reef.harness.tasks.run import TaskResult
from reef.harness.tree import HarnessTree, TreePatch
from reef.recipe.harness import HarnessEvolutionRecipe
from reef.service.app import create_app
from reef.train.pipeline import PipelineSettings

#: Each patch strictly raises the marker count, so every evaluated step wins and publishes.
PATCHES = (TreePatch(changed_rules("marker rules\n")), TreePatch(changed_rules("marker marker rules\n")))
#: The default seed: no files, so nothing is served before the first publish.
EMPTY_SEED = HarnessTree({})
HEADERS = {"x-reef-scenario": "delivery"}
UPDATE_TIMEOUT_S = 10.0
UPDATE_POLL_S = 0.01


def evaluate(task: str, result: TaskResult) -> float:
    return float(result.trajectory[-1]["rules"].count("marker"))


def channel_dispatcher(
    tmp_path: Path,
    patches,
    *,
    seed: HarnessTree = EMPTY_SEED,
    inference_stages: bool = False,
    job_samples: int = 1,
) -> Dispatcher:
    """A ``delivery`` scenario whose method makes ``patches`` in order, then nothing; its root snapshot is
    ``seed``, and each step takes ``job_samples`` traces."""
    candidates = iter(patches)
    stages: dict = (
        {
            "record2sample": {"module": "reef.record2sample.traces:InferenceSamples"},
        }
        if inference_stages
        else {}
    )
    stages["sample2job"] = {"config": {"job_samples": job_samples}}
    recipe, live = rebuilt_recipe(
        *harness_recipe(
            tmp_path,
            lambda workspace, services: next(candidates, None),
            seed=seed,
            score=evaluate,
        ),
        pipeline_stages=PipelineSettings.from_setting(stages, spec=HarnessEvolutionRecipe.pipeline_spec(), environ={}),
    )
    dispatcher = harness_dispatcher(tmp_path, recipe, live)
    dispatcher.get_or_create_scenario("delivery")
    return dispatcher


@asynccontextmanager
async def served(dispatcher: Dispatcher) -> AsyncIterator[TestClient]:
    """A started test client for the real app over ``dispatcher``, closed on exit."""
    client = TestClient(TestServer(create_app(dispatcher, inference_handler=EchoInference())))
    await client.start_server()
    try:
        deadline = asyncio.get_running_loop().time() + UPDATE_TIMEOUT_S
        while True:
            response = await client.get("/reef/scenarios/delivery/inference/status")
            status = await response.json()
            if response.status == 200 and status["state"] == "ready":
                break
            if asyncio.get_running_loop().time() >= deadline:
                pytest.fail(f"inference did not start: {status}")
            await asyncio.sleep(UPDATE_POLL_S)
        yield client
    finally:
        await client.close()


async def scored_exchange(client: TestClient) -> None:
    """One inference and its failing report: a batch of one, which wakes one step."""
    body = {"messages": [{"role": "user", "content": "hi"}]}
    response = await client.post("/v1/chat/completions", headers=HEADERS, json=body)
    assert response.status == 200
    receipt = response.headers["x-reef-agent-record-id"]
    response = await client.post("/reef/report", headers=HEADERS, json={"score": 0.0, "references": [receipt]})
    assert response.status == 200


async def evaluated_step(client: TestClient) -> dict:
    """Drive one evaluated step through the wire; the manifest of the snapshot it published."""
    response = await client.get("/reef/harness", headers=HEADERS)
    previous = (await response.json())["artifact_id"] if response.status == 200 else None
    await scored_exchange(client)
    deadline = asyncio.get_running_loop().time() + UPDATE_TIMEOUT_S
    while True:
        response = await client.get("/reef/harness", headers=HEADERS)
        if response.status == 200:
            manifest = await response.json()
            if manifest["artifact_id"] != previous:
                assert manifest["evaluation"]["selected"] is True
                return manifest
        else:
            assert response.status == 404
            await response.read()
        if asyncio.get_running_loop().time() >= deadline:
            pytest.fail("the harness update did not commit")
        await asyncio.sleep(UPDATE_POLL_S)
