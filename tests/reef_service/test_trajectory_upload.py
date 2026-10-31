"""Trajectory items (RFC #1028) that a scenario uses train without records."""

from __future__ import annotations

import asyncio
import json

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.support.trajectory_uploads import (
    DATA,
    HEADERS,
    TOKEN,
    RecordOnlyJobData,
    RenderingSamples,
    UploadRecipe,
    atif,
    dispatcher_for,
    trajectory_lines,
    upload_trajectories,
)

from reef.dispatcher import Dispatcher
from reef.service.app import create_app
from reef.train.types import TrajectorySample

AUTH = {"Authorization": f"Bearer {TOKEN}"}


async def wait_for_batches(recipe: UploadRecipe, count: int) -> None:
    async with asyncio.timeout(5):
        while len(recipe.batches) < count:
            await asyncio.sleep(0.01)
    assert len(recipe.batches) == count


async def create_scenario(client: TestClient) -> None:
    response = await client.post("/reef/scenarios", headers=AUTH, json={"name": "s"})
    assert response.status in (200, 201), await response.text()


def sample_ids(dispatcher: Dispatcher) -> list[str]:
    return [stored.sample_id for stored in dispatcher.get_or_create_scenario("s").store.samples.page()]


def test_trajectory_items_train_once_also_after_a_restart(tmp_path):
    recipe = UploadRecipe()

    async def create_and_train(dispatcher: Dispatcher) -> None:
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            await create_scenario(client)
            response = await upload_trajectories(client, AUTH, "t1", "t2")
            assert response.status == 201, await response.text()
            body = await response.json()
            assert [item["id"] for item in body["items"]] == ["t1", "t2"]
            assert body["items"][0]["metadata"]["scenarios"] == {"s": "train"}
            await wait_for_batches(recipe, 1)

            # A retry after training changes nothing. The same ID with another trajectory conflicts.
            response = await upload_trajectories(client, AUTH, "t1")
            assert response.status == 201
            conflict = json.dumps({**atif("other"), "session_id": "t1"}).encode()
            response = await upload_trajectories(client, AUTH, lines=conflict)
            assert response.status == 409

    async def create_after_restart(dispatcher: Dispatcher) -> None:
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            response = await upload_trajectories(client, AUTH, "t3", "t4")
            assert response.status == 201, await response.text()
            await create_scenario(client)
            await wait_for_batches(recipe, 2)

    dispatcher = dispatcher_for(tmp_path, recipe)
    try:
        asyncio.run(create_and_train(dispatcher))
        assert dispatcher.get_or_create_scenario("s").records.count("s") == 0
    finally:
        dispatcher.close()
    dispatcher = dispatcher_for(tmp_path, recipe)
    try:
        asyncio.run(create_after_restart(dispatcher))
    finally:
        dispatcher.close()

    first, second = recipe.batches
    assert first.samples == (TrajectorySample(atif("t1")), TrajectorySample(atif("t2")))
    assert second.samples == (TrajectorySample(atif("t3")), TrajectorySample(atif("t4")))


def test_a_scenario_uses_a_trajectory_only_through_a_recipe_that_accepts_uploaded_trajectories(tmp_path):
    dispatcher = dispatcher_for(tmp_path, UploadRecipe(batch_stage=RecordOnlyJobData))

    async def run():
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            await create_scenario(client)
            response = await upload_trajectories(client, AUTH, "trajectory-1")
            assert response.status == 400
            assert "does not accept uploaded trajectories" in await response.text()
            assert dispatcher.data_store.query("s") == ()
            assert sample_ids(dispatcher) == []

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_an_invalid_upload_is_refused_and_stores_nothing(tmp_path):
    dispatcher = dispatcher_for(tmp_path, UploadRecipe())

    def line(**fields) -> bytes:
        return json.dumps({**atif("t"), **fields}).encode() + b"\n"

    async def run():
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            response = await upload_trajectories(client, {}, "t")
            assert response.status == 401
            for lines in (
                b"",
                b"not json\n",
                b"[1]\n",
                line(session_id=""),
                line(steps=[]),
                line(schema_version="v1"),
                line() + line(),
                trajectory_lines(*(f"t{index}" for index in range(1001))),
            ):
                response = await upload_trajectories(client, AUTH, lines=lines)
                assert response.status == 400, (lines[:80], await response.text())
            for metadata in ({"kind": "task"}, {"kind": "trajectory", "scenarios": {"s": "eval"}}):
                response = await client.post(
                    f"{DATA}/trajectories",
                    headers={**AUTH, "Content-Type": "application/x-ndjson"},
                    params={"metadata": json.dumps(metadata)},
                    data=trajectory_lines("t"),
                )
                assert response.status == 400, (metadata, await response.text())
            # Trajectories come as JSON Lines alone: not as a JSON body, and not with another content type.
            body = {"items": [{"id": "t", "content": atif("t"), "metadata": {"kind": "trajectory"}}]}
            refused = await client.post(DATA, headers=AUTH, json=body)
            assert refused.status == 400 and "JSON Lines" in await refused.text()
            wrong = await client.post(f"{DATA}/trajectories", headers=AUTH, json=atif("t"))
            assert wrong.status == 415
            assert dispatcher.data_store.query("s") == ()

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


@pytest.mark.parametrize("route", ("/reef/records", "/reef/records/batch"))
def test_the_record_import_does_not_take_a_trajectory(tmp_path, route):
    dispatcher = dispatcher_for(tmp_path, UploadRecipe())
    record = {"agent_record_id": "trajectory-1", "request_type": "trajectory", "payload": atif("trajectory-1")}

    async def run():
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            body = {"records": [record]} if route.endswith("batch") else record
            response = await client.post(route, headers=HEADERS, json=body)
            assert response.status == 400

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_a_trajectory_that_cannot_train_is_refused_and_jobs_get_the_rendered_samples(tmp_path):
    recipe = UploadRecipe(task_stage=RenderingSamples)
    dispatcher = dispatcher_for(tmp_path, recipe)

    async def run():
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            await create_scenario(client)
            response = await upload_trajectories(client, AUTH, "good", "bad")
            assert response.status == 400
            assert "trajectory 'bad' cannot train: it has no agent step" in await response.text()
            assert dispatcher.data_store.query("s") == (), "nothing of it is stored"
            response = await upload_trajectories(client, AUTH, "t1", "t2")
            assert response.status == 201, await response.text()
            await wait_for_batches(recipe, 1)

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()
    (batch,) = recipe.batches
    assert [item.training for item in batch.samples] == [
        {"tokens": [1, 2], "loss_mask": [1], "rollout_log_probs": []}
    ] * 2


def test_an_item_that_leaves_a_scenario_leaves_its_samples_and_the_item_stays(tmp_path):
    dispatcher = dispatcher_for(tmp_path, UploadRecipe())

    async def run():
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            # Items that name no scenario, or a scenario that does not exist yet, wait in the data store.
            response = await upload_trajectories(client, AUTH, "t1", "t2", scenario=None)
            assert response.status == 201
            await create_scenario(client)
            assert sample_ids(dispatcher) == []
            response = await client.post(
                f"{DATA}/metadata", headers=AUTH, json={"ids": ["t1", "t2"], "patch": {"scenarios": {"s": "train"}}}
            )
            assert response.status == 200, await response.text()
            assert sample_ids(dispatcher) == ["t1", "t2"]
            response = await client.post(
                f"{DATA}/metadata", headers=AUTH, json={"ids": ["t1"], "patch": {"scenarios": {"s": None}}}
            )
            assert response.status == 200
            assert sample_ids(dispatcher) == ["t2"]
            response = await client.delete(f"{DATA}/t2", headers=AUTH)
            assert response.status == 200
            assert sample_ids(dispatcher) == []
            # The scenario's artifacts are in the data store too, as artifact items.
            response = await client.get(f"{DATA}?kind=trajectory", headers=AUTH)
            assert [item["id"] for item in (await response.json())["items"]] == ["t1"]
            response = await client.get(f"{DATA}/t1", headers=AUTH)
            assert (await response.json())["content"] == atif("t1")
            assert (await client.get(f"{DATA}/t2", headers=AUTH)).status == 404

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()
