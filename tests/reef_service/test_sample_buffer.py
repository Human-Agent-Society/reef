"""The sample listing and the buffer routes: a client sees the samples and, in manual mode, chooses what trains."""

from __future__ import annotations

import asyncio
from dataclasses import replace

from aiohttp.test_utils import TestClient, TestServer
from tests.reef_service.test_trajectory_upload import HEADERS, TOKEN, UploadRecipe, dispatcher_for, uploaded

from reef.service.app import create_app

AUTH = {"Authorization": HEADERS["Authorization"]}


def ids(listing: dict) -> list[str]:
    return [sample["sample_id"] for sample in listing["samples"]]


def instruction(record_id: str) -> dict[str, str]:
    return {"agent_record_id": record_id, "text": "train these", "session": "session-1", "release_id": "release-1"}


def test_a_client_lists_the_samples_and_edits_the_buffer_only_in_manual_mode(tmp_path):
    recipe = UploadRecipe()
    dispatcher = dispatcher_for(tmp_path, recipe)

    async def run() -> None:
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            response = await client.post("/reef/trajectories", headers=HEADERS, json=uploaded("t1"))
            assert response.status == 200, await response.text()
            listing = await (await client.get("/reef/scenarios/s/samples", headers=AUTH)).json()
            assert [(sample["sample_id"], sample["kind"], sample["component"]) for sample in listing["samples"]] == [
                ("t1", "trajectory", None)
            ]
            assert listing["next"] is None

            # In auto, the buffer is read-only: what the next job takes.
            buffer = await (await client.get("/reef/scenarios/s/buffer", headers=AUTH)).json()
            assert (buffer["training_mode"], ids(buffer)) == ("auto", ["t1"])
            refused = await client.post("/reef/scenarios/s/buffer/insert", headers=AUTH, json={"sample_ids": ["t1"]})
            assert refused.status == 409

            # A switch to manual starts from the auto buffer; new samples do not join it by themselves.
            switched = await client.post("/reef/scenarios/s/update", headers=AUTH, json={"training_mode": "manual"})
            assert switched.status == 200, await switched.text()
            await client.post("/reef/trajectories", headers=HEADERS, json=uploaded("t2", "t3"))
            assert ids(await (await client.get("/reef/scenarios/s/buffer", headers=AUTH)).json()) == ["t1"]
            inserted = await client.post(
                "/reef/scenarios/s/buffer/insert", headers=AUTH, json={"sample_ids": ["t3", "t2"]}
            )
            assert ids(await inserted.json()) == ["t1", "t2", "t3"]
            removed = await client.post("/reef/scenarios/s/buffer/remove", headers=AUTH, json={"sample_ids": ["t2"]})
            assert ids(await removed.json()) == ["t1", "t3"]
            for body, status in (({"sample_ids": ["missing"]}, 400), ({"sample_ids": []}, 400), ({"ids": []}, 400)):
                response = await client.post("/reef/scenarios/s/buffer/insert", headers=AUTH, json=body)
                assert response.status == status, await response.text()

            # An instruction trains the whole buffer, which the commit empties.
            trained = await client.post(
                "/reef/train", headers={"x-reef-scenario": "s", **AUTH}, json=instruction("r1")
            )
            assert trained.status == 200, await trained.text()
            async with asyncio.timeout(5):
                while not recipe.batches:
                    await asyncio.sleep(0.01)
            assert [sample.trajectory["session_id"] for sample in recipe.batches[0].samples] == ["t1", "t3"]
            async with asyncio.timeout(5):
                while ids(await (await client.get("/reef/scenarios/s/buffer", headers=AUTH)).json()):
                    await asyncio.sleep(0.01)
            listing = await (await client.get("/reef/scenarios/s/samples", headers=AUTH)).json()
            assert {sample["sample_id"]: sample["train_counts"] for sample in listing["samples"]} == {
                "t1": {"": 1},
                "t2": {},
                "t3": {"": 1},
            }
            # A trained sample can go into the buffer again.
            again = await client.post("/reef/scenarios/s/buffer/insert", headers=AUTH, json={"sample_ids": ["t1"]})
            assert [(sample["sample_id"], sample["train_count"]) for sample in (await again.json())["samples"]] == [
                ("t1", 1)
            ]

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_a_listing_pages_through_the_store(tmp_path):
    dispatcher = dispatcher_for(tmp_path, replace(UploadRecipe()))

    async def run() -> None:
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            await client.post("/reef/trajectories", headers=HEADERS, json=uploaded("t1", "t2", "t3"))
            first = await (await client.get("/reef/scenarios/s/samples?limit=2", headers=AUTH)).json()
            assert ids(first) == ["t1", "t2"] and first["next"] is not None
            rest = await (await client.get(f"/reef/scenarios/s/samples?after={first['next']}", headers=AUTH)).json()
            assert ids(rest) == ["t3"] and rest["next"] is None
            bad = await client.get("/reef/scenarios/s/samples?limit=0", headers=AUTH)
            assert bad.status == 400

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_a_reload_in_manual_mode_keeps_the_manual_buffer_as_the_client_left_it(tmp_path):
    recipe = UploadRecipe()

    async def switch(dispatcher) -> None:
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            await client.post("/reef/trajectories", headers=HEADERS, json=uploaded("t1"))
            await client.post("/reef/scenarios/s/update", headers=AUTH, json={"training_mode": "manual"})
            await client.post("/reef/trajectories", headers=HEADERS, json=uploaded("t2"))

    async def read(dispatcher) -> dict:
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            return await (await client.get("/reef/scenarios/s/buffer", headers=AUTH)).json()

    dispatcher = dispatcher_for(tmp_path, recipe)
    try:
        asyncio.run(switch(dispatcher))
    finally:
        dispatcher.close()
    dispatcher = dispatcher_for(tmp_path, recipe)
    try:
        buffer = asyncio.run(read(dispatcher))
    finally:
        dispatcher.close()
    # The stored mode is restored; the upload that came after the switch did not join the manual buffer.
    assert (buffer["training_mode"], ids(buffer)) == ("manual", ["t1"])


def test_an_instruction_that_waits_for_an_empty_buffer_trains_once_a_client_inserts_samples(tmp_path):
    recipe = UploadRecipe()
    dispatcher = dispatcher_for(tmp_path, recipe)

    async def run() -> None:
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            await client.post("/reef/trajectories", headers=HEADERS, json=uploaded("t0"))
            await client.post("/reef/scenarios/s/update", headers=AUTH, json={"training_mode": "manual"})
            await client.post("/reef/scenarios/s/buffer/remove", headers=AUTH, json={"sample_ids": ["t0"]})
            await client.post("/reef/trajectories", headers=HEADERS, json=uploaded("t1"))
            queued = await client.post("/reef/train", headers={"x-reef-scenario": "s", **AUTH}, json=instruction("r1"))
            assert queued.status == 200, await queued.text()
            await asyncio.sleep(0.2)
            assert not recipe.batches, "the instruction waits while the manual buffer is empty"
            await client.post("/reef/scenarios/s/buffer/insert", headers=AUTH, json={"sample_ids": ["t1"]})
            async with asyncio.timeout(5):
                while not recipe.batches:
                    await asyncio.sleep(0.01)
            assert [sample.trajectory["session_id"] for sample in recipe.batches[0].samples] == ["t1"]

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()
