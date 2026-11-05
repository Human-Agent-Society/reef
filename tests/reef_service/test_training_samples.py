"""The sample listing, the next automatic job, and requests that name the samples that they train."""

from __future__ import annotations

import asyncio
from dataclasses import replace

from aiohttp.test_utils import TestClient, TestServer
from tests.support.trajectory_uploads import HEADERS, TOKEN, UploadRecipe, dispatcher_for, upload_trajectories

from reef.service.app import create_app

AUTH = {"Authorization": HEADERS["Authorization"]}


def ids(listing: dict) -> list[str]:
    return [sample["sample_id"] for sample in listing["samples"]]


def train_body(record_id: str, *samples: str) -> dict:
    body = {"agent_record_id": record_id, "text": "train these", "session": "session-1", "release_id": "release-1"}
    return {**body, "samples": list(samples)} if samples else body


def test_a_client_lists_the_samples_and_a_request_trains_the_ones_that_it_names(tmp_path):
    recipe = UploadRecipe()
    dispatcher = dispatcher_for(tmp_path, recipe)
    train = {"x-reef-scenario": "s", **AUTH}

    async def run() -> None:
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            await client.post("/reef/scenarios", headers=HEADERS, json={"name": "s"})
            response = await upload_trajectories(client, HEADERS, "t1")
            assert response.status == 201, await response.text()
            listing = await (await client.get("/reef/scenarios/s/samples", headers=AUTH)).json()
            assert [(sample["sample_id"], sample["kind"], sample["component"]) for sample in listing["samples"]] == [
                ("t1", "trajectory", None)
            ]
            assert listing["next"] is None
            next_job = await (await client.get("/reef/scenarios/s/next-job", headers=AUTH)).json()
            assert (next_job["training_mode"], ids(next_job)) == ("auto", ["t1"])

            switched = await client.post("/reef/scenarios/s/update", headers=AUTH, json={"training_mode": "manual"})
            assert switched.status == 200, await switched.text()
            await upload_trajectories(client, HEADERS, "t2", "t3")
            for body in (
                train_body("r0", "missing"),
                {**train_body("r0"), "samples": []},
                train_body("r0"),
                {**train_body("r0"), "samples": ["t1", "t1"]},
            ):
                refused = await client.post("/reef/train", headers=train, json=body)
                assert refused.status == 400, await refused.text()

            # The job trains the named samples in store order; the others wait.
            trained = await client.post("/reef/train", headers=train, json=train_body("r1", "t3", "t1"))
            assert trained.status == 200, await trained.text()
            async with asyncio.timeout(5):
                while not recipe.batches:
                    await asyncio.sleep(0.01)
            assert [sample.trajectory["session_id"] for sample in recipe.batches[0].samples] == ["t1", "t3"]
            async with asyncio.timeout(5):
                while True:
                    listing = await (await client.get("/reef/scenarios/s/samples", headers=AUTH)).json()
                    counts = {sample["sample_id"]: sample["train_counts"] for sample in listing["samples"]}
                    if counts["t1"]:
                        break
                    await asyncio.sleep(0.01)
            assert counts == {"t1": {"": 1}, "t2": {}, "t3": {"": 1}}
            # A trained sample can train again.
            again = await client.post("/reef/train", headers=train, json=train_body("r2", "t1"))
            assert again.status == 200, await again.text()
            async with asyncio.timeout(5):
                while len(recipe.batches) < 2:
                    await asyncio.sleep(0.01)
            assert [sample.trajectory["session_id"] for sample in recipe.batches[1].samples] == ["t1"]

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_a_listing_pages_through_the_store(tmp_path):
    dispatcher = dispatcher_for(tmp_path, replace(UploadRecipe()))

    async def run() -> None:
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            await client.post("/reef/scenarios", headers=HEADERS, json={"name": "s"})
            await upload_trajectories(client, HEADERS, "t1", "t2", "t3")
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


def test_sample_processing_reads_retained_tasks_and_rejects_cross_scenario_details(tmp_path):
    from reef.storage.work_queue import NewWork

    dispatcher = dispatcher_for(tmp_path, UploadRecipe())

    async def run():
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            await client.post("/reef/scenarios", headers=HEADERS, json={"name": "s"})
            # An unknown work kind remains in the queue, without a worker racing these reads.
            queue = dispatcher.queue
            first = queue.add(NewWork("samples", "first", "s", "inspection", {"component": "weights"}))
            queue.add(NewWork("samples", "last", "s", "inspection", {"component": "weights"}))
            queue.add(NewWork("samples", "foreign", "other", "inspection"))
            queue.cancel("samples", first.id, "test failure")
            path = "/reef/scenarios/s/sample-processing"
            response = await client.get(path + "?limit=1", headers=AUTH)
            assert response.status == 200, await response.text()
            listing = await response.json()
            assert listing["tasks"][0]["id"] == "last"
            assert listing["counts"]["failed"] == 1
            assert listing["retention_seconds"] == 604800
            next_page = await (await client.get(path + f'?before={listing["next"]}&limit=1', headers=AUTH)).json()
            assert next_page["tasks"][0]["id"] == "first"
            assert next_page["tasks"][0]["samples_created"] is None
            assert (await client.get(path + "/foreign", headers=AUTH)).status == 404
            assert (await client.get(path + "/missing", headers=AUTH)).status == 404
            assert (await client.get(path + "/first", headers=AUTH)).status == 200
            for query in ("limit=0", "before=0", "state=bad", "limit=x"):
                assert (await client.get(path + "?" + query, headers=AUTH)).status == 400
            assert (await client.get(path)).status == 401

    asyncio.run(run())
