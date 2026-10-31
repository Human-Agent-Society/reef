"""The data routes (RFC #1028): each scenario uses only the items of its owner."""

from __future__ import annotations

import asyncio
import io
import tarfile
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer
from tests.support.task_service_stubs import task_for
from tests.support.trajectory_uploads import TOKEN, UploadRecipe, dispatcher_for, upload_trajectories

from reef.artifact import InMemoryRepositoryBackend
from reef.core.tasks import write_harbor_task
from reef.dispatcher import Dispatcher
from reef.service.app import create_app
from reef.storage.sqlite import SQLiteScenarioStorage

AUTH = {"Authorization": f"Bearer {TOKEN}"}


def served(dispatcher: Dispatcher, check) -> None:
    async def run() -> None:
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            await check(client)

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_a_scenario_created_with_an_owner_uses_that_owners_items_only(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, UploadRecipe())

    async def check(client: TestClient) -> None:
        response = await client.post("/reef/scenarios", headers=AUTH, json={"name": "math", "owner": "acme"})
        assert response.status == 201, await response.text()
        assert (await response.json())["owner"] == "acme"
        created = await upload_trajectories(client, AUTH, "t1", scenario="math", base="/reef/namespaces/acme/data")
        assert created.status == 201, await created.text()
        listing = await (await client.get("/reef/scenarios/math/samples", headers=AUTH)).json()
        assert [sample["sample_id"] for sample in listing["samples"]] == ["t1"]
        # Another owner's items cannot name the scenario.
        refused = await upload_trajectories(client, AUTH, "t2", scenario="math", base="/reef/namespaces/globex/data")
        assert refused.status == 400
        assert "has the owner 'acme'" in await refused.text()
        # The owner is set at creation: a scenario that holds data keeps it.
        moved = await client.post("/reef/scenarios", headers=AUTH, json={"name": "math", "owner": "globex"})
        assert moved.status == 400
        assert (await client.post("/reef/scenarios", headers=AUTH, json={"name": "x", "owner": "a/b"})).status == 400

    served(dispatcher, check)


def test_a_query_pages_by_sequence_and_a_task_item_reads_without_content(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, UploadRecipe())
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add(write_harbor_task(task_for("task-a", "record-1", 9000), tmp_path / "tasks"), arcname="task-a")

    async def check(client: TestClient) -> None:
        base = "/reef/namespaces/acme/data"
        await upload_trajectories(client, AUTH, "t1", "t2", "t3", scenario=None, base=base)
        response = await client.post(
            f"{base}/tasks", headers={**AUTH, "Content-Type": "application/gzip"}, data=buffer.getvalue()
        )
        assert response.status == 201, await response.text()
        first = await (await client.get(f"{base}?limit=2", headers=AUTH)).json()
        assert [item["id"] for item in first["items"]] == ["t1", "t2"]
        rest = await (await client.get(f"{base}?after={first['next']}", headers=AUTH)).json()
        assert ([item["id"] for item in rest["items"]], rest["next"]) == (["t3", "task-a"], None)
        tasks = await (await client.get(f"{base}?kind=task", headers=AUTH)).json()
        assert [item["id"] for item in tasks["items"]] == ["task-a"]
        task = await (await client.get(f"{base}/task-a", headers=AUTH)).json()
        assert (task["content"], task["metadata"]["kind"]) == (None, "task")
        for query in ("?kind=dataset", "?after=x", "?limit=0"):
            assert (await client.get(base + query, headers=AUTH)).status == 400, query
        assert (await client.delete(f"{base}/missing", headers=AUTH)).status == 404

    served(dispatcher, check)


def test_a_deployment_without_a_data_store_serves_no_data_route(tmp_path: Path) -> None:
    (tmp_path / "initial").mkdir()
    dispatcher = Dispatcher(
        UploadRecipe(),
        InMemoryRepositoryBackend.factory(tmp_path / "initial", root=tmp_path / "repository"),
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
    )

    async def check(client: TestClient) -> None:
        response = await client.get("/reef/namespaces/acme/data", headers=AUTH)
        assert response.status == 503
        assert "no data store" in await response.text()

    served(dispatcher, check)


def test_a_client_record_item_needs_the_content_and_id_of_a_record(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, UploadRecipe())
    content = {
        "agent_record_id": "r1",
        "request_type": "report",
        "payload": {"score": 1.0},
        "created_at": 1.0,
        "references": [],
        "artifact_ref": None,
    }
    metadata = {"kind": "record", "origin": {"scenario": "old", "release_id": None, "artifact_id": None}}

    async def check(client: TestClient) -> None:
        good = {"items": [{"id": "old/r1", "content": content, "metadata": metadata}]}
        assert (await client.post("/reef/namespaces/acme/data", headers=AUTH, json=good)).status == 201
        for item_id, body in [("r1", content), ("old/r2", {**content, "agent_record_id": "r2", "payload": []})]:
            bad = {"items": [{"id": item_id, "content": body, "metadata": metadata}]}
            response = await client.post("/reef/namespaces/acme/data", headers=AUTH, json=bad)
            assert response.status == 400, item_id

    served(dispatcher, check)
