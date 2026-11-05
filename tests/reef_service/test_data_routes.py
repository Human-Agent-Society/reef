"""The data routes (RFC #1028): each scenario uses only the items of its owner."""

from __future__ import annotations

import asyncio
import io
import json
import tarfile
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer
from tests.support.task_service_stubs import task_for
from tests.support.trajectory_uploads import TOKEN, UploadRecipe, dispatcher_for, upload_trajectories

from reef.artifact import InMemoryRepositoryBackend
from reef.core.tasks import write_harbor_task
from reef.dispatcher import Dispatcher
from reef.service.app import create_app
from reef.storage.data import NewItem
from reef.storage.sql_coordination import memory_coordination
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
        assert all("content" not in item for item in first["items"])
        # content=1 gives each item's content, as reading the item does.
        with_content = await (await client.get(f"{base}?limit=2&content=1", headers=AUTH)).json()
        t1 = await (await client.get(f"{base}/t1", headers=AUTH)).json()
        assert with_content["items"][0]["content"] == t1["content"] and t1["content"] is not None
        rest = await (await client.get(f"{base}?after={first['next']}", headers=AUTH)).json()
        assert ([item["id"] for item in rest["items"]], rest["next"]) == (["t3", "task-a"], None)
        tasks = await (await client.get(f"{base}?kind=task", headers=AUTH)).json()
        assert [item["id"] for item in tasks["items"]] == ["task-a"]
        task = await (await client.get(f"{base}/task-a", headers=AUTH)).json()
        assert (task["content"], task["metadata"]["kind"]) == (None, "task")
        for query in ("?kind=dataset", "?after=x", "?limit=0", "?content=2", "?content=yes"):
            assert (await client.get(base + query, headers=AUTH)).status == 400, query
        assert (await client.delete(f"{base}/missing", headers=AUTH)).status == 404

    served(dispatcher, check)


def test_a_deployment_without_a_data_store_serves_no_data_route(tmp_path: Path) -> None:
    (tmp_path / "initial").mkdir()
    dispatcher = Dispatcher(
        UploadRecipe(),
        InMemoryRepositoryBackend.factory(directory=tmp_path / "repository"),
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
        coordination=memory_coordination(),
    )

    async def check(client: TestClient) -> None:
        response = await client.get("/reef/namespaces/acme/data", headers=AUTH)
        assert response.status == 503
        assert "no data store" in await response.text()

    served(dispatcher, check)


class TextOnlyRecipe(UploadRecipe):
    @property
    def refuses_images(self) -> bool:
        return True


IMAGE_PART = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}


def record_item_body(record_id: str, content: object, scenarios: dict[str, str]) -> dict[str, object]:
    payload = {"messages": [{"role": "user", "content": content}], "response": {"content": "a"}}
    return {
        "items": [
            {
                "id": f"old/{record_id}",
                "content": {
                    "agent_record_id": record_id,
                    "request_type": "inference",
                    "payload": payload,
                    "created_at": 1.0,
                    "references": [],
                    "artifact_id": None,
                },
                "metadata": {
                    "kind": "record",
                    "scenarios": scenarios,
                    "origin": {"scenario": "old", "artifact_id": None},
                },
            }
        ]
    }


def test_a_recipe_that_refuses_images_refuses_each_item_that_holds_one(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, TextOnlyRecipe())
    data = "/reef/namespaces/s/data"
    image_step = {"step_id": 1, "source": "user", "message": [{"type": "image", "source": {"path": "a.png"}}]}
    trajectory = {
        "schema_version": "ATIF-v1.4",
        "session_id": "t",
        "agent": {"name": "a"},
        "steps": [image_step, {"step_id": 2, "source": "agent", "message": "a"}],
    }

    async def check(client: TestClient) -> None:
        assert (await client.post("/reef/scenarios", headers=AUTH, json={"name": "s"})).status in (200, 201)
        refused = await client.post(data, headers=AUTH, json=record_item_body("r1", [IMAGE_PART], {"s": "train"}))
        assert refused.status == 400
        assert "item 'old/r1' holds an image" in await refused.text()
        lines = (json.dumps(trajectory) + "\n").encode()
        refused = await upload_trajectories(client, AUTH, lines=lines, base=data)
        assert refused.status == 400
        assert "item 't' holds an image" in await refused.text()
        assert dispatcher.data_store.query("s") == ()
        # A scenario that does not exist yet refuses it too.
        later = await client.post(data, headers=AUTH, json=record_item_body("r4", [IMAGE_PART], {"later": "train"}))
        assert later.status == 400
        # An item that names no scenario is stored. A metadata update cannot add it to a scenario.
        stored = await client.post(data, headers=AUTH, json=record_item_body("r2", [IMAGE_PART], {}))
        assert stored.status == 201, await stored.text()
        patch = {"ids": ["old/r2"], "patch": {"scenarios": {"s": "train"}}}
        refused = await client.post(f"{data}/metadata", headers=AUTH, json=patch)
        assert refused.status == 400
        assert "item 'old/r2' holds an image" in await refused.text()
        assert (await client.post(data, headers=AUTH, json=record_item_body("r3", "q", {"s": "train"}))).status == 201

    served(dispatcher, check)


def test_each_sync_skips_a_stored_record_item_with_an_image(tmp_path: Path, caplog) -> None:
    dispatcher = dispatcher_for(tmp_path, TextOnlyRecipe())
    data = "/reef/namespaces/s/data"
    entry = record_item_body("image", [IMAGE_PART], {"s": "train"})["items"][0]
    # Reef stored this item before the recipe refused images.
    dispatcher.data_store.create("s", [NewItem(entry["id"], entry["content"], entry["metadata"])])

    async def check(client: TestClient) -> None:
        assert (await client.post("/reef/scenarios", headers=AUTH, json={"name": "s"})).status in (200, 201)
        assert "skips record item 'old/image'" in caplog.text
        caplog.clear()
        created = await client.post(data, headers=AUTH, json=record_item_body("text", "q", {"s": "train"}))
        assert created.status == 201, await created.text()
        records = dispatcher.get_or_create_scenario("s").records
        assert records.get("s", "old/text") is not None
        assert records.get("s", "old/image") is None
        assert "skips record item 'old/image'" in caplog.text

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
