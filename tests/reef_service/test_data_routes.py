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
        # The uploads give no set_id, so the trajectories are in the group without a value.
        ungrouped = await (await client.get(f"{base}?kind=trajectory&group_key=set_id", headers=AUTH)).json()
        assert [item["id"] for item in ungrouped["items"]] == ["t1", "t2", "t3"]
        named = await (await client.get(f"{base}?group_key=set_id&group_value=a", headers=AUTH)).json()
        assert named["items"] == []
        for query in (
            "?kind=dataset",
            "?after=x",
            "?limit=0",
            "?content=2",
            "?content=yes",
            "?group_value=a",
            "?group_key=set.id",
        ):
            assert (await client.get(base + query, headers=AUTH)).status == 400, query
        assert (await client.delete(f"{base}/missing", headers=AUTH)).status == 404

    served(dispatcher, check)


def test_a_collection_names_benchmarks_and_adds_their_tasks_to_a_scenario(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, UploadRecipe())

    def archive(*names: str) -> bytes:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            for port, name in enumerate(names, start=9000):
                tar.add(write_harbor_task(task_for(name, "record-1", port), tmp_path / name), arcname=name)
        return buffer.getvalue()

    async def check(client: TestClient) -> None:
        base, stored = "/reef/namespaces/acme/data", "/reef/namespaces/acme/collections"
        await client.post("/reef/scenarios", headers=AUTH, json={"name": "math", "owner": "acme"})
        for names, benchmarks in ((("add", "sort"), ["full"]), (("sum",), ["extra"])):
            response = await client.post(
                f"{base}/tasks",
                params={"metadata": json.dumps({"benchmarks": benchmarks})},
                headers={**AUTH, "Content-Type": "application/gzip"},
                data=archive(*names),
            )
            assert response.status == 201, await response.text()
        # A task can be in two benchmarks.
        patch = {"ids": ["add"], "patch": {"benchmarks": ["full", "lite"]}}
        assert (await client.post(f"{base}/metadata", headers=AUTH, json=patch)).status == 200
        listed = await (await client.get(f"{base}?benchmark=lite", headers=AUTH)).json()
        assert [item["id"] for item in listed["items"]] == ["add"]

        coding = {"benchmarks": ["lite", "full"]}
        saved = await client.put(f"{stored}/coding", headers=AUTH, json=coding)
        assert saved.status == 200, await saved.text()
        assert await saved.json() == {
            "owner": "acme",
            "name": "coding",
            "collections": [],
            "average": "task",
            **coding,
        }
        everything = {"benchmarks": ["extra"], "collections": ["coding"], "average": "benchmark"}
        assert (await client.put(f"{stored}/all", headers=AUTH, json=everything)).status == 200
        names = [entry["name"] for entry in (await (await client.get(stored, headers=AUTH)).json())["collections"]]
        assert names == ["all", "coding"]
        read = await (await client.get(f"{stored}/all", headers=AUTH)).json()
        assert read["benchmark_task_counts"] == [
            {"benchmark": "extra", "task_count": 1},
            {"benchmark": "lite", "task_count": 1},
            {"benchmark": "full", "task_count": 2},
        ]
        assert (read["task_count"], read["average"]) == (3, "benchmark")
        for name, body in (
            ("coding", {"collections": ["all"]}),
            ("open", {"collections": ["missing"]}),
            ("named", {"name": "other", "benchmarks": ["lite"]}),
        ):
            assert (await client.put(f"{stored}/{name}", headers=AUTH, json=body)).status == 400, name
        # A collection can be empty: it has no task, so it adds none to a scenario.
        assert (await client.put(f"{stored}/empty", headers=AUTH, json={})).status == 200
        empty = await (await client.get(f"{stored}/empty", headers=AUTH)).json()
        assert (empty["benchmarks"], empty["collections"], empty["task_count"]) == ([], [], 0)
        to_eval = {"scenario": "math", "split": "eval"}
        assert (await client.post(f"{stored}/empty/scenarios", headers=AUTH, json=to_eval)).status == 400
        assert (await client.delete(f"{stored}/empty", headers=AUTH)).status == 200

        added = await client.post(
            f"{stored}/coding/scenarios", headers=AUTH, json={"scenario": "math", "split": "eval"}
        )
        assert added.status == 200, await added.text()
        assert (await added.json())["task_count"] == 2
        used = await (await client.get(f"{base}?kind=task&scenario=math", headers=AUTH)).json()
        assert {item["id"]: item["metadata"]["scenarios"] for item in used["items"]} == {
            "add": {"math": "eval"},
            "sort": {"math": "eval"},
        }
        bad_split = {"scenario": "math", "split": "dev"}
        assert (await client.post(f"{stored}/coding/scenarios", headers=AUTH, json=bad_split)).status == 400
        assert (await client.post(f"{stored}/coding/scenarios", headers=AUTH, json={"scenario": "math"})).status == 400

        # A collection that another one names stays; a deleted collection changes no task.
        assert (await client.delete(f"{stored}/coding", headers=AUTH)).status == 400
        assert (await client.delete(f"{stored}/all", headers=AUTH)).status == 200
        assert (await client.delete(f"{stored}/coding", headers=AUTH)).status == 200
        for method in ("GET", "DELETE"):
            assert (await client.request(method, f"{stored}/coding", headers=AUTH)).status == 404
        missing = {"scenario": "math", "split": "eval"}
        assert (await client.post(f"{stored}/coding/scenarios", headers=AUTH, json=missing)).status == 404
        assert len((await (await client.get(f"{base}?benchmark=full", headers=AUTH)).json())["items"]) == 2

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
        # An item that names no scenario, or a scenario that does not exist yet, is refused too.
        for record_id, scenarios in [("r2", {}), ("r4", {"later": "train"})]:
            refused = await client.post(data, headers=AUTH, json=record_item_body(record_id, [IMAGE_PART], scenarios))
            assert refused.status == 400
            assert f"item 'old/{record_id}' holds an image" in await refused.text()
        assert dispatcher.data_store.query("s") == ()
        assert (await client.post(data, headers=AUTH, json=record_item_body("r3", "q", {"s": "train"}))).status == 201

    served(dispatcher, check)


def test_a_client_record_item_needs_the_content_and_id_of_a_record(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, UploadRecipe())
    content = {
        "agent_record_id": "r1",
        "request_type": "report",
        "payload": {"score": 1.0},
        "created_at": 1.0,
        "references": [],
        "artifact_id": None,
    }
    metadata = {"kind": "record", "origin": {"scenario": "old", "artifact_id": None}}

    async def check(client: TestClient) -> None:
        good = {"items": [{"id": "old/r1", "content": content, "metadata": metadata}]}
        assert (await client.post("/reef/namespaces/acme/data", headers=AUTH, json=good)).status == 201
        for item_id, body in [("r1", content), ("old/r2", {**content, "agent_record_id": "r2", "payload": []})]:
            bad = {"items": [{"id": item_id, "content": body, "metadata": metadata}]}
            response = await client.post("/reef/namespaces/acme/data", headers=AUTH, json=bad)
            assert response.status == 400, item_id

    served(dispatcher, check)
