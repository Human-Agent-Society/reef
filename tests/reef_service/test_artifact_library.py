"""The workspace artifact library: artifact items in the data store, and imports of them into other scenarios."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.support._policy_recipe import TestPolicyRecipe
from tests.support.artifact_versions import ImportingComponent
from tests.support.runtime_stubs import runtime_bindings
from tests.support.trajectory_uploads import UploadRecipe, dispatcher_for

from reef.artifact import InMemoryRepositoryBackend
from reef.artifact.artifact import SnapshotRef
from reef.artifact.files import directory_checksum
from reef.artifact.sources import ArtifactSources
from reef.dispatcher import Dispatcher
from reef.service.app import create_app
from reef.service.artifact_imports import ArtifactImports
from reef.service.data_service import DataWorker, copy_scenario_artifacts, data_jobs
from reef.service.data_transfer import ArchiveWriters
from reef.service.task_import import DataScenarioStorage
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sql_data import SQLiteDataStore
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.storage.tasks import LocalTaskFileStore

DATA = "/reef/namespaces/acme/data"


class UriBackend(InMemoryRepositoryBackend):
    """An in-memory repository that names each snapshot's directory with a ``file`` URI, as an object store names it
    with an ``s3`` URI."""

    def snapshot_uri(self, ref: SnapshotRef) -> str:
        return self.materialize(ref).local_path.as_uri() + "/"


def weights(directory: Path, content: bytes) -> Path:
    """A full-weight artifact: the layout of the test policy's releases."""
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}")
    (directory / "model.safetensors").write_bytes(content)
    return directory


def cell(tmp_path: Path, name: str) -> Dispatcher:
    """One deployment, a Cell, whose data service copies records and artifacts, as assembly builds it.

    Every Cell of ``tmp_path`` shares the data store under ``tmp_path / "data"``; each has its own repository and
    records.
    """
    root = tmp_path / name
    runtime = ImportingComponent(root / "exported")
    task_files = LocalTaskFileStore(tmp_path / "task-files")
    data = SQLiteDataStore(tmp_path / "data", task_files)
    storage = SQLiteScenarioStorage(root / "records", task_files=task_files, data=data)
    dispatcher = Dispatcher(
        TestPolicyRecipe(**runtime_bindings(runtime, runtime.inference)),
        UriBackend.factory(directory=root / "repository"),
        local_artifact_dir=root / "staged",
        scenario_storage=DataScenarioStorage(storage, data),
        record_storage=storage,
        data_store=data,
        coordination=memory_coordination(),
    )
    dispatcher.workers.add(
        DataWorker(
            dispatcher.queue,
            data_jobs(dispatcher, ArtifactSources.from_setting(()), ArchiveWriters()),
            worker_id="data-test",
        )
    )
    return dispatcher


def client_of(dispatcher: Dispatcher, tmp_path: Path) -> TestClient:
    imports = ArtifactImports(ArtifactSources.from_setting(()), tmp_path / "imports")
    return TestClient(TestServer(create_app(dispatcher, artifact_imports=imports)))


async def library_item(client: TestClient, item_id: str) -> dict[str, Any]:
    """The artifact item ``item_id`` once the data service copied it."""
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        listing = await (await client.get(f"{DATA}?kind=artifact")).json()
        found = [item for item in listing["items"] if item["id"] == item_id]
        if found:
            return await (await client.get(f"{DATA}/{item_id}")).json()
        await asyncio.sleep(0.05)
    raise AssertionError(f"the data service did not copy {item_id}")


@pytest.mark.integration
def test_a_scenarios_artifacts_enter_the_library_and_a_scenario_in_another_cell_imports_one(tmp_path: Path) -> None:
    first, second = cell(tmp_path, "cell-a"), cell(tmp_path, "cell-b")

    async def run() -> None:
        async with client_of(first, tmp_path) as client, client_of(second, tmp_path) as other_cell:
            for name, cell_client in (("math", client), ("other", other_cell)):
                created = await cell_client.post("/reef/scenarios", json={"name": name, "owner": "acme"})
                assert created.status == 201, await created.text()
            source = weights(tmp_path / "source", b"w9")
            imported = await (
                await client.post("/reef/scenarios/math/artifacts", json={"source": source.as_uri()})
            ).json()
            item_id = f"math/{imported['artifact_id']}"

            # The other Cell reads the item from the shared data store.
            item = await library_item(other_cell, item_id)
            assert item["metadata"]["origin"] == {
                "scenario": "math",
                "artifact_id": imported["artifact_id"],
            }
            assert item["metadata"]["scenarios"] == {}
            content = item["content"]
            assert (content["component"], content["kind"]) == (imported["component"], "model")
            assert "content_id" not in content
            assert content["source_uri"].startswith("file://") and content["source_uri"].endswith("/")
            # The scenario lists the artifact, so the library keeps its item.
            refused = await client.delete(f"{DATA}/{item_id}")
            assert refused.status == 400 and "still exists" in await refused.text()

            # A scenario of the owner in the other Cell imports the item: the same artifact, under the same ID, and
            # the library keeps one item of it (RFC #1218).
            into = await other_cell.post("/reef/scenarios/other/artifacts", json={"item": item_id})
            assert into.status == 201, await into.text()
            receipt = await into.json()
            assert receipt["artifact_id"] == imported["artifact_id"]
            listed = (await (await other_cell.get("/reef/scenarios/other/artifacts")).json())["artifacts"]
            assert listed[0]["source"] == {"uri": content["source_uri"], "version": "", "item": item_id}
            again = await other_cell.post("/reef/scenarios/other/artifacts", json={"item": item_id})
            assert (await again.json())["artifact_id"] == imported["artifact_id"]
            assert len((await (await other_cell.get("/reef/scenarios/other/artifacts")).json())["artifacts"]) == len(
                listed
            )
            # The data service copies the scenario's first artifact in the background: wait until it did.
            first_artifact = next(row for row in listed if not row["source"].get("item"))
            await library_item(other_cell, f"other/{first_artifact['artifact_id']}")
            assert copy_scenario_artifacts(second, "other") == 0
            missing = await client.get(f"{DATA}/other/{receipt['artifact_id']}")
            assert missing.status == 404

            # A client adds files that it stored as an item of its own, imports it, and deletes it.
            body = upload_item("upload-1", weights(tmp_path / "upload", b"w10"))
            assert (await client.post(DATA, json=body)).status == 201
            first_use = await (
                await other_cell.post("/reef/scenarios/other/artifacts", json={"item": "upload-1"})
            ).json()
            # An upload gets one ID at its first use, and it keeps it.
            stored = await (await client.get(f"{DATA}/upload-1")).json()
            assert stored["metadata"]["artifact_id"] == first_use["artifact_id"]
            patch = {"ids": ["upload-1"], "patch": {"artifact_id": "artifact:other"}}
            assert (await client.post(f"{DATA}/metadata", json=patch)).status == 400
            assert (await client.delete(f"{DATA}/upload-1")).status == 200

            # A scenario uses an artifact item by importing it, not through metadata.scenarios.
            body["items"][0]["id"] = "upload-2"
            body["items"][0]["metadata"]["scenarios"] = {"other": "train"}
            refused = await client.post(DATA, json=body)
            assert refused.status == 400 and "importing it" in await refused.text()
            unknown = await other_cell.post("/reef/scenarios/other/artifacts", json={"item": "missing"})
            assert unknown.status == 404

    try:
        asyncio.run(run())
    finally:
        first.close()
        second.close()


def upload_item(item_id: str, directory: Path) -> dict[str, Any]:
    """The body that stores ``directory`` as the client's artifact item ``item_id``."""
    content = {"source_uri": directory.as_uri(), "kind": "model", "component": None, "content_id": None}
    return {"items": [{"id": item_id, "content": content, "metadata": {"kind": "artifact"}}]}


@pytest.mark.integration
def test_a_new_scenario_starts_from_an_artifact_item_of_its_owner(tmp_path: Path) -> None:
    """RFC #1062, RFC #1070: the item is the scenario's root snapshot, so it is the active artifact and the training
    start, and the creation row names the item as its source.

    A process trains one scenario, so each new scenario has a Cell of its own; the Cells share the data store.
    """
    cells = {name: cell(tmp_path, name) for name in ("math", "from-scenario", "from-upload")}

    async def run() -> None:
        async with client_of(cells["math"], tmp_path) as client:
            assert (await client.post("/reef/scenarios", json={"name": "math", "owner": "acme"})).status == 201
            source = weights(tmp_path / "source", b"w9")
            imported = await (
                await client.post("/reef/scenarios/math/artifacts", json={"source": source.as_uri()})
            ).json()
            item_id = f"math/{imported['artifact_id']}"
            await library_item(client, item_id)
            upload = weights(tmp_path / "upload", b"w10")
            assert (await client.post(DATA, json=upload_item("upload-1", upload))).status == 201
        for name, item, expected_checksum in (
            ("from-scenario", item_id, directory_checksum(source)),
            ("from-upload", "upload-1", directory_checksum(upload)),
        ):
            async with client_of(cells[name], tmp_path) as client:
                created = await client.post(
                    "/reef/scenarios", json={"name": name, "owner": "acme", "artifact_id": item}
                )
                assert created.status == 201, await created.text()
                answer = await created.json()
                started = answer["artifact_id"]
                if name == "from-scenario":
                    assert started == imported["artifact_id"]
                assert answer == {
                    "scenario": name,
                    "owner": "acme",
                    "components": {imported["component"]: started},
                    "artifact_id": started,
                }
                scenario = cells[name].get_or_create_scenario(name)
                assert scenario is not None
                assert (
                    directory_checksum(
                        scenario.repository.resolve(scenario.repository.require_current_artifact()).local_path
                    )
                    == expected_checksum
                )
                assert scenario.artifact_status() == {
                    imported["component"]: {"active": started, "training_start": started}
                }
                listed = (await (await client.get(f"/reef/scenarios/{name}/artifacts")).json())["artifacts"]
                (root,) = [row for row in listed if row["artifact_id"] == started]
                assert root["operation"] == "creation" and root["parent_artifact_id"] is None
                assert (root["source"]["kind"], root["source"]["item"]) == ("item", item)
                # The root is the only release: no default start was written first.
                assert len(scenario.snapshot_queries.snapshot_rows(scenario.scenario_step)) == 1

                # An existing scenario keeps its start: creation is idempotent, and artifact_id is not read.
                again = await client.post(
                    "/reef/scenarios", json={"name": name, "owner": "acme", "artifact_id": "missing"}
                )
                assert again.status == 200, await again.text()
                assert await again.json() == answer

    try:
        asyncio.run(run())
    finally:
        for dispatcher in cells.values():
            dispatcher.close()


@pytest.mark.integration
def test_an_item_with_the_old_serving_profile_key_starts_a_scenario_and_can_drop_the_key(tmp_path: Path) -> None:
    """RFC #1263: Reef ignores ``serving_profile``, an item key of RFC #1233, and a patch can remove it."""
    dispatcher = cell(tmp_path, "cell-a")
    body = upload_item("qwen-base", weights(tmp_path / "qwen-base", b"q"))
    body["items"][0]["metadata"]["serving_profile"] = "qwen@1"

    async def run() -> None:
        async with client_of(dispatcher, tmp_path) as client:
            assert (await client.post(DATA, json=body)).status == 201
            created = await client.post(
                "/reef/scenarios", json={"name": "math", "owner": "acme", "artifact_id": "qwen-base"}
            )
            assert created.status == 201, await created.text()
            patched = await client.post(
                f"{DATA}/metadata", json={"ids": ["qwen-base"], "patch": {"serving_profile": None}}
            )
            assert patched.status == 200, await patched.text()

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


@pytest.mark.integration
def test_a_refused_start_leaves_no_scenario(tmp_path: Path) -> None:
    """A start that the import refuses answers as the import route does, and registers no scenario."""
    dispatcher = cell(tmp_path, "cell-a")

    async def run() -> None:
        async with client_of(dispatcher, tmp_path) as client:
            notes = tmp_path / "notes"
            notes.mkdir()
            (notes / "notes.txt").write_text("no weights")
            assert (await client.post(DATA, json=upload_item("notes", notes))).status == 201
            # The component's layout refuses a directory without weights, and the owner stores no item "missing".
            for item, status in (("notes", 400), ("missing", 404)):
                refused = await client.post(
                    "/reef/scenarios", json={"name": "refused", "owner": "acme", "artifact_id": item}
                )
                assert refused.status == status, await refused.text()
                assert not dispatcher.has_scenario("refused")
                assert (await (await client.get("/reef/scenarios")).json())["scenarios"] == []
            # The name is free again.
            assert (await client.post("/reef/scenarios", json={"name": "refused", "owner": "acme"})).status == 201

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


@pytest.mark.integration
def test_a_new_owner_gets_the_artifact_items_that_a_copy_made_for_the_default_owner(tmp_path: Path) -> None:
    """The creation commit's copy can run before the scenario takes its owner; the owner's store still gets them."""
    dispatcher = dispatcher_for(tmp_path, UploadRecipe())
    try:
        scenario = dispatcher.get_or_create_scenario("s")
        item_id = f"s/{scenario.artifacts()[0]['artifact_id']}"
        store = dispatcher.data_store
        assert store is not None
        wait_until(lambda: store.read("s", item_id) is not None)
        dispatcher.set_scenario_owner("s", "acme")
        wait_until(lambda: store.read("acme", item_id) is not None)
    finally:
        dispatcher.close()


def wait_until(condition, seconds: float = 10.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() >= deadline:
            raise AssertionError("the data service did not copy the item")
        time.sleep(0.05)


def test_an_artifact_item_without_a_source_uri_is_listed_but_not_imported(tmp_path: Path) -> None:
    # The in-memory repository names no URI, so its artifacts are in the library without a source_uri.
    dispatcher = dispatcher_for(tmp_path, UploadRecipe())

    async def run() -> None:
        async with client_of(dispatcher, tmp_path) as client:
            assert (await client.post("/reef/scenarios", json={"name": "s", "owner": "acme"})).status == 201
            artifacts = (await (await client.get("/reef/scenarios/s/artifacts")).json())["artifacts"]
            item = await library_item(client, f"s/{artifacts[0]['artifact_id']}")
            assert (item["content"]["source_uri"], item["content"]["kind"]) == (None, None)
            refused = await client.post("/reef/scenarios/s/artifacts", json={"item": item["id"]})
            assert refused.status == 400 and "no source_uri" in await refused.text()

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()
