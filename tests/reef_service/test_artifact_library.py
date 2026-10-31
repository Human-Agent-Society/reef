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
from reef.artifact.artifact import ArtifactRef
from reef.artifact.files import directory_content_id
from reef.artifact.sources import ArtifactSources
from reef.dispatcher import Dispatcher
from reef.service.app import create_app
from reef.service.artifact_imports import ArtifactImports
from reef.service.data_service import DataWorker
from reef.service.data_transfer import ArchiveWriters
from reef.service.task_import import DataScenarioStorage
from reef.storage.sql_data import SQLiteDataStore
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.storage.tasks import LocalTaskFileStore

DATA = "/reef/namespaces/acme/data"


class UriBackend(InMemoryRepositoryBackend):
    """An in-memory repository that names each release's directory with a ``file`` URI, as an object store names it
    with an ``s3`` URI."""

    def release_uri(self, ref: ArtifactRef) -> str:
        return self.materialize(ref).local_path.as_uri() + "/"


def weights(directory: Path, content: bytes) -> Path:
    """A full-weight release: the layout of the test policy's releases."""
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
    initial = root / "initial"
    initial.mkdir(parents=True)
    (initial / "model.txt").write_text("w0")
    runtime = ImportingComponent(root / "exported")
    task_files = LocalTaskFileStore(tmp_path / "task-files")
    data = SQLiteDataStore(tmp_path / "data", task_files)
    storage = SQLiteScenarioStorage(root / "records", task_files=task_files)
    dispatcher = Dispatcher(
        TestPolicyRecipe(**runtime_bindings(runtime, runtime.inference)),
        UriBackend.factory(initial, root=root / "repository"),
        local_artifact_dir=root / "staged",
        scenario_storage=DataScenarioStorage(storage, data),
        record_storage=storage,
        data_store=data,
    )
    dispatcher.workers.add(
        DataWorker(dispatcher, ArtifactSources.from_setting(()), ArchiveWriters(), worker_id="data-test")
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
                "release_id": imported["release_id"],
                "artifact_id": imported["artifact_id"],
            }
            assert item["metadata"]["scenarios"] == {}
            content = item["content"]
            assert (content["component"], content["kind"]) == (imported["component"], "model")
            assert content["content_id"] == directory_content_id(source)
            assert content["source_uri"].startswith("file://") and content["source_uri"].endswith("/")
            # The scenario lists the artifact, so the library keeps its item.
            refused = await client.delete(f"{DATA}/{item_id}")
            assert refused.status == 400 and "still exists" in await refused.text()

            # A scenario of the owner in the other Cell imports the item: a copy of the same content, which names
            # the item.
            into = await other_cell.post("/reef/scenarios/other/artifacts", json={"item": item_id})
            assert into.status == 201, await into.text()
            receipt = await into.json()
            assert receipt["content_id"] == directory_content_id(source)
            listed = (await (await other_cell.get("/reef/scenarios/other/artifacts")).json())["artifacts"]
            assert listed[0]["source"] == {"uri": content["source_uri"], "version": "", "item": item_id}
            # The import is an artifact of "other", so it enters the library too.
            await library_item(client, f"other/{receipt['artifact_id']}")

            # A client adds files that it stored as an item of its own, imports it, and deletes it.
            upload = weights(tmp_path / "upload", b"w10")
            body = {
                "items": [
                    {
                        "id": "upload-1",
                        "content": {
                            "source_uri": upload.as_uri(),
                            "kind": "model",
                            "component": None,
                            "content_id": None,
                        },
                        "metadata": {"kind": "artifact"},
                    }
                ]
            }
            assert (await client.post(DATA, json=body)).status == 201
            assert (await other_cell.post("/reef/scenarios/other/artifacts", json={"item": "upload-1"})).status == 201
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


@pytest.mark.integration
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
