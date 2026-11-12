"""Task items (RFC #1028) join the imported tasks of each scenario that their metadata names, in its split."""

from __future__ import annotations

import asyncio
import io
import json
import tarfile
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlencode

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.support.harness_jobs import configured_recipe, evolution_config, runtime
from tests.support.task_service_stubs import task_for
from tests.support.trajectory_uploads import dispatcher_for

from reef.core.plugins import PluginReference
from reef.core.tasks import write_harbor_task
from reef.recipe import Recipe
from reef.recipe.harness import HarnessEvolutionRecipe
from reef.record2sample import NoSamples
from reef.sample2job import Sample2Job
from reef.service.app import create_app
from reef.train import JobData
from reef.train.pipeline import PipelineSettings

TOKEN = "task-upload-test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
GZIP = {**AUTH, "Content-Type": "application/gzip"}


class TaskReadingJobData(Sample2Job):
    accepts_imported_tasks = True

    @classmethod
    def from_config(cls, config, *, context, record2sample, environ):
        return cls(context, None, None)

    def make_job_data(self, samples, data_id):
        return JobData(data_id, samples)


class RecordOnlyJobData(TaskReadingJobData):
    accepts_imported_tasks = False


def task_recipe(sample2job: type[Sample2Job] = TaskReadingJobData) -> Recipe:
    """A recipe that trains nothing, whose sample-to-job stage is ``sample2job``."""
    return Recipe(
        pipeline_stages=PipelineSettings(
            record2sample=PluginReference(NoSamples, {}), sample2job=PluginReference(sample2job, {})
        )
    )


def tasks_url(split: str | None = "train", scenario: str = "s", **metadata: object) -> str:
    """The task create route of owner ``s``, with the metadata that uses the tasks in ``split`` of ``scenario``."""
    if split is not None:
        metadata = {"scenarios": {scenario: split}, **metadata}
    return f"/reef/namespaces/s/data/tasks?{urlencode({'metadata': json.dumps(metadata)})}"


async def create_scenario(client: TestClient) -> None:
    response = await client.post("/reef/scenarios", headers=AUTH, json={"name": "s"})
    assert response.status in (200, 201), await response.text()


def archive(root: Path, *names: str, port: int = 9000) -> bytes:
    """A tar.gz of one Harbor task directory for each name."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for offset, name in enumerate(names):
            tar.add(write_harbor_task(task_for(name, "record-1", port + offset), root / str(port)), arcname=name)
    return buffer.getvalue()


def raw_archive(*entries: tarfile.TarInfo) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for entry in entries:
            tar.addfile(entry, io.BytesIO(b"x" * entry.size) if entry.isfile() else None)
    return buffer.getvalue()


def entry(name: str, kind: bytes = tarfile.REGTYPE, size: int = 1, link: str = "") -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type, info.size, info.linkname = kind, size if kind == tarfile.REGTYPE else 0, link
    return info


def test_task_items_join_the_split_and_a_retry_changes_nothing(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, task_recipe())

    async def run() -> None:
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            await create_scenario(client)
            body = archive(tmp_path, "task-b", "task-a")
            response = await client.post(tasks_url("train"), headers=GZIP, data=body)
            assert response.status == 201, await response.text()
            receipt = await response.json()
            assert [item["id"] for item in receipt["items"]] == ["task-a", "task-b"]
            assert receipt["items"][0]["metadata"] == {"kind": "task", "owner": "s", "scenarios": {"s": "train"}}
            again = await client.post(tasks_url("train"), headers=GZIP, data=body)
            assert await again.json() == receipt

            held_out = archive(tmp_path, "task-c", port=9100)
            assert (await client.post(tasks_url("eval"), headers=GZIP, data=held_out)).status == 201
            # The same name with other files conflicts. A split moves with a metadata update.
            other = archive(tmp_path, "task-a", port=9200)
            assert (await client.post(tasks_url("train"), headers=GZIP, data=other)).status == 409
            moved = await client.post(
                "/reef/namespaces/s/data/metadata",
                headers=AUTH,
                json={"ids": ["task-b"], "patch": {"scenarios": {"s": "eval"}}},
            )
            assert moved.status == 200, await moved.text()

            listing = await (await client.get("/reef/scenarios/s/tasks", headers=AUTH)).json()
            assert [(task["name"], task["split"], task["trained"]) for task in listing["tasks"]] == [
                ("task-a", "train", False),
                ("task-c", "eval", False),
                ("task-b", "eval", False),
            ]

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_a_task_items_files_are_replaced_in_place_and_a_bad_replace_changes_nothing(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, task_recipe())

    async def run() -> None:
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            await create_scenario(client)
            assert (await client.post(tasks_url("eval"), headers=GZIP, data=archive(tmp_path, "task-a"))).status == 201
            created = dispatcher.data_store.read("s", "task-a")
            url = "/reef/namespaces/s/data/tasks/task-a"

            response = await client.put(url, headers=GZIP, data=archive(tmp_path, "task-a", port=9200))
            assert response.status == 200, await response.text()
            receipt = await response.json()
            replaced = dispatcher.data_store.read("s", "task-a")
            assert receipt["item"]["id"] == "task-a" and receipt["item"]["digest"] == replaced.digest
            assert replaced.digest != created.digest and replaced.metadata == created.metadata
            assert dispatcher.data_store.task_path(replaced) != dispatcher.data_store.task_path(created)

            # Another name, several tasks, an unknown item or another content type change nothing.
            for target, body, status in (
                (url, archive(tmp_path, "task-b"), 400),
                (url, archive(tmp_path, "task-a", "task-b", port=9300), 400),
                ("/reef/namespaces/s/data/tasks/missing", archive(tmp_path, "missing"), 404),
            ):
                assert (await client.put(target, headers=GZIP, data=body)).status == status
            assert (await client.put(url, headers=AUTH, data=b"{}")).status == 415
            assert dispatcher.data_store.read("s", "task-a").digest == replaced.digest

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_a_train_use_needs_a_recipe_that_reads_imported_tasks_and_an_eval_use_does_not(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, task_recipe(RecordOnlyJobData))

    async def run() -> None:
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            await create_scenario(client)
            response = await client.post(tasks_url("train"), headers=GZIP, data=archive(tmp_path, "t"))
            assert response.status == 400
            assert "does not train on imported tasks" in await response.text()
            assert dispatcher.data_store.query("s", kind="task") == ()
            # Artifact evaluations run the eval split, so it takes tasks whatever the recipe trains on.
            evaluated = await client.post(tasks_url("eval"), headers=GZIP, data=archive(tmp_path, "t"))
            assert evaluated.status == 201
            samples = dispatcher.storage.sample_store("s", dispatcher.get_or_create_scenario("s").store)
            assert [task.sample_id for task in samples.imported_tasks(split="eval")] == ["t"]

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


OFFLINE_STAGE = {"pipeline": {"sample2job": {"module": "reef.sample2job.traces:TrainTaskJobData"}}}


@pytest.mark.parametrize(
    ("pipeline", "published"),
    [(OFFLINE_STAGE, ["task-a"]), ({}, [])],
    ids=["offline-runs", "traces-only"],
)
def test_a_harness_scenario_takes_a_train_upload_when_its_stage_trains_imported_tasks(
    tmp_path: Path, pipeline: dict, published: list[str]
) -> None:
    """Only the offline stage trains imported tasks, so only it takes a train upload, even when the evaluation reads
    the imported eval split. An eval upload is open to every recipe, because an evaluation runs it."""
    evaluation = {"tasks": [], "imported_tasks": True}
    config = {**evolution_config(evaluation=evaluation, state_dir=str(tmp_path / "state")), **pipeline}
    # In manual mode no train task starts a step while the test runs.
    recipe = configured_recipe(HarnessEvolutionRecipe.from_environment({}, config=config, runtime=runtime()))
    dispatcher = dispatcher_for(tmp_path, replace(recipe, training_mode="manual"))

    async def run() -> None:
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            await create_scenario(client)
            response = await client.post(tasks_url("train"), headers=GZIP, data=archive(tmp_path, "task-a"))
            assert response.status == (201 if published else 400), await response.text()
            held_out = archive(tmp_path, "held-out", port=9100)
            assert (await client.post(tasks_url("eval"), headers=GZIP, data=held_out)).status == 201
            samples = dispatcher.storage.sample_store("s", dispatcher.get_or_create_scenario("s").store)
            assert [task.sample_id for task in samples.imported_tasks(split="train")] == published
            assert [task.sample_id for task in samples.imported_tasks(split="eval")] == ["held-out"]

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_an_invalid_task_create_is_refused_and_stores_nothing(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, task_recipe())
    good = archive(tmp_path, "task-a")

    async def run() -> None:
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            unauthorized = await client.post(tasks_url(), data=good)
            assert unauthorized.status == 401
            wrong_type = await client.post(tasks_url(), headers=AUTH, data=good)
            assert wrong_type.status == 415
            for url in (
                "/reef/namespaces/s/data/tasks?metadata=not-json",
                tasks_url(None, kind="trajectory"),
                tasks_url("test"),
            ):
                response = await client.post(url, headers=GZIP, data=good)
                assert response.status == 400, url
            for body, reason in [
                (b"not an archive", "not a gzip-compressed tar archive"),
                (raw_archive(entry("../escape")), "not a relative path"),
                (raw_archive(entry("/abs/file")), "not a relative path"),
                (raw_archive(entry("README.md")), "outside a task directory"),
                (raw_archive(entry("Task_A", tarfile.DIRTYPE)), "is not a task name"),
                (
                    raw_archive(entry("task", tarfile.DIRTYPE), entry("task/link", tarfile.SYMTYPE, link="/etc")),
                    "not a file",
                ),
                (raw_archive(entry("task", tarfile.DIRTYPE), entry("task/notes.txt")), "is not a Harbor task"),
            ]:
                response = await client.post(tasks_url(), headers=GZIP, data=body)
                assert response.status == 400, reason
                assert reason in await response.text()
            assert dispatcher.data_store.query("s", kind="task") == ()

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_a_body_over_the_unpacked_limit_is_refused_before_it_unpacks(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("reef.service.routes.data.MAX_UNPACKED_BYTES", 64)
    dispatcher = dispatcher_for(tmp_path, task_recipe())

    async def run() -> None:
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            response = await client.post(tasks_url(), headers=GZIP, data=archive(tmp_path, "task-a"))
            assert response.status == 413

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()
