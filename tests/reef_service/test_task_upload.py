"""Uploaded Harbor tasks join a scenario's imported tasks in the split that the request names."""

from __future__ import annotations

import asyncio
import io
import tarfile
from dataclasses import dataclass
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer
from tests.support.task_service_stubs import task_for

from reef.artifact import InMemoryRepositoryBackend
from reef.core.tasks import write_harbor_task
from reef.dispatcher import Dispatcher
from reef.recipe import Recipe
from reef.record2task import NoTasks
from reef.service.app import create_app
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.storage.tasks import IMPORTED_TASK_STORE
from reef.task2job import Task2Job
from reef.train import JobData, Trainer
from reef.train.pipeline import TrainingPipeline

TOKEN = "task-upload-test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
GZIP = {**AUTH, "Content-Type": "application/gzip"}


class TaskReadingJobData(Task2Job):
    accepts_imported_tasks = True

    def make_job_data(self, tasks, data_id):
        return JobData(data_id, tasks)


class RecordOnlyJobData(TaskReadingJobData):
    accepts_imported_tasks = False


@dataclass(frozen=True, kw_only=True)
class TaskRecipe(Recipe):
    batch_stage: type[Task2Job] = TaskReadingJobData

    def build(self, scenario, records, *, algorithm_state=None, experiment_logger=None, task_storage=None):
        return Trainer.build(
            scenario,
            records,
            pipeline_factory=lambda context: TrainingPipeline(
                context, record2task=NoTasks(), task2job=self.batch_stage(context, None, None)
            ),
            algorithm_state=algorithm_state,
            experiment_logger=experiment_logger,
            task_storage=task_storage,
        )


def dispatcher_for(tmp_path: Path, recipe: Recipe) -> Dispatcher:
    initial = tmp_path / "initial"
    initial.mkdir(exist_ok=True)
    return Dispatcher(
        recipe,
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
    )


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


def test_uploaded_tasks_join_the_split_and_a_retry_changes_nothing(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, TaskRecipe())

    async def run() -> None:
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            body = archive(tmp_path, "task-b", "task-a")
            response = await client.post("/reef/scenarios/s/tasks?split=train", headers=GZIP, data=body)
            assert response.status == 200, await response.text()
            receipt = await response.json()
            assert (receipt["scenario"], receipt["split"]) == ("s", "train")
            assert [task["name"] for task in receipt["tasks"]] == ["task-a", "task-b"]
            again = await client.post("/reef/scenarios/s/tasks?split=train", headers=GZIP, data=body)
            assert await again.json() == receipt

            held_out = archive(tmp_path, "task-c", port=9100)
            assert (await client.post("/reef/scenarios/s/tasks?split=eval", headers=GZIP, data=held_out)).status == 200
            # The same name with other files, or in the other split, conflicts.
            other = archive(tmp_path, "task-a", port=9200)
            assert (await client.post("/reef/scenarios/s/tasks?split=train", headers=GZIP, data=other)).status == 409
            moved = await client.post("/reef/scenarios/s/tasks?split=eval", headers=GZIP, data=body)
            assert moved.status == 409

            listing = await (await client.get("/reef/scenarios/s/tasks", headers=AUTH)).json()
            assert [(task["name"], task["split"], task["trained"]) for task in listing["tasks"]] == [
                ("task-a", "train", False),
                ("task-b", "train", False),
                ("task-c", "eval", False),
            ]

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_an_upload_needs_a_recipe_that_reads_imported_tasks(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, TaskRecipe(batch_stage=RecordOnlyJobData))

    async def run() -> None:
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            response = await client.post(
                "/reef/scenarios/s/tasks?split=train", headers=GZIP, data=archive(tmp_path, "t")
            )
            assert response.status == 400
            assert "does not use imported tasks" in await response.text()
            store = dispatcher.get_or_create_scenario("s").store.tasks.open(IMPORTED_TASK_STORE)
            assert store.tasks() == ()

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_an_invalid_upload_is_refused_and_publishes_nothing(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, TaskRecipe())
    good = archive(tmp_path, "task-a")

    async def run() -> None:
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            unauthorized = await client.post("/reef/scenarios/s/tasks?split=train", data=good)
            assert unauthorized.status == 401
            wrong_type = await client.post("/reef/scenarios/s/tasks?split=train", headers=AUTH, data=good)
            assert wrong_type.status == 415
            for query in ("", "?split=test"):
                response = await client.post(f"/reef/scenarios/s/tasks{query}", headers=GZIP, data=good)
                assert response.status == 400
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
                response = await client.post("/reef/scenarios/s/tasks?split=train", headers=GZIP, data=body)
                assert response.status == 400, reason
                assert reason in await response.text()
            listing = await (await client.get("/reef/scenarios/s/tasks", headers=AUTH)).json()
            assert listing == {"scenario": "s", "tasks": []}

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_a_body_over_the_unpacked_limit_is_refused_before_it_unpacks(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("reef.service.routes.tasks.MAX_UNPACKED_BYTES", 64)
    dispatcher = dispatcher_for(tmp_path, TaskRecipe())

    async def run() -> None:
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            body = archive(tmp_path, "task-a")
            response = await client.post("/reef/scenarios/s/tasks?split=train", headers=GZIP, data=body)
            assert response.status == 413

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()
