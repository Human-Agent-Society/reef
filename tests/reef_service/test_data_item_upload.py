"""Uploaded ATIF trajectories and Harbor tasks train as data items, without records."""

from __future__ import annotations

import asyncio
import io
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from aiohttp.test_utils import TestClient, TestServer

from reef.artifact import InMemoryRepositoryBackend
from reef.core.evaluation import EvaluationResult, SelectionDecision, UpdateCandidate
from reef.dispatcher import Dispatcher
from reef.recipe import Recipe
from reef.service.app import create_app
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.task2batch import Task2Batch
from reef.train import CandidateBackend, PreparedStep, Trainer, TrainingBatch, TrainStepResult
from reef.train.pipeline import TrainingPipeline
from reef.train.types import TaskItem, TrajectoryItem

TOKEN = "upload-test-token"
HEADERS = {"x-reef-scenario": "s", "Authorization": f"Bearer {TOKEN}"}


def atif(text: str) -> dict[str, Any]:
    return {
        "schema_version": "ATIF-v1.4",
        "session_id": text,
        "agent": {"name": "fixture", "version": "1"},
        "steps": [{"step_id": 1, "source": "user", "message": text}],
    }


def uploaded(*item_ids: str) -> dict[str, Any]:
    return {"trajectories": [{"item_id": item_id, "trajectory": atif(item_id)} for item_id in item_ids]}


def write_task(root: Path, instruction: str = "Print hello.") -> Path:
    task = root / "demo"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "instruction.md").write_text(instruction)
    (task / "environment" / "Dockerfile").write_text("FROM alpine\n")
    (task / "tests" / "test.sh").write_text("exit 0\n")
    return task


def task_archive(*entries: tuple[str, Path]) -> bytes:
    """A gzip tar that holds each ``(name, path)`` pair: the file or directory ``path`` under ``name``."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, path in entries:
            tar.add(path, arcname=name)
    return buffer.getvalue()


class CountingBackend(CandidateBackend):
    """Keeps every batch and skips its step, so the trainer commits it."""

    def __init__(self, batches: list[TrainingBatch]) -> None:
        self.batches = batches

    def initial_state(self) -> Mapping[str, Any]:
        return {"trained": 0}

    def prepare_step(self, batch: TrainingBatch, state: Mapping[str, Any], scenario_step: int) -> PreparedStep:
        self.batches.append(batch)
        return PreparedStep.skipped(state={"trained": int(state["trained"]) + len(batch.items)})

    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
        raise NotImplementedError

    def settle_step(self, prepared: PreparedStep, decision: SelectionDecision) -> TrainStepResult:
        raise NotImplementedError

    def abort_step(self, prepared: PreparedStep) -> None:
        pass


class DataItemBatches(Task2Batch):
    accepted_items = frozenset({TrajectoryItem, TaskItem})

    def make_batch(self, tasks, batch_id):
        return TrainingBatch(batch_id, tasks)


class TaskOnlyBatches(DataItemBatches):
    accepted_items = frozenset({TaskItem})


@dataclass(frozen=True, kw_only=True)
class DataItemRecipe(Recipe):
    batch_stage: type[Task2Batch] = DataItemBatches
    batches: list[TrainingBatch] = field(default_factory=list)

    def build(self, scenario, records, *, algorithm_state=None, experiment_logger=None, task_storage=None):
        return Trainer.build(
            scenario,
            records,
            pipeline_factory=lambda context: TrainingPipeline(context, task2batch=self.batch_stage(context, 2)),
            candidate_backend=CountingBackend(self.batches),
            algorithm_state=algorithm_state,
            experiment_logger=experiment_logger,
            training_mode=self.training_mode,
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


async def wait_for_batches(recipe: DataItemRecipe, count: int) -> None:
    async with asyncio.timeout(5):
        while len(recipe.batches) < count:
            await asyncio.sleep(0.01)
    assert len(recipe.batches) == count


def test_uploaded_data_items_train_once_also_after_a_restart(tmp_path):
    recipe = DataItemRecipe()
    task = write_task(tmp_path / "source")
    changed = write_task(tmp_path / "changed", instruction="Print goodbye.")
    task_headers = {**HEADERS, "x-reef-item-id": "task-1"}

    async def upload_and_train(dispatcher: Dispatcher) -> None:
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            response = await client.post("/reef/trajectories", headers=HEADERS, json=uploaded("trajectory-1"))
            assert response.status == 200, await response.text()
            assert await response.json() == {
                "scenario": "s",
                "items": [{"item_id": "trajectory-1", "kind": "trajectory"}],
            }
            response = await client.post("/reef/tasks", headers=task_headers, data=task_archive(("demo", task)))
            assert response.status == 200, await response.text()
            receipt = await response.json()
            assert receipt["item_id"] == "task-1" and receipt["task"]["name"] == "demo"
            await wait_for_batches(recipe, 1)

            # A retry after training changes nothing. The same ID with other content conflicts.
            response = await client.post("/reef/trajectories", headers=HEADERS, json=uploaded("trajectory-1"))
            assert response.status == 200
            response = await client.post("/reef/tasks", headers=task_headers, data=task_archive(("demo", task)))
            assert response.status == 200
            conflict = {"trajectories": [{"item_id": "trajectory-1", "trajectory": atif("other")}]}
            response = await client.post("/reef/trajectories", headers=HEADERS, json=conflict)
            assert response.status == 409
            response = await client.post("/reef/tasks", headers=task_headers, data=task_archive(("demo", changed)))
            assert response.status == 409

    async def upload_after_restart(dispatcher: Dispatcher) -> None:
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            response = await client.post(
                "/reef/trajectories", headers=HEADERS, json=uploaded("trajectory-2", "trajectory-3")
            )
            assert response.status == 200, await response.text()
            await wait_for_batches(recipe, 2)

    dispatcher = dispatcher_for(tmp_path, recipe)
    try:
        asyncio.run(upload_and_train(dispatcher))
        assert dispatcher.get_or_create_scenario("s").records.count("s") == 0
    finally:
        dispatcher.close()
    dispatcher = dispatcher_for(tmp_path, recipe)
    try:
        asyncio.run(upload_after_restart(dispatcher))
    finally:
        dispatcher.close()

    first, second = recipe.batches
    uploaded_trajectory, uploaded_task = first.items
    assert uploaded_trajectory == TrajectoryItem(atif("trajectory-1"))
    assert isinstance(uploaded_task, TaskItem) and uploaded_task.task_path.name == "demo"
    assert (uploaded_task.task_path / "instruction.md").read_text() == "Print hello."
    assert second.items == (TrajectoryItem(atif("trajectory-2")), TrajectoryItem(atif("trajectory-3")))


def test_an_upload_needs_a_recipe_that_accepts_its_kind(tmp_path):
    dispatcher = dispatcher_for(tmp_path, DataItemRecipe(batch_stage=TaskOnlyBatches))

    async def run():
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            response = await client.post("/reef/trajectories", headers=HEADERS, json=uploaded("trajectory-1"))
            assert response.status == 400
            assert "does not accept TrajectoryItem uploads" in await response.text()
            assert dispatcher.get_or_create_scenario("s").store.data_items.count() == 0

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_an_invalid_upload_is_refused_and_saves_nothing(tmp_path):
    dispatcher = dispatcher_for(tmp_path, DataItemRecipe())
    task = write_task(tmp_path / "source")
    no_verifier = write_task(tmp_path / "no-verifier")
    (no_verifier / "tests" / "test.sh").unlink()
    (tmp_path / "link").symlink_to(task / "instruction.md")
    task_headers = {**HEADERS, "x-reef-item-id": "task-1"}

    async def run():
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            response = await client.post("/reef/trajectories", json=uploaded("trajectory-1"))
            assert response.status == 401
            for body in (
                {},
                {"trajectories": []},
                {"trajectories": [{"item_id": "t", "trajectory": atif("t"), "extra": 1}]},
                {"trajectories": [{"item_id": " ", "trajectory": atif("t")}]},
                {"trajectories": [{"item_id": "t", "trajectory": {**atif("t"), "steps": []}}]},
                {"trajectories": [{"item_id": "t", "trajectory": {**atif("t"), "schema_version": "v1"}}]},
                {**uploaded("t"), "scenario": "other"},
                uploaded(*(f"t{index}" for index in range(1001))),
            ):
                response = await client.post("/reef/trajectories", headers=HEADERS, json=body)
                assert response.status == 400, await response.text()
            response = await client.post(
                "/reef/trajectories", headers={"Authorization": f"Bearer {TOKEN}"}, json=uploaded("t")
            )
            assert response.status == 400
            response = await client.post("/reef/tasks", headers=HEADERS, data=task_archive(("demo", task)))
            assert response.status == 400
            assert "x-reef-item-id" in await response.text()
            for archive in (
                b"not a gzip tar",
                task_archive(("demo", task), ("other", task)),
                task_archive(("demo", task), ("demo/link", tmp_path / "link")),
                task_archive(("demo", task), ("demo/../escape", task / "instruction.md")),
                task_archive(("demo", task / "instruction.md")),
                task_archive(("demo", no_verifier)),
            ):
                response = await client.post("/reef/tasks", headers=task_headers, data=archive)
                assert response.status == 400, await response.text()
            assert dispatcher.get_or_create_scenario("s").store.data_items.count() == 0

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


@pytest.mark.parametrize("route", ("/reef/records", "/reef/records/batch"))
def test_the_record_import_does_not_take_a_trajectory(tmp_path, route):
    dispatcher = dispatcher_for(tmp_path, DataItemRecipe())
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
