"""Uploaded ATIF trajectories train without records."""

from __future__ import annotations

import asyncio
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
from reef.train.types import TrajectoryItem

TOKEN = "upload-test-token"
HEADERS = {"x-reef-scenario": "s", "Authorization": f"Bearer {TOKEN}"}


def atif(text: str) -> dict[str, Any]:
    return {
        "schema_version": "ATIF-v1.4",
        "session_id": text,
        "agent": {"name": "fixture", "version": "1"},
        "steps": [{"step_id": 1, "source": "user", "message": text}],
    }


def uploaded(*trajectory_ids: str) -> dict[str, Any]:
    return {
        "trajectories": [
            {"trajectory_id": trajectory_id, "trajectory": atif(trajectory_id)} for trajectory_id in trajectory_ids
        ]
    }


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


class UploadBatches(Task2Batch):
    accepts_trajectories = True

    def make_batch(self, tasks, batch_id):
        return TrainingBatch(batch_id, tasks)


class RecordOnlyBatches(UploadBatches):
    accepts_trajectories = False


@dataclass(frozen=True, kw_only=True)
class UploadRecipe(Recipe):
    batch_stage: type[Task2Batch] = UploadBatches
    batches: list[TrainingBatch] = field(default_factory=list)

    def build(self, scenario, records, *, algorithm_state=None, experiment_logger=None, task_storage=None):
        return Trainer.build(
            scenario,
            records,
            pipeline_factory=lambda context: TrainingPipeline(context, task2batch=self.batch_stage(context, None)),
            candidate_backend=CountingBackend(self.batches),
            algorithm_state=algorithm_state,
            experiment_logger=experiment_logger,
            training_mode=self.training_mode,
            task_storage=task_storage,
            # An uploaded trajectory is one sample, so two uploads make one batch.
            batch_size=2,
        )


def dispatcher_for(tmp_path: Path, recipe: Recipe) -> Dispatcher:
    initial = tmp_path / "initial"
    initial.mkdir(exist_ok=True)
    return Dispatcher(
        recipe,
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
    )


async def wait_for_batches(recipe: UploadRecipe, count: int) -> None:
    async with asyncio.timeout(5):
        while len(recipe.batches) < count:
            await asyncio.sleep(0.01)
    assert len(recipe.batches) == count


def test_uploaded_trajectories_train_once_also_after_a_restart(tmp_path):
    recipe = UploadRecipe()

    async def upload_and_train(dispatcher: Dispatcher) -> None:
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            response = await client.post("/reef/trajectories", headers=HEADERS, json=uploaded("t1", "t2"))
            assert response.status == 200, await response.text()
            assert await response.json() == {"scenario": "s", "trajectory_ids": ["t1", "t2"]}
            await wait_for_batches(recipe, 1)

            # A retry after training changes nothing. The same ID with another trajectory conflicts.
            response = await client.post("/reef/trajectories", headers=HEADERS, json=uploaded("t1"))
            assert response.status == 200
            conflict = {"trajectories": [{"trajectory_id": "t1", "trajectory": atif("other")}]}
            response = await client.post("/reef/trajectories", headers=HEADERS, json=conflict)
            assert response.status == 409

    async def upload_after_restart(dispatcher: Dispatcher) -> None:
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            response = await client.post("/reef/trajectories", headers=HEADERS, json=uploaded("t3", "t4"))
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
    assert first.items == (TrajectoryItem(atif("t1")), TrajectoryItem(atif("t2")))
    assert second.items == (TrajectoryItem(atif("t3")), TrajectoryItem(atif("t4")))


def test_an_upload_needs_a_recipe_that_accepts_uploaded_trajectories(tmp_path):
    dispatcher = dispatcher_for(tmp_path, UploadRecipe(batch_stage=RecordOnlyBatches))

    async def run():
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            response = await client.post("/reef/trajectories", headers=HEADERS, json=uploaded("trajectory-1"))
            assert response.status == 400
            assert "does not accept uploaded trajectories" in await response.text()
            assert dispatcher.get_or_create_scenario("s").store.trajectories.count() == 0

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_an_invalid_upload_is_refused_and_saves_nothing(tmp_path):
    dispatcher = dispatcher_for(tmp_path, UploadRecipe())

    async def run():
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            response = await client.post("/reef/trajectories", json=uploaded("trajectory-1"))
            assert response.status == 401
            for body in (
                {},
                {"trajectories": []},
                {"trajectories": [{"trajectory_id": "t", "trajectory": atif("t"), "extra": 1}]},
                {"trajectories": [{"trajectory_id": " ", "trajectory": atif("t")}]},
                {"trajectories": [{"trajectory_id": "t", "trajectory": {**atif("t"), "steps": []}}]},
                {"trajectories": [{"trajectory_id": "t", "trajectory": {**atif("t"), "schema_version": "v1"}}]},
                {**uploaded("t"), "scenario": "other"},
                uploaded(*(f"t{index}" for index in range(1001))),
            ):
                response = await client.post("/reef/trajectories", headers=HEADERS, json=body)
                assert response.status == 400, await response.text()
            response = await client.post(
                "/reef/trajectories", headers={"Authorization": f"Bearer {TOKEN}"}, json=uploaded("t")
            )
            assert response.status == 400
            assert dispatcher.get_or_create_scenario("s").store.trajectories.count() == 0

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


@pytest.mark.parametrize("route", ("/reef/records", "/reef/records/batch"))
def test_the_record_import_does_not_take_a_trajectory(tmp_path, route):
    dispatcher = dispatcher_for(tmp_path, UploadRecipe())
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
