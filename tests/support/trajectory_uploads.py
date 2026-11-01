"""A recipe that trains uploaded trajectories, and a dispatcher and request bodies for its upload tests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aiohttp import ClientResponse
from aiohttp.test_utils import TestClient

from tests.support.runtime_stubs import RecordingComponent

from reef.artifact import InMemoryRepositoryBackend
from reef.artifact.repository import RepositoryBackendFactory
from reef.artifact.sources import ArtifactSources
from reef.dispatcher import Dispatcher
from reef.recipe import Recipe
from reef.record2sample import NoSamples
from reef.sample2job import Sample2Job
from reef.service.data_service import DataWorker, data_jobs
from reef.service.data_transfer import ArchiveWriters
from reef.service.task_import import DataScenarioStorage
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sql_data import SQLiteDataStore
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.storage.tasks import LocalTaskFileStore
from reef.train import JobData, Trainer
from reef.train.jobs import JobLoop
from reef.train.pipeline import TrainingPipeline

TOKEN = "upload-test-token"
HEADERS = {"x-reef-scenario": "s", "Authorization": f"Bearer {TOKEN}"}


def atif(text: str) -> dict[str, Any]:
    return {
        "schema_version": "ATIF-v1.4",
        "session_id": text,
        "agent": {"name": "fixture", "version": "1"},
        "steps": [{"step_id": 1, "source": "user", "message": text}],
    }


#: The data route of the owner of scenario ``s``, which is ``s`` itself.
DATA = "/reef/namespaces/s/data"


def trajectory_lines(*item_ids: str) -> bytes:
    """A JSON Lines upload body: the ATIF document of each ID, whose ``session_id`` is the ID."""
    return "".join(json.dumps(atif(item_id)) + "\n" for item_id in item_ids).encode()


def trajectory_metadata(scenario: str | None = "s") -> dict[str, Any]:
    """The metadata of a trajectory upload, used by ``scenario`` when it is given."""
    return {"kind": "trajectory", "scenarios": {} if scenario is None else {scenario: "train"}}


async def upload_trajectories(
    client: TestClient,
    headers: Mapping[str, str],
    *item_ids: str,
    scenario: str | None = "s",
    base: str = DATA,
    lines: bytes | None = None,
) -> ClientResponse:
    """Upload a trajectory item for each ID (or the JSON Lines ``lines``) to the data route ``base``."""
    return await client.post(
        f"{base}/trajectories",
        headers={**headers, "Content-Type": "application/x-ndjson"},
        params={"metadata": json.dumps(trajectory_metadata(scenario))},
        data=trajectory_lines(*item_ids) if lines is None else lines,
    )


class UploadJobData(Sample2Job):
    accepts_trajectories = True

    def make_job_data(self, samples, data_id):
        return JobData(data_id, samples)


class RecordOnlyJobData(UploadJobData):
    accepts_trajectories = False


class RenderingSamples(NoSamples):
    """Renders an uploaded trajectory by marking it, and refuses one whose session is ``bad``."""

    def uploaded_sample(self, trajectory):
        if trajectory.trajectory["session_id"] == "bad":
            raise ValueError("it has no agent step")
        return trajectory.with_training(tokens=[1, 2], loss_mask=[1], rollout_log_probs=[])


@dataclass(frozen=True, kw_only=True)
class UploadRecipe(Recipe):
    batch_stage: type[Sample2Job] = UploadJobData
    task_stage: type[NoSamples] = NoSamples
    batches: list[JobData] = field(default_factory=list)

    def build(
        self,
        scenario,
        records,
        *,
        algorithm_state=None,
        experiment_logger=None,
        task_storage=None,
        releases=None,
        samples=None,
    ):
        return Trainer.build(
            scenario,
            records,
            # An uploaded trajectory is one sample, so two uploads make one job's data.
            pipeline_factory=lambda context: TrainingPipeline(
                context, record2sample=self.task_stage(), sample2job=self.batch_stage(context, None, 2)
            ),
            job_loop=JobLoop(RecordingComponent(self.batches), scenario=scenario),
            algorithm_state=algorithm_state,
            experiment_logger=experiment_logger,
            training_mode=self.training_mode,
            task_storage=task_storage,
            samples=samples,
        )


def dispatcher_for(tmp_path: Path, recipe: Recipe, factory: RepositoryBackendFactory | None = None) -> Dispatcher:
    """A dispatcher with a data store, as deployment assembly builds it; the scenario storage owns the task files.

    ``factory`` defaults to an in-memory repository under ``tmp_path``; a test that restarts passes its own.
    """
    initial = tmp_path / "initial"
    initial.mkdir(parents=True, exist_ok=True)
    task_files = LocalTaskFileStore(tmp_path / "task-files")
    data = SQLiteDataStore(tmp_path / "data", task_files)
    storage = SQLiteScenarioStorage(tmp_path / "records", task_files=task_files)
    dispatcher = Dispatcher(
        recipe,
        factory or InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        scenario_storage=DataScenarioStorage(storage, data),
        record_storage=storage,
        data_store=data,
        coordination=memory_coordination(),
    )
    # The data service copies the records to record items, as assembly runs it.
    dispatcher.workers.add(
        DataWorker(
            dispatcher.queue,
            data_jobs(dispatcher, ArtifactSources.from_setting(()), ArchiveWriters()),
            worker_id="data-test",
        )
    )
    return dispatcher
