"""A recipe that trains uploaded trajectories, and a dispatcher and request bodies for its upload tests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Self

from aiohttp import ClientResponse
from aiohttp.test_utils import TestClient
from tests.support.runtime_stubs import (
    RecordedJob,
    RecordingComponent,
    StubInferenceRuntime,
    job_sized_stages,
    live_objects,
)

from reef.artifact import DefaultStart, InMemoryRepositoryBackend
from reef.artifact.repository import RepositoryBackendFactory
from reef.artifact.sources import ArtifactSources
from reef.dispatcher import Dispatcher
from reef.pipeline_context import PipelineContext
from reef.recipe import Recipe
from reef.recipe.base import WeightTrainingRecipe
from reef.record2sample import NoSamples, Record2Sample
from reef.sample2job import Sample2Job, Sample2JobSettings
from reef.scenario.live_objects import LiveObjects
from reef.service.data_service import DataWorker, data_jobs
from reef.service.data_transfer import ArchiveWriters
from reef.service.task_import import DataScenarioStorage
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sql_data import SQLiteDataStore
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.storage.tasks import LocalTaskFileStore
from reef.train import JobData
from reef.train.pipeline import PipelineSpec

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

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2sample: Record2Sample,
        environ: Mapping[str, str],
    ) -> Self:
        settings = Sample2JobSettings.parse(config, plugin_class=cls)
        return cls(context, settings.buffer_size, settings.job_samples)

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
class UploadRecipe(WeightTrainingRecipe):
    """A weight recipe whose stages are ``task_stage`` and ``batch_stage``, with two samples in a job.

    ``dispatcher_for`` gives its scenarios an ``UploadComponent``: the jobs go into ``batches``, and a new scenario
    starts from ``start``.
    """

    batch_stage: type[Sample2Job] = UploadJobData
    task_stage: type[NoSamples] = NoSamples
    batches: list[RecordedJob] = field(default_factory=list)
    #: The content of a new scenario's root snapshot.
    start: DefaultStart = field(default_factory=DefaultStart)

    def __post_init__(self) -> None:
        if self.pipeline_stages is None:
            # An uploaded trajectory is one sample, so two uploads make one job's data.
            stages = job_sized_stages(PipelineSpec(self.task_stage, self.batch_stage), job_samples=2)
            object.__setattr__(self, "pipeline_stages", stages)
        super().__post_init__()

    @classmethod
    def pipeline_spec(cls) -> PipelineSpec:
        return PipelineSpec(NoSamples, UploadJobData)


class UploadComponent(RecordingComponent):
    """A recording component whose facts start a new scenario from ``start``."""

    def __init__(self, batches: list[RecordedJob], start: DefaultStart) -> None:
        super().__init__(batches)
        self.start = start

    def stub_default_start(self) -> DefaultStart:
        return self.start


def dispatcher_for(
    tmp_path: Path,
    recipe: Recipe,
    factory: RepositoryBackendFactory | None = None,
    live: LiveObjects | None = None,
) -> Dispatcher:
    """A dispatcher with a data store, as deployment assembly builds it; the scenario storage owns the task files.
    ``live`` holds the runtime and the job components of ``recipe``; the dispatcher gets it with the data store.
    The default for an ``UploadRecipe`` is an ``UploadComponent``, whose jobs the dispatcher's training worker runs,
    and a stub runtime that several training scenarios share. The default for any other recipe is no object.

    ``factory`` defaults to an in-memory repository under ``tmp_path``; a test that restarts passes its own.
    """
    if live is None and isinstance(recipe, UploadRecipe):
        runtime = StubInferenceRuntime(shared_by_training_scenarios=True)
        live = live_objects(UploadComponent(recipe.batches, recipe.start), runtime)
    initial = tmp_path / "initial"
    initial.mkdir(parents=True, exist_ok=True)
    task_files = LocalTaskFileStore(tmp_path / "task-files")
    data = SQLiteDataStore(tmp_path / "data", task_files)
    coordination = memory_coordination()
    storage = SQLiteScenarioStorage(tmp_path / "records", task_files=task_files, data=data)
    dispatcher = Dispatcher(
        recipe,
        factory or InMemoryRepositoryBackend.factory(directory=tmp_path / "repository"),
        live=LiveObjects(data_store=data) if live is None else replace(live, data_store=data),
        scenario_storage=DataScenarioStorage(storage, data),
        record_storage=storage,
        coordination=coordination,
    )
    # The data service runs the data jobs, as assembly runs it.
    dispatcher.workers.add(
        DataWorker(
            dispatcher.queue,
            data_jobs(dispatcher, ArtifactSources.from_setting(()), ArchiveWriters()),
            worker_id="data-test",
        )
    )
    return dispatcher
