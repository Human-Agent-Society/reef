"""A recipe that trains uploaded trajectories, and a dispatcher and request bodies for its upload tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reef.artifact import InMemoryRepositoryBackend
from reef.artifact.repository import RepositoryBackendFactory
from reef.artifact.sources import ArtifactSources
from reef.core.evaluation import EvaluationResult, SelectionDecision, UpdateCandidate
from reef.dispatcher import Dispatcher
from reef.recipe import Recipe
from reef.record2sample import NoSamples
from reef.sample2job import Sample2Job
from reef.service.data_service import DataWorker
from reef.service.data_transfer import ArchiveWriters
from reef.service.task_import import DataScenarioStorage
from reef.storage.sql_data import SQLiteDataStore
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.storage.tasks import LocalTaskFileStore
from reef.train import CandidateBackend, JobData, PreparedStep, Trainer, TrainStepResult
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


def trajectory_items(*item_ids: str, scenario: str | None = "s") -> dict[str, Any]:
    """The body of a create call: a trajectory item for each ID, used by ``scenario`` when it is given."""
    scenarios = {} if scenario is None else {scenario: "train"}
    return {
        "items": [
            {"id": item_id, "content": atif(item_id), "metadata": {"kind": "trajectory", "scenarios": scenarios}}
            for item_id in item_ids
        ]
    }


class CountingBackend(CandidateBackend):
    """Keeps every batch and skips its step, so the trainer commits it."""

    def __init__(self, batches: list[JobData]) -> None:
        self.batches = batches

    def initial_state(self) -> Mapping[str, Any]:
        return {"trained": 0}

    def prepare_step(self, batch: JobData, state: Mapping[str, Any], scenario_step: int) -> PreparedStep:
        self.batches.append(batch)
        return PreparedStep.skipped(state={"trained": int(state["trained"]) + len(batch.samples)})

    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
        raise NotImplementedError

    def settle_step(self, prepared: PreparedStep, decision: SelectionDecision) -> TrainStepResult:
        raise NotImplementedError

    def abort_step(self, prepared: PreparedStep) -> None:
        pass


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
        self, scenario, records, *, algorithm_state=None, experiment_logger=None, task_storage=None, samples=None
    ):
        return Trainer.build(
            scenario,
            records,
            pipeline_factory=lambda context: TrainingPipeline(
                context, record2sample=self.task_stage(), sample2job=self.batch_stage(context, None, None)
            ),
            candidate_backend=CountingBackend(self.batches),
            algorithm_state=algorithm_state,
            experiment_logger=experiment_logger,
            training_mode=self.training_mode,
            task_storage=task_storage,
            samples=samples,
            # An uploaded trajectory is one sample, so two uploads make one job's data.
            default_job_samples=2,
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
    )
    # The data service copies the records to record items, as assembly runs it.
    dispatcher.workers.add(
        DataWorker(dispatcher, ArtifactSources.from_setting(()), ArchiveWriters(), worker_id="data-test")
    )
    return dispatcher
