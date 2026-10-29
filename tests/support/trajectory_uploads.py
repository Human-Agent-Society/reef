"""A recipe that trains uploaded trajectories, and a dispatcher and request bodies for its upload tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reef.artifact import InMemoryRepositoryBackend
from reef.core.evaluation import EvaluationResult, SelectionDecision, UpdateCandidate
from reef.dispatcher import Dispatcher
from reef.recipe import Recipe
from reef.record2sample import NoSamples
from reef.sample2job import Sample2Job
from reef.storage.sqlite import SQLiteScenarioStorage
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


def uploaded(*trajectory_ids: str) -> dict[str, Any]:
    return {
        "trajectories": [
            {"trajectory_id": trajectory_id, "trajectory": atif(trajectory_id)} for trajectory_id in trajectory_ids
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


def dispatcher_for(tmp_path: Path, recipe: Recipe) -> Dispatcher:
    initial = tmp_path / "initial"
    initial.mkdir(exist_ok=True)
    return Dispatcher(
        recipe,
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
    )
