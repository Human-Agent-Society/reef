"""Shared scenario setup for experiment reporting and commit tests."""

from __future__ import annotations

import time

from tests.support._policy_recipe import TestPolicyRecipe
from tests.support.runtime_stubs import StubInferenceRuntime, StubTrainingComponent, runtime_bindings

from reef.artifact import InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.dispatcher import Dispatcher
from reef.observability import ExperimentTracker, NullExperimentLogger
from reef.storage.sqlite import SQLiteScenarioStorage


class ImportingInference(StubInferenceRuntime):
    """Serve each release under the version that its materialized ``model.txt`` names; the initial one is ``w0``.

    ``restore_order`` is shared with the training component, so a test sees which side an operation reached first.
    """

    def __init__(self, restore_order: list[str]) -> None:
        super().__init__(base_url="http://trainer")
        self.restore_order = restore_order

    def serve(self, artifact):
        return super().serve(artifact.materialize())

    def activate_checkpoint(self, artifact, *, scenario=None):
        self.restore_order.append("inference")
        return super().activate_checkpoint(artifact, scenario=scenario)


class ImportingComponent(StubTrainingComponent):
    """A component whose ``n``-th job saves ``w<n>``; ``train-from`` imports a release as the next job's start.

    ``inference`` is the runtime that serves its checkpoints. ``activated`` lists the versions that it served, and
    ``trained`` counts the submitted jobs.
    """

    def __init__(self, checkpoint_dir) -> None:
        super().__init__(checkpoint_dir)
        self.restore_order: list[str] = []
        self.inference = ImportingInference(self.restore_order)

    @property
    def activated(self) -> list[str]:
        return self.inference.activated

    @property
    def trained(self) -> int:
        return len(self.submitted)

    def import_start(self, release):
        self.restore_order.append("training")
        return super().import_start(release)


class RecordingExperimentTracker(ExperimentTracker):
    def __init__(self) -> None:
        self.events = []
        self.starts = []

    def bind_scenario(self, **kwargs):
        del kwargs
        return NullExperimentLogger()

    def correlation_metrics(self, context):
        del context
        return {}

    def record(self, event):
        self.events.append(event)

    def record_train_from(self, event):
        self.starts.append(event)

    def close(self):
        pass


def inference(index: int, runtime_load_id: str | None = None) -> AgentRecord:
    """The inference record of job ``index``, served by the version that job ``index - 1`` published.

    ``runtime_load_id`` names another serving version, such as one that an activation served.
    """
    version = f"w{index - 1}" if runtime_load_id is None else runtime_load_id
    payload: dict[str, object] = {
        "tokens": [1, 2],
        "loss_mask": [0, 1],
        "rollout_log_probs": [-0.2],
        "runtime_load_id": version,
    }
    return AgentRecord.create(
        scenario="math",
        request_type=RequestType.INFERENCE,
        payload=payload,
        agent_record_id=f"i{index}",
    )


def report(index: int) -> AgentRecord:
    return AgentRecord.create(
        scenario="math",
        request_type=RequestType.REPORT,
        payload={"score": 1.0, "references": [f"i{index}"]},
        agent_record_id=f"r{index}",
        references=(f"i{index}",),
    )


def dispatcher(tmp_path, *, experiment_tracker=None):
    initial = tmp_path / "initial"
    initial.mkdir(exist_ok=True)
    (initial / "model.txt").write_text("w0")
    runtime = ImportingComponent(tmp_path / "exported")
    backend_factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    value = Dispatcher(
        TestPolicyRecipe(**runtime_bindings(runtime, runtime.inference)),
        backend_factory,
        local_artifact_dir=tmp_path / "staged",
        experiment_tracker=experiment_tracker,
        scenario_storage=SQLiteScenarioStorage(tmp_path / "agent-record"),
    )
    return value, runtime, backend_factory


def finished_jobs(dispatcher: Dispatcher) -> int:
    """The number of training jobs that scenario ``math`` finished, as ``GET /reef/status`` reports it."""
    dispatcher.get_or_create_scenario("math")
    return dispatcher.build_training_status()["scenarios"]["math"]["finished_jobs"]


def train(dispatcher: Dispatcher, index: int, runtime_load_id: str | None = None) -> None:
    """Accept job ``index``'s records and wait until the job it submits ends.

    A job commits three times: its submission, the decision on its checkpoint, and its end.
    """
    finished = finished_jobs(dispatcher)
    dispatcher.accept_record(inference(index, runtime_load_id))
    dispatcher.accept_record(report(index))
    for _ in range(5000):
        if finished_jobs(dispatcher) > finished:
            return
        time.sleep(0.001)
    raise AssertionError("async training did not finish its job")
