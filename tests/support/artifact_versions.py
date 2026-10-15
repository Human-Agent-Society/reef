"""Shared scenario setup for experiment reporting and commit tests."""

from __future__ import annotations

import time

from tests.support._policy_recipe import TestPolicyRecipe
from tests.support.runtime_stubs import StubTrainingRuntime, runtime_bindings

from reef.artifact import InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.dispatcher import Dispatcher
from reef.observability import ExperimentTracker, NullExperimentLogger
from reef.recipe.checkpoint_strategy import EveryNVersions
from reef.runtime.interfaces import ActivatedModel, ModelCandidate, PreparedTrainingStep
from reef.storage.sqlite import SQLiteScenarioStorage


class RollbackRuntime(StubTrainingRuntime):
    def __init__(self, checkpoint_dir) -> None:
        super().__init__(base_url="http://trainer")
        self.checkpoint_dir = checkpoint_dir
        self.trained = 0
        self.restored: list[str] = []
        self.restore_order = []
        self.candidate_versions: dict[str, str] = {}

    @property
    def inference_handler(self):
        return None

    def prepare_training_step(
        self, batch, objective, algorithm_state, scheduling, scenario_step, *, parent_runtime_load_id=None
    ):
        del batch, objective
        return PreparedTrainingStep(
            action="train",
            payload={"scenario_step": scenario_step},
            next_algorithm_state={"steps": int(algorithm_state.get("steps", 0)) + 1},
            metrics={},
        )

    def train_candidate(self, payload):
        self.trained += 1
        version = f"w{self.trained}"
        checkpoint = self.checkpoint_dir / str(payload["scenario_step"])
        checkpoint.mkdir(parents=True)
        (checkpoint / "model.txt").write_text(version)
        job_id = f"job-{payload['scenario_step']}"
        self.candidate_versions[job_id] = version
        return ModelCandidate(
            candidate_id=job_id,
            training_job_id=job_id,
            checkpoint_path=str(checkpoint),
            current_runtime_load_id=None,
        )

    def activate_candidate(self, candidate):
        return ActivatedModel(candidate.candidate_id, self.candidate_versions[candidate.candidate_id])

    def reject_candidate(self, candidate):
        self.candidate_versions.pop(candidate.candidate_id, None)

    @property
    def supports_checkpoint_restore(self):
        return True

    def restore_checkpoint(self, artifact):
        assert not self.inference.inference_admission_status["open"]
        self.restore_order.append("training")

    def restore_serving_checkpoint(self, artifact):
        assert not self.inference.inference_admission_status["open"]
        self.restore_order.append("inference")
        version = artifact.local_path.joinpath("model.txt").read_text()
        self.restored.append(version)
        return f"restored:{version}"


class RecordingExperimentTracker(ExperimentTracker):
    def __init__(self) -> None:
        self.events = []
        self.rollbacks = []

    def bind_scenario(self, **kwargs):
        del kwargs
        return NullExperimentLogger()

    def correlation_metrics(self, context):
        del context
        return {}

    def record(self, event):
        self.events.append(event)

    def record_rollback(self, event):
        self.rollbacks.append(event)

    def close(self):
        pass


def inference(index: int, runtime_load_id: str | None = None) -> AgentRecord:
    """The inference record of step ``index``, served by the version that step ``index - 1`` published.

    ``runtime_load_id`` names another serving version, such as one that a rollback restored.
    """
    version = runtime_load_id if runtime_load_id is not None or index == 1 else f"w{index - 1}"
    payload: dict[str, object] = {"tokens": [1, 2], "loss_mask": [0, 1], "rollout_log_probs": [-0.2]}
    if version is not None:
        payload["runtime_load_id"] = version
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


def dispatcher(tmp_path, *, checkpoint_every: int = 1, experiment_tracker=None):
    initial = tmp_path / "initial"
    initial.mkdir(exist_ok=True)
    (initial / "model.txt").write_text("w0")
    runtime = RollbackRuntime(tmp_path / "exported")
    backend_factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    value = Dispatcher(
        TestPolicyRecipe(
            **runtime_bindings(runtime),
            batch_size=1,
            checkpoint_strategy=EveryNVersions(checkpoint_every),
        ),
        backend_factory,
        local_artifact_dir=tmp_path / "staged",
        experiment_tracker=experiment_tracker,
        scenario_storage=SQLiteScenarioStorage(tmp_path / "agent-record"),
    )
    return value, runtime, backend_factory


def train(dispatcher: Dispatcher, index: int, runtime_load_id: str | None = None) -> None:
    dispatcher.accept_record(inference(index, runtime_load_id))
    dispatcher.accept_record(report(index))
    for _ in range(1000):
        if dispatcher.get_or_create_scenario("math").scenario_step >= index:
            return
        time.sleep(0.001)
    raise AssertionError("async training did not commit")
