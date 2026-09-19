"""Reef schedules two independent runtimes without backend-specific control calls."""

from __future__ import annotations

import pytest

from reef.core.batches import TrainingBatch
from reef.core.evaluation import EvaluationResult, SelectionDecision
from reef.runtime.interfaces import (
    ActivatedModel,
    CandidateTrainingDeferred,
    InferenceHandler,
    InferenceRuntime,
    ModelCandidate,
    PreparedTrainingStep,
    StaleCandidate,
    TrainingRuntime,
)
from reef.runtime.scheduler import RuntimeScheduler
from reef.train.algos import StepScheduling


class CheckpointTrainer(TrainingRuntime):
    """An exporter with its own journal and no receiver dependency."""

    def __init__(self, *, colocate=False, concurrent=False, journal=True, max_staleness=0):
        self.journal = {"status": "IDLE", "colocate": colocate} if journal else None
        self.concurrent = concurrent
        self.staleness = max_staleness
        self.failure = None
        self.calls = []

    @property
    def concurrent_training_scenarios(self):
        return self.concurrent

    @property
    def max_staleness(self):
        return self.staleness

    def training_job_status(self):
        return self.journal

    def prepare_training_step(
        self, batch, objective, algorithm_state, scheduling, scenario_step, *, serving_runtime_load_id=None
    ):
        self.calls.append(("prepare", serving_runtime_load_id))
        return PreparedTrainingStep("train", algorithm_state, {}, {"rollout_id": scenario_step})

    def train_candidate(self, payload):
        self.calls.append(("train", payload))
        if self.failure is not None:
            raise self.failure
        if self.journal is not None:
            self.journal.update(status="CHECKPOINT", training_job_id="job-1", rollout_id=0)
        return ModelCandidate(
            candidate_id="job-1",
            training_job_id="job-1",
            checkpoint_path="/checkpoints/job-1",
            current_runtime_load_id=None,
        )

    def reject_candidate(self, candidate, decision):
        self.reject_training_job(candidate.training_job_id)

    def reject_training_job(self, training_job_id):
        self.calls.append(("reject", training_job_id))
        if self.failure is not None:
            raise self.failure
        if self.journal is not None:
            self.journal["status"] = "REJECTED"


class WeightReceiver(InferenceRuntime):
    """A receiver that knows versions and operation IDs, never a trainer."""

    def __init__(self):
        super().__init__(base_url="http://inference")
        self.loaded = "engine:0"
        self.calls = []
        self.ack_failure = None

    @property
    def inference_handler(self) -> InferenceHandler:
        raise NotImplementedError("scheduler tests never execute provider requests")

    def serving_runtime_load_id(self):
        return self.loaded

    def activate_candidate(self, candidate):
        assert not self.inference_admission_status["open"]
        self.calls.append(("activate", candidate.training_job_id))
        self.loaded = "engine:1"
        return ActivatedModel(candidate.candidate_id, self.loaded)

    def resume_weight_update(self, training_job_id):
        assert not self.inference_admission_status["open"]
        self.calls.append(("recover", training_job_id))
        self.loaded = "engine:1"
        return ActivatedModel(training_job_id, self.loaded)

    def acknowledge_publication(self, training_job_id):
        assert not self.inference_admission_status["open"]
        self.calls.append(("acknowledge", training_job_id))
        if self.ack_failure is not None:
            raise self.ack_failure


def staged_journal(training, *, state="READY_TO_COMMIT", scenario=None):
    training.journal.update(status=state, training_job_id="job-1", rollout_id=0, scenario=scenario)


@pytest.mark.parametrize("colocate", [False, True])
def test_selected_weights_remain_unpublished_until_matching_commit(colocate):
    training = CheckpointTrainer(colocate=colocate)
    inference = WeightReceiver()
    scheduler = RuntimeScheduler(training, inference)

    candidate = scheduler.train_candidate({"rollout_id": 0})
    assert candidate.current_runtime_load_id == "engine:0"
    assert inference.inference_admission_status["open"] is (not colocate)
    scheduler.activate_candidate(candidate)
    staged_journal(training)
    assert inference.current_runtime_load_id() == "engine:0"
    assert inference.serving_runtime_load_id() == "engine:1"

    scheduler.acknowledge_commit(1, "different-job")
    assert inference.calls == [("activate", "job-1")]
    assert not inference.inference_admission_status["open"]
    scheduler.acknowledge_commit(1, "job-1")
    assert inference.calls == [("activate", "job-1"), ("acknowledge", "job-1")]
    assert inference.current_runtime_load_id() == "engine:1"
    assert inference.inference_admission_status["open"]
    assert training.calls == [("train", {"rollout_id": 0})]


def test_local_candidate_backend_uses_same_publication_order_without_remote_journal():
    training = CheckpointTrainer(journal=False)
    inference = WeightReceiver()
    scheduler = RuntimeScheduler(training, inference)
    candidate = scheduler.train_candidate({})
    scheduler.activate_candidate(candidate)
    assert not inference.inference_admission_status["open"]
    assert inference.current_runtime_load_id() == "engine:0"
    scheduler.acknowledge_commit(1, candidate.training_job_id)
    assert inference.current_runtime_load_id() == "engine:1"
    assert inference.inference_admission_status["open"]


def test_recovery_preserves_another_scenarios_commit_fence():
    training = CheckpointTrainer(concurrent=True)
    staged_journal(training, state="UPDATING_WEIGHTS", scenario="math")
    inference = WeightReceiver()
    scheduler = RuntimeScheduler(training, inference)

    scheduler.acknowledge_commit(1, "job-1", scenario="code")
    assert inference.calls == []
    assert not inference.inference_admission_status["open"]
    scheduler.acknowledge_commit(1, "job-1", scenario="math")
    assert inference.calls == [("recover", "job-1"), ("acknowledge", "job-1")]
    assert inference.inference_admission_status["open"]
    assert training.calls == []


def test_failed_commit_acknowledgement_keeps_loaded_weights_unavailable_until_retry():
    training = CheckpointTrainer()
    inference = WeightReceiver()
    scheduler = RuntimeScheduler(training, inference)
    scheduler.activate_candidate(scheduler.train_candidate({}))
    staged_journal(training)
    inference.ack_failure = TimeoutError("receiver acknowledgement result unknown")
    with pytest.raises(TimeoutError, match="result unknown"):
        scheduler.acknowledge_commit(1, "job-1")
    assert not inference.inference_admission_status["open"]
    assert inference.current_runtime_load_id() == "engine:0"

    inference.ack_failure = None
    scheduler.acknowledge_commit(1, "job-1")
    assert inference.calls == [("activate", "job-1"), ("acknowledge", "job-1"), ("acknowledge", "job-1")]
    assert inference.current_runtime_load_id() == "engine:1"
    assert inference.inference_admission_status["open"]


@pytest.mark.parametrize("failure", [StaleCandidate(), CandidateTrainingDeferred({"available": False})])
def test_retryable_training_outcome_reopens_unchanged_colocated_inference(failure):
    training = CheckpointTrainer(colocate=True)
    inference = WeightReceiver()
    scheduler = RuntimeScheduler(training, inference)
    training.failure = failure
    with pytest.raises(type(failure)):
        scheduler.train_candidate({})
    assert scheduler.operations.snapshot()["stale_batches_total"] == int(isinstance(failure, StaleCandidate))
    assert inference.inference_admission_status["open"]
    assert inference.current_runtime_load_id() == "engine:0"
    assert inference.calls == []


@pytest.mark.parametrize("durable_state,admit", [("IDLE", True), ("RUNNING", False)])
def test_uncertain_colocated_failure_only_reopens_if_no_training_started(durable_state, admit):
    training = CheckpointTrainer(colocate=True)
    inference = WeightReceiver()
    scheduler = RuntimeScheduler(training, inference)
    training.journal["status"] = durable_state
    training.failure = TimeoutError("training result unknown")
    with pytest.raises(TimeoutError, match="training result unknown"):
        scheduler.train_candidate({})
    assert inference.inference_admission_status["open"] is admit
    assert inference.current_runtime_load_id() == "engine:0"


def test_rejection_must_finish_before_colocated_inference_reopens():
    training = CheckpointTrainer(colocate=True)
    inference = WeightReceiver()
    scheduler = RuntimeScheduler(training, inference)
    candidate = scheduler.train_candidate({})
    decision = SelectionDecision("reject", "test", "1", "failed evaluation", EvaluationResult("test", "1", {}))
    training.failure = TimeoutError("rejection result unknown")
    with pytest.raises(TimeoutError, match="rejection result unknown"):
        scheduler.reject_candidate(candidate, decision)
    assert not inference.inference_admission_status["open"]
    staged_journal(training, state="REJECTING")
    training.failure = None
    scheduler.recover_pending_step(0)
    assert inference.inference_admission_status["open"]
    assert inference.current_runtime_load_id() == "engine:0"
    assert inference.calls == []
    assert training.calls[-1] == ("reject", "job-1")


@pytest.mark.parametrize("max_staleness,expected", [(0, "engine:0"), (1, "engine:1")])
def test_preparation_passes_only_the_required_serving_version_value(max_staleness, expected):
    training = CheckpointTrainer(max_staleness=max_staleness)
    inference = WeightReceiver()
    scheduler = RuntimeScheduler(training, inference)
    inference.loaded = "engine:1"
    prepared = scheduler.prepare_training_step(TrainingBatch("batch-1"), "test", {}, StepScheduling(), 0)
    assert prepared.payload == {"rollout_id": 0}
    assert training.calls == [("prepare", expected)]
    assert not hasattr(training, "inference_runtime")
    assert not hasattr(inference, "training_runtime")


@pytest.mark.parametrize("status", ["READY_TO_COMMIT", "HEAD_COMMITTED", "COMPLETE"])
def test_attaching_to_an_unacknowledged_update_starts_with_closed_admission(status):
    training = CheckpointTrainer()
    staged_journal(training, state=status)
    inference = WeightReceiver()
    inference.loaded = "engine:1"
    scheduler = RuntimeScheduler(training, inference)
    assert not inference.inference_admission_status["open"]
    assert inference.current_runtime_load_id() is None
    scheduler.acknowledge_commit(1, "job-1")
    assert inference.inference_admission_status["open"]
    assert inference.current_runtime_load_id() == "engine:1"
