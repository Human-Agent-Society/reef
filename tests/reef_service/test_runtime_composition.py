"""Separate training and inference contracts, coordinated by the existing backend."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import reef
from reef.runtime import InferenceRuntime, PreparedTrainingStep, TrainingRuntime
from reef.runtime.adapters.executor_inference import ExecutorInferenceRuntime
from reef.runtime.adapters.executor_runtime import ExecutorTrainingRuntime, connect_executor_runtimes
from reef.runtime.candidates import ModelCandidate
from reef.train.runtime_backend import RuntimeTrainingBackend

from .test_executor_runtime import Coordinator


class CheckpointTrainer(TrainingRuntime):
    def __init__(self, checkpoint: Path) -> None:
        self.checkpoint = checkpoint

    def prepare_training_step(
        self, batch, step_preparer, algorithm_state, scenario_step, *, serving_runtime_load_id=None
    ):
        return PreparedTrainingStep("train", algorithm_state, {}, {"value": 7})

    def train_candidate(self, payload):
        self.checkpoint.write_text(str(payload["value"]))
        return ModelCandidate(
            candidate_id="job-1",
            training_job_id="job-1",
            checkpoint_path=str(self.checkpoint),
            current_runtime_load_id=None,
        )

    def reject_candidate(self, candidate, decision):
        self.checkpoint.unlink()


def test_training_exports_checkpoint_without_inference_or_aggregate_runtime(tmp_path):
    training = CheckpointTrainer(tmp_path / "checkpoint")
    assert not issubclass(TrainingRuntime, InferenceRuntime)
    assert not hasattr(reef, "ModelRuntime")
    for attribute in ("base_url", "inference_backend", "acquire_inference", "activate_candidate", "inference"):
        assert not hasattr(training, attribute)
    candidate = training.train_candidate({"value": 7})
    assert Path(candidate.checkpoint_path).read_text() == "7"
    training.shutdown()


@pytest.mark.parametrize("colocate", [False, True])
def test_existing_backend_keeps_receiver_paused_until_matching_durable_commit(colocate):
    async def run():
        control = Coordinator(colocate=colocate)
        training, inference = connect_executor_runtimes(train_group_handle=control)
        backend = RuntimeTrainingBackend(training, "sft", inference_runtime=inference)
        assert isinstance(training, ExecutorTrainingRuntime)
        assert isinstance(inference, ExecutorInferenceRuntime)
        assert not hasattr(inference, "train_candidate")
        candidate = backend.train_candidate({"rollout_id": 0})
        assert inference.inference_admission_status["open"] is (not colocate)
        backend.activate_candidate(candidate)
        pending = asyncio.create_task(inference.acquire_inference())
        try:
            await asyncio.sleep(0)
            assert not pending.done()
            backend.recover_pending_step(1, committed_training_job_id="different-job")
            assert inference.inference_admission_status["open"] is False
            backend.acknowledge_commit(1, candidate.training_job_id)
            handle = await asyncio.wait_for(pending, timeout=1)
            assert inference.current_runtime_load_id() == "engine:1"
            handle.release()
        finally:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
            training.shutdown()
            inference.shutdown()
        assert control.shutdown_events == ["shutdown"]

    asyncio.run(run())


def test_training_adapter_never_constructs_an_inference_component():
    training = ExecutorTrainingRuntime(Coordinator(inference_url=None))
    candidate = training.train_candidate({"rollout_id": 0})
    assert candidate.checkpoint_path == "/checkpoint"
    assert not hasattr(training, "inference_runtime")
    assert not hasattr(training, "base_url")
    assert not hasattr(training, "activate_candidate")
    training.shutdown()


def test_receiver_shutdown_does_not_stop_training_workers():
    control = Coordinator()
    training, inference = connect_executor_runtimes(train_group_handle=control)
    inference.shutdown()
    assert control.shutdown_events == []
    training.train_candidate({"rollout_id": 0})
    training.shutdown()
    assert control.shutdown_events == ["shutdown"]


def test_recovery_retargets_only_the_inference_component():
    control = Coordinator(inference_url="http://old-engine")
    training, inference = connect_executor_runtimes(train_group_handle=control)
    backend = RuntimeTrainingBackend(training, "sft", inference_runtime=inference)
    control.inference_url = "http://replacement-engine/"
    backend.recover_pending_step(0)
    assert inference.base_url == "http://replacement-engine"
    assert not hasattr(training, "base_url")
    training.shutdown()
    inference.shutdown()


def test_assembly_returns_two_components_and_rejects_conflicting_inference_selection():
    from reef.runtime.adapters.inference_proxy import InferenceProxyRuntime

    chosen = InferenceProxyRuntime(base_url="http://chosen-engine")
    control = Coordinator(inference_url=None)
    training, inference = connect_executor_runtimes(train_group_handle=control, inference=chosen)
    assert isinstance(training, TrainingRuntime)
    assert inference is chosen
    with pytest.raises(ValueError, match="pass inference or inference_url"):
        connect_executor_runtimes(train_group_handle=control, inference=chosen, inference_url="http://other-engine")
    training.shutdown()
    inference.shutdown()


def test_attached_receiver_waits_for_coordinator_recovery():
    control = Coordinator()
    training, inference = connect_executor_runtimes(train_group_handle=control)
    assert not inference.inference_admission_status["open"]
    RuntimeTrainingBackend(training, "sft", inference_runtime=inference)
    assert inference.inference_admission_status["open"]
    training.shutdown()
    inference.shutdown()
