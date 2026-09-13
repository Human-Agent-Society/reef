"""Independent training contracts and composed inference/publication behavior."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from reef.core.batches import TrainingBatch
from reef.runtime import (
    ExecutorModelRuntime,
    InferenceRuntime,
    ModelRuntime,
    PreparedTrainingStep,
    RuntimeRegistry,
    TrainingJobResult,
    TrainingRuntime,
)
from reef.runtime.adapters.inference_proxy import InferenceProxyRuntime

from .test_executor_runtime import Coordinator


class CheckpointTrainingRuntime(TrainingRuntime):
    def __init__(self, checkpoint: Path) -> None:
        self.checkpoint = checkpoint

    def health(self) -> Mapping[str, Any]:
        return {"ok": True}

    def prepare_training_step(
        self, batch: TrainingBatch, step_preparer: str, algorithm_state: Mapping[str, Any]
    ) -> PreparedTrainingStep:
        return PreparedTrainingStep("train", algorithm_state, {}, {"value": 7})

    def execute_training_job(self, payload: Mapping[str, Any]) -> TrainingJobResult:
        self.checkpoint.write_text(str(payload["value"]))
        return TrainingJobResult("checkpoint", "candidate:1", str(self.checkpoint), training_job_id="job-1")


def test_training_runs_without_inference_or_publication_contract(tmp_path):
    runtime = CheckpointTrainingRuntime(tmp_path / "checkpoint")
    assert not issubclass(TrainingRuntime, InferenceRuntime)
    assert not issubclass(ModelRuntime, InferenceRuntime)
    assert not isinstance(runtime, ModelRuntime)
    for attribute in ("base_url", "inference_backend", "acquire_inference", "activate_candidate"):
        assert not hasattr(runtime, attribute)
    assert runtime.health() == {"ok": True}
    result = runtime.execute_training_job({"value": 7})
    assert result.outcome == "checkpoint"
    assert Path(result.checkpoint_path).read_text() == "7"
    runtime.shutdown()


@pytest.mark.parametrize("colocate", [False, True])
def test_composed_inference_waits_for_durable_commit_through_both_entry_points(colocate):
    async def run():
        training = Coordinator(colocate=colocate)
        inference = InferenceProxyRuntime(base_url="http://independent-engine")
        runtime = ExecutorModelRuntime(train_group_handle=training, inference=inference)
        assert runtime.training is training
        assert runtime.inference is inference
        assert runtime.inference_backend is inference.inference_backend
        candidate = runtime.train_candidate({"rollout_id": 0})
        assert inference.inference_admission_status["open"] is (not colocate)
        runtime.activate_candidate(candidate)
        assert inference.inference_admission_status["open"] is False
        pending = [asyncio.create_task(owner.acquire_inference()) for owner in (runtime, inference)]
        try:
            await asyncio.sleep(0)
            assert all(not task.done() for task in pending)
            # A different durable job cannot authorize either entry point.
            runtime.reconcile_training_job(1, committed_training_job_id="different-job")
            assert inference.inference_admission_status["open"] is False
            runtime.reconcile_training_job(1, committed_training_job_id=candidate.training_job_id)
            handles = await asyncio.wait_for(asyncio.gather(*pending), timeout=1)
            assert runtime.inference_admission_status == {"open": True, "active": 2}
            assert inference.inference_admission_status == runtime.inference_admission_status
            for handle in handles:
                handle.release()
            assert inference.inference_admission_status["active"] == 0
        finally:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            runtime.shutdown()
        assert inference.inference_admission_status["open"] is False

    asyncio.run(run())


def test_factory_accepts_an_independent_inference_runtime_without_endpoint_discovery():
    from reef.runtime.executor.uniproc import UniProcExecutor

    inference = InferenceProxyRuntime(base_url="http://chosen-engine", inference_timeout_s=17)
    executor = UniProcExecutor.from_workers((Coordinator(inference_url=None),), owned=True)
    runtime = RuntimeRegistry().build(
        {"type": "executor_training", "executor": executor, "inference": inference}, model_path="model"
    )
    try:
        assert isinstance(runtime, ModelRuntime)
        assert runtime.inference is inference
        assert runtime.base_url == "http://chosen-engine"
        assert runtime.inference_timeout_s == 17
    finally:
        runtime.shutdown()


def test_recovery_updates_composed_endpoint_and_backend_together():
    training = Coordinator(inference_url="http://original-engine")
    runtime = ExecutorModelRuntime(train_group_handle=training)
    inference = runtime.inference
    backend = inference.inference_backend
    training.inference_url = "http://replacement-engine/"
    runtime.reconcile_training_job(0)
    assert runtime.inference is inference
    assert runtime.inference_backend is backend
    assert runtime.base_url == inference.base_url == "http://replacement-engine"
    runtime.shutdown()


def test_shutdown_closes_inference_even_when_training_cleanup_fails():
    events = []

    class FailingTraining(Coordinator):
        def shutdown(self):
            events.append("training")
            raise RuntimeError("training cleanup failed")

    class OwnedInference(InferenceProxyRuntime):
        def shutdown(self):
            events.append("inference")

    inference = OwnedInference(base_url="http://engine")
    runtime = ExecutorModelRuntime(train_group_handle=FailingTraining(), inference=inference)
    with pytest.raises(RuntimeError, match="training cleanup failed"):
        runtime.shutdown()
    assert events == ["training", "inference"]
    assert inference.inference_admission_status["open"] is False


def test_composition_rejects_conflicting_endpoint_selection():
    with pytest.raises(ValueError, match="pass inference or inference_url"):
        ExecutorModelRuntime(
            train_group_handle=Coordinator(),
            inference=InferenceProxyRuntime(base_url="http://engine"),
            inference_url="http://other-engine",
        )
