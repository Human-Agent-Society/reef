from __future__ import annotations

import asyncio
import pickle
from collections.abc import Mapping
from typing import Any

import pytest

from reef.artifact import LiveWeightArtifactRef
from reef.core import RuntimeLoadSpan
from reef.runtime import (
    PreparedTrainingStep,
    RayRuntime,
    RayRuntimeError,
    RayTrainGroupHandle,
    TrainingJobResult,
    TrainingRuntime,
)
from reef.runtime.adapters import ray_runtime
from reef.runtime.adapters.ray_runtime import RemoteRayTrainGroupHandle
from reef.runtime.base import InferenceAdmissionController
from reef.runtime.inference import InferenceBackend, InferenceStream
from reef.service.app import RequestService
from reef.service.streaming import stream_record
from reef.surface import RuntimeLoadMismatch, create_weight_surface
from reef.train.algos import StepScheduling, StepSignal
from reef.train.evaluation import EvaluationResult, SelectionDecision
from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_step as slime_prepare_step
from reef.train.types import GroupedPolicyBatch, PolicyBatch, PolicySample

from ._grouped_pg import GROUPED_PG_PREPARER as _TEST_GROUPED_PREPARER

_TEST_SFT_PREPARER = "reef_service.test_ray_runtime:_prepare_test_sft"


def _prepare_test_sft(batch, state) -> StepSignal:
    """Package-test custom preparer; cookbook examples are not in the wheel."""
    if not isinstance(batch, PolicyBatch):
        raise TypeError(f"sft requires PolicyBatch, got {type(batch).__name__}")
    return StepSignal(
        action="train",
        loss_family="sft",
        next_algorithm_state={"steps": int(state.get("steps", 0)) + 1},
    )


class FakeTrainGroupHandle(RayTrainGroupHandle):
    def health(self) -> Mapping[str, Any]:
        return {
            "colocate": False,
            "training_job": {
                "deferred_weight_update": True,
                "status": getattr(self, "status", "IDLE"),
                "rollout_id": 0,
                "training_job_id": "job-0",
            },
        }

    def prepare_training_step(
        self,
        batch,
        step_preparer: str,
        algorithm_state: Mapping[str, Any],
    ) -> PreparedTrainingStep:
        return slime_prepare_step(batch, step_preparer, algorithm_state)

    def execute_training_job(self, payload: Mapping[str, Any]) -> TrainingJobResult:
        del payload
        self.status = "CHECKPOINT"
        return TrainingJobResult(
            outcome="checkpoint",
            runtime_load_id="pending",
            checkpoint_path="/checkpoint",
            training_job_id="job-0",
        )

    def update_serving_weights(self, training_job_id: str) -> TrainingJobResult:
        assert training_job_id == "job-0"
        self.status = "READY_TO_COMMIT"
        return TrainingJobResult(
            outcome="complete",
            runtime_load_id="v1",
            checkpoint_path="/checkpoint",
            training_job_id=training_job_id,
        )

    def reject_training_candidate(self, training_job_id: str) -> None:
        assert training_job_id == "job-0"
        self.status = "REJECTED"

    def acknowledge_training_commit(self, training_job_id: str) -> None:
        assert training_job_id == "job-0"
        self.status = "COMPLETE"


class DeferredWeightUpdateTrainGroupHandle(FakeTrainGroupHandle):
    def __init__(
        self,
        *,
        colocate: bool = False,
        status: str = "IDLE",
        rollout_id: int = 0,
        lora_adapter: str | None = None,
    ) -> None:
        self.colocate = colocate
        self.status = status
        self.rollout_id = rollout_id
        self.training_job_id = f"job-{rollout_id}"
        self.runtime_load_id = "engine:0"
        self.lora_adapter = lora_adapter
        self.calls: list[str] = []

    def serving_runtime_load_id(self) -> str:
        return self.runtime_load_id

    def health(self) -> Mapping[str, Any]:
        return {
            "colocate": self.colocate,
            "lora_adapter": self.lora_adapter,
            "training_job": {
                "deferred_weight_update": True,
                "status": self.status,
                "rollout_id": self.rollout_id,
                "training_job_id": self.training_job_id,
                "commit_acknowledged": self.status == "COMPLETE",
            },
        }

    def execute_training_job(self, payload: Mapping[str, Any]) -> TrainingJobResult:
        del payload
        self.calls.append("execute")
        self.status = "CHECKPOINT"
        return TrainingJobResult(
            outcome="checkpoint",
            runtime_load_id="pending",
            checkpoint_path="/checkpoint",
            training_job_id=self.training_job_id,
        )

    def update_serving_weights(self, training_job_id: str) -> TrainingJobResult:
        assert training_job_id == self.training_job_id
        self.calls.append("update_weights")
        self.runtime_load_id = "engine:1"
        self.status = "READY_TO_COMMIT"
        return TrainingJobResult(
            outcome="complete",
            runtime_load_id="engine:1",
            checkpoint_path="/checkpoint",
            training_job_id=training_job_id,
        )

    def acknowledge_training_commit(self, training_job_id: str) -> None:
        assert training_job_id == self.training_job_id
        self.calls.append("acknowledge")
        self.status = "COMPLETE"

    def reject_training_candidate(self, training_job_id: str) -> None:
        assert training_job_id == self.training_job_id
        self.calls.append("reject")
        self.status = "REJECTED"


def policy_batch() -> PolicyBatch:
    return PolicyBatch(
        "batch-1",
        (
            PolicySample(
                "i1",
                (1, 2),
                (0, 1),
                (-0.2, -0.15),
                1.0,
                "v0",
                topk_indices=((1, 2), (3, 4)),
                topk_log_probs=((-0.2, -0.9), (-0.15, -0.8)),
            ),
            PolicySample(
                "i2",
                (3, 4),
                (1, 1),
                (-0.1, -0.05),
                0.5,
                "v0",
                topk_indices=((5, 6), (7, 8)),
                topk_log_probs=((-0.1, -0.7), (-0.05, -0.6)),
            ),
        ),
    )


def grouped_policy_batch(*, versions: tuple[str | None, str | None] = ("v0", "v0")) -> GroupedPolicyBatch:
    return GroupedPolicyBatch(
        "batch-2",
        (
            (
                PolicySample("g1", (1, 2), (1,), (-0.1,), 0.2, versions[0]),
                PolicySample("g2", (3, 4), (1,), (-0.2,), 0.8, versions[1]),
            ),
        ),
    )


@pytest.mark.unit
def test_ray_runtime_is_a_training_runtime() -> None:
    runtime = RayRuntime(train_group_handle=FakeTrainGroupHandle(), inference_url="http://router")

    assert isinstance(runtime, TrainingRuntime)
    assert runtime.inference_backend is not None


@pytest.mark.unit
def test_ray_runtime_reports_the_served_adapter_from_train_group_health() -> None:
    plain = RayRuntime(
        train_group_handle=DeferredWeightUpdateTrainGroupHandle(),
        inference_url="http://router",
    )
    assert plain.serving_adapter_name() is None

    lora = RayRuntime(
        train_group_handle=DeferredWeightUpdateTrainGroupHandle(lora_adapter="reef_lora"),
        inference_url="http://router",
    )
    assert lora.serving_adapter_name() == "reef_lora"


@pytest.mark.unit
def test_ray_runtime_serves_one_adapter_per_scenario_from_train_group_health() -> None:
    handle = DeferredWeightUpdateTrainGroupHandle()
    handle.lora_mode = "scenario"
    handle.lora_adapters = {"math": {"runtime_load_id": "engine:3", "adapter": "reef-adapter-bWF0aA.engine:3"}}
    original_health = handle.health

    def health():
        return {**original_health(), "lora_mode": handle.lora_mode, "lora_adapters": handle.lora_adapters}

    handle.health = health  # type: ignore[method-assign]
    runtime = RayRuntime(train_group_handle=handle, inference_url="http://router")
    assert runtime.concurrent_training_scenarios is True
    assert runtime.serving_adapter_name() is None
    assert runtime.serving_adapter_runtime_load_id("math") == "engine:3"
    assert runtime.serving_adapter_runtime_load_id("code") is None

    plain = RayRuntime(train_group_handle=DeferredWeightUpdateTrainGroupHandle(), inference_url="http://router")
    assert plain.concurrent_training_scenarios is False


@pytest.mark.unit
def test_ray_runtime_rejects_a_malformed_served_adapter_name() -> None:
    handle = DeferredWeightUpdateTrainGroupHandle()
    handle.lora_adapter = ""
    with pytest.raises(RayRuntimeError, match="serving adapter name"):
        RayRuntime(train_group_handle=handle, inference_url="http://router")


@pytest.mark.unit
def test_ray_runtime_builds_a_tokenizer_aware_inference_backend() -> None:
    captured = {}
    sentinel = object()

    def factory(upstream_url, *, model_path, timeout_s, **config):
        captured.update(upstream_url=upstream_url, model_path=model_path, timeout_s=timeout_s, config=config)
        return sentinel

    runtime = RayRuntime(
        train_group_handle=FakeTrainGroupHandle(),
        inference_url="http://router/",
        model_path="/models/qwen",
        inference_timeout_s=42,
        inference_backend_factory=factory,
        inference_backend_config={"tool_call_parser": "qwen25"},
    )

    assert runtime.inference_backend is sentinel
    assert captured == {
        "upstream_url": "http://router",
        "model_path": "/models/qwen",
        "timeout_s": 42,
        "config": {"tool_call_parser": "qwen25"},
    }


@pytest.mark.unit
def test_ray_runtime_prepares_and_executes_one_transaction() -> None:
    runtime = RayRuntime(train_group_handle=FakeTrainGroupHandle(), inference_url="http://router")
    prepared = runtime.prepare_training_step(policy_batch(), "sft", {}, 7)

    assert prepared.payload is not None
    payload = prepared.payload
    assert payload["rollout_id"] == 7 and payload["expected_runtime_load_id"] == "v0"
    assert runtime.execute_training_job(payload) == TrainingJobResult(
        outcome="complete",
        runtime_load_id="v1",
        checkpoint_path="/checkpoint",
        training_job_id="job-0",
    )


@pytest.mark.unit
def test_ray_runtime_requires_deferred_weight_updates() -> None:
    class AtomicHandle(FakeTrainGroupHandle):
        def health(self) -> Mapping[str, Any]:
            return {"training_job": {"deferred_weight_update": False, "status": "COMPLETE"}}

    with pytest.raises(RayRuntimeError, match="requires deferred serving-weight updates"):
        RayRuntime(train_group_handle=AtomicHandle(), inference_url="http://router")


@pytest.mark.unit
def test_ray_runtime_prepares_sft_without_reef_advantages() -> None:
    runtime = RayRuntime(train_group_handle=FakeTrainGroupHandle(), inference_url="http://router")

    prepared = runtime.prepare_training_step(policy_batch(), _TEST_SFT_PREPARER, {}, 3)
    assert prepared.payload is not None
    payload = prepared.payload

    assert payload["loss"] == "sft"
    assert "advantages" not in payload
    assert payload["expected_runtime_load_id"] == "v0"
    assert payload["rollout_id"] == 3


@pytest.mark.unit
def test_ray_runtime_rejects_unstructured_training_results() -> None:
    class InvalidResultTrainGroup(FakeTrainGroupHandle):
        def execute_training_job(self, payload: Mapping[str, Any]) -> Any:
            return dict(payload)

    runtime = RayRuntime(train_group_handle=InvalidResultTrainGroup(), inference_url="http://router")

    with pytest.raises(RayRuntimeError, match="invalid training result: dict"):
        runtime.execute_training_job({"rollout_id": 1})


@pytest.mark.unit
@pytest.mark.parametrize(
    "result",
    [
        TrainingJobResult(
            outcome="complete",
            runtime_load_id="v1",
            checkpoint_path="/checkpoint",
        ),
        TrainingJobResult(outcome="stale", runtime_load_id="v1"),
        TrainingJobResult(outcome="storage_blocked", runtime_load_id="v1", storage={"blocked": True}),
    ],
)
def test_training_job_results_round_trip_across_process_boundary(result: TrainingJobResult) -> None:
    assert pickle.loads(pickle.dumps(result)) == result


@pytest.mark.unit
@pytest.mark.parametrize(
    ("batch", "step_preparer", "state", "advantages", "loss"),
    [
        (policy_batch(), "openclawrl", {}, (1.0, 0.5), "openclawrl"),
        (grouped_policy_batch(), _TEST_GROUPED_PREPARER, {}, (-1.0, 1.0), "pg"),
    ],
)
def test_ray_runtime_preserves_backend_prepared_policy_signals(batch, step_preparer, state, advantages, loss) -> None:
    runtime = RayRuntime(train_group_handle=FakeTrainGroupHandle(), inference_url="http://router")

    prepared = runtime.prepare_training_step(batch, step_preparer, state, 9)
    assert prepared.payload is not None
    payload = prepared.payload

    assert payload["loss"] == loss
    assert payload["advantages"] == pytest.approx(advantages)
    assert payload["expected_runtime_load_id"] == "v0"
    assert payload["rollout_id"] == 9


@pytest.mark.unit
@pytest.mark.parametrize(
    ("batch", "step_preparer", "error", "message"),
    [
        (grouped_policy_batch(), "sft", TypeError, "requires PolicyBatch"),
        (
            grouped_policy_batch(),
            _TEST_SFT_PREPARER,
            TypeError,
            "requires PolicyBatch",
        ),
        (policy_batch(), "unknown", ValueError, "unknown step preparer"),
    ],
)
def test_ray_runtime_rejects_unsupported_preparer_batch_pairs(batch, step_preparer, error, message) -> None:
    runtime = RayRuntime(train_group_handle=FakeTrainGroupHandle(), inference_url="http://router")

    with pytest.raises(error, match=message):
        runtime.prepare_training_step(batch, step_preparer, {}, 0)


@pytest.mark.unit
def test_ray_runtime_sends_heterogeneous_samples_to_exact_staleness_admission() -> None:
    class VersionedHandle(FakeTrainGroupHandle):
        def serving_runtime_load_id(self) -> str | None:
            return "engine:1"

    runtime = RayRuntime(train_group_handle=VersionedHandle(), inference_url="http://router")
    prepared = runtime.prepare_training_step(
        grouped_policy_batch(versions=("engine:0", "engine:1")),
        _TEST_GROUPED_PREPARER,
        {},
        0,
    )

    assert prepared.payload is not None
    assert prepared.payload["expected_runtime_load_id"] == "engine:1"
    assert prepared.payload["max_staleness"] == 0
    assert prepared.payload["producing_runtime_load_ids"] == ["engine:0", "engine:1"]


@pytest.mark.unit
def test_ray_runtime_rejects_empty_serving_runtime_load_id() -> None:
    class EmptyRuntimeLoadIdHandle(FakeTrainGroupHandle):
        def serving_runtime_load_id(self) -> str | None:
            return ""

    with pytest.raises(RayRuntimeError, match="non-empty serving runtime load ID"):
        RayRuntime(train_group_handle=EmptyRuntimeLoadIdHandle(), inference_url="http://router")


@pytest.mark.unit
def test_slime_backend_preparation_emits_framework_agnostic_rows() -> None:
    prepared = slime_prepare_step(policy_batch(), "sft", {})
    assert prepared.payload is not None
    payload = prepared.payload

    assert payload["loss"] == "sft"
    assert "advantages" not in payload
    assert payload["samples"] == [
        ["i1", [1, 2], [0, 1], [-0.2, -0.15], 1.0],
        ["i2", [3, 4], [1, 1], [-0.1, -0.05], 0.5],
    ]


@pytest.mark.unit
@pytest.mark.unit
def test_slime_backend_preparation_emits_sao_rows_with_action_mask_and_provenance() -> None:
    # SAO ships an 8-element row: the action mask (for skip-observation GAE)
    # and rollout provenance (producing runtime load ID, creation time) have no
    # slot in the policy 5-tuple. Each sample is its own rollout (no grouping)
    # and advantages are never shipped — the critic computes them in-backend.
    batch = PolicyBatch(
        "math:sao:1",
        (
            PolicySample(
                source_agent_record_id="i1",
                tokens=(5, 1, 2, 3),
                loss_mask=(1, 0, 1),
                action_mask=(1, 0, 1),
                rollout_log_probs=(-0.1, -0.2, -0.3),
                reward=0.9,
                runtime_load_id="slime-v3",
                rollout_created_at=1234.5,
            ),
        ),
    )

    prepared = slime_prepare_step(batch, "sao", {})
    assert prepared.payload is not None
    payload = prepared.payload

    assert payload["loss"] == "sao"
    assert "advantages" not in payload
    assert payload["rollout_ids"] == [0]
    assert payload["samples"] == [
        [
            "i1",
            [5, 1, 2, 3],
            [1, 0, 1],
            [-0.1, -0.2, -0.3],
            0.9,
            [1, 0, 1],
            "slime-v3",
            1234.5,
        ]
    ]


@pytest.mark.unit
def test_slime_backend_preparation_sao_rows_tolerate_missing_provenance() -> None:
    batch = PolicyBatch(
        "math:sao:1",
        (
            PolicySample(
                source_agent_record_id="i1",
                tokens=(5, 1),
                loss_mask=(1,),
                action_mask=(1,),
                rollout_log_probs=(-0.1,),
                reward=0.5,
            ),
        ),
    )

    prepared = slime_prepare_step(batch, "sao", {})
    assert prepared.payload is not None
    payload = prepared.payload

    # A rollout served before runtime-load-ID tracking still ships; the last two
    # slots carry None rather than dropping the row.
    assert payload["samples"][0][-2:] == [None, None]


def sao_batch(runtime_load_id: str | None = "slime-v3") -> PolicyBatch:
    return PolicyBatch(
        "math:sao:1",
        (
            PolicySample(
                source_agent_record_id="i1",
                tokens=(5, 1, 2),
                loss_mask=(1, 1),
                action_mask=(1, 0),
                rollout_log_probs=(-0.1, -0.2),
                reward=0.9,
                runtime_load_id=runtime_load_id,
            ),
        ),
    )


@pytest.mark.unit
def test_ray_runtime_prepares_a_sao_job_from_producing_runtime_load_id() -> None:
    class NoProbeHandle(FakeTrainGroupHandle):
        def __init__(self) -> None:
            self.probes = 0

        def serving_runtime_load_id(self) -> str | None:
            self.probes += 1
            return "engine:0"

    handle = NoProbeHandle()
    runtime = RayRuntime(train_group_handle=handle, inference_url="http://router")
    initialization_probes = handle.probes
    prepared = runtime.prepare_training_step(sao_batch(), "sao", {}, 4)
    assert prepared.payload is not None
    payload = prepared.payload

    assert payload["loss"] == "sao"
    assert payload["rollout_id"] == 4 and payload["expected_runtime_load_id"] == "slime-v3"
    assert "max_staleness" not in payload
    assert handle.probes == initialization_probes


@pytest.mark.unit
def test_ray_runtime_fences_enabled_sao_window_with_serving_version() -> None:
    class VersionedHandle(FakeTrainGroupHandle):
        def __init__(self) -> None:
            self.probes = 0

        def serving_runtime_load_id(self) -> str | None:
            self.probes += 1
            return "engine:3"

    handle = VersionedHandle()
    runtime = RayRuntime(
        train_group_handle=handle,
        inference_url="http://router",
        max_staleness=2,
    )
    initialization_probes = handle.probes

    prepared = runtime.prepare_training_step(sao_batch("engine:1"), "sao", {}, 4)

    assert prepared.payload is not None
    assert prepared.payload["expected_runtime_load_id"] == "engine:3"
    assert prepared.payload["max_staleness"] == 2
    assert prepared.payload["producing_runtime_load_ids"] == ["engine:1"]
    assert prepared.payload["samples"][0][6] == "engine:1"
    assert handle.probes == initialization_probes + 1


@pytest.mark.unit
def test_ray_runtime_preserves_enabled_sao_mixed_provenance() -> None:
    class VersionedHandle(FakeTrainGroupHandle):
        def serving_runtime_load_id(self) -> str | None:
            return "engine:3"

    first = sao_batch("engine:1").samples[0]
    second = PolicySample(
        source_agent_record_id="i2",
        tokens=(6, 1, 2),
        loss_mask=(1, 1),
        action_mask=(1, 0),
        rollout_log_probs=(-0.3, -0.4),
        reward=0.7,
        runtime_load_id="engine:2",
    )
    batch = PolicyBatch("math:sao:2", (first, second))
    runtime = RayRuntime(
        train_group_handle=VersionedHandle(),
        inference_url="http://router",
        max_staleness=2,
    )

    prepared = runtime.prepare_training_step(batch, "sao", {}, 0)

    assert prepared.payload is not None
    assert prepared.payload["producing_runtime_load_ids"] == ["engine:1", "engine:2"]
    assert [row[6] for row in prepared.payload["samples"]] == ["engine:1", "engine:2"]


@pytest.mark.unit
def test_ray_runtime_preserves_enabled_grouped_mixed_provenance() -> None:
    class VersionedHandle(FakeTrainGroupHandle):
        def serving_runtime_load_id(self) -> str | None:
            return "engine:3"

    runtime = RayRuntime(
        train_group_handle=VersionedHandle(),
        inference_url="http://router",
        max_staleness=2,
    )

    prepared = runtime.prepare_training_step(
        grouped_policy_batch(versions=("engine:1", "engine:2")),
        _TEST_GROUPED_PREPARER,
        {},
        0,
    )

    assert prepared.payload is not None
    assert prepared.payload["expected_runtime_load_id"] == "engine:3"
    assert prepared.payload["producing_runtime_load_ids"] == ["engine:1", "engine:2"]


@pytest.mark.unit
def test_ray_runtime_preserves_sao_batch_when_serving_version_is_unverified() -> None:
    runtime = RayRuntime(
        train_group_handle=FakeTrainGroupHandle(),
        inference_url="http://router",
        max_staleness=2,
    )

    with pytest.raises(RayRuntimeError, match="verified serving runtime load ID"):
        runtime.prepare_training_step(sao_batch("engine:1"), "sao", {}, 0)


@pytest.mark.unit
def test_ray_runtime_carries_shared_provenance_for_other_losses() -> None:
    class VersionedHandle(FakeTrainGroupHandle):
        def serving_runtime_load_id(self) -> str | None:
            return "engine:3"

    runtime = RayRuntime(
        train_group_handle=VersionedHandle(),
        inference_url="http://router",
        max_staleness=2,
    )

    prepared = runtime.prepare_training_step(policy_batch(), "sft", {}, 0)

    assert prepared.payload is not None
    assert prepared.payload["expected_runtime_load_id"] == "engine:3"
    assert prepared.payload["max_staleness"] == 2
    assert prepared.payload["producing_runtime_load_ids"] == ["v0", "v0"]


@pytest.mark.unit
def test_ray_runtime_carries_mixed_token_runtime_load_ids_to_bounded_admission() -> None:
    class VersionedHandle(FakeTrainGroupHandle):
        def serving_runtime_load_id(self) -> str | None:
            return "engine:7"

    sample = PolicySample(
        source_agent_record_id="i1",
        tokens=(10, 20, 21, 22),
        loss_mask=(1, 1, 1),
        rollout_log_probs=(-0.1, -0.2, -0.3),
        reward=1.0,
        runtime_load_id=None,
        runtime_load_spans=(
            RuntimeLoadSpan(0, 1, "engine:6"),
            RuntimeLoadSpan(1, 3, "engine:7"),
        ),
    )
    runtime = RayRuntime(
        train_group_handle=VersionedHandle(),
        inference_url="http://router",
        max_staleness=2,
    )

    prepared = runtime.prepare_training_step(PolicyBatch("batch", (sample,)), "sft", {}, 0)

    assert prepared.payload is not None
    assert prepared.payload["producing_runtime_load_ids"] == [None]
    assert prepared.payload["producing_runtime_load_spans"] == [
        [
            {"start": 0, "end": 1, "runtime_load_id": "engine:6"},
            {"start": 1, "end": 3, "runtime_load_id": "engine:7"},
        ]
    ]


@pytest.mark.unit
def test_ray_runtime_sends_mixed_spans_to_exact_admission_instead_of_poisoning_the_batch() -> None:
    class VersionedHandle(FakeTrainGroupHandle):
        def serving_runtime_load_id(self) -> str | None:
            return "engine:7"

    sample = PolicySample(
        source_agent_record_id="i1",
        tokens=(10, 20, 21),
        loss_mask=(1, 1),
        rollout_log_probs=(-0.1, -0.2),
        reward=1.0,
        runtime_load_spans=(
            RuntimeLoadSpan(0, 1, "engine:6"),
            RuntimeLoadSpan(1, 2, "engine:7"),
        ),
    )
    runtime = RayRuntime(train_group_handle=VersionedHandle(), inference_url="http://router")

    prepared = runtime.prepare_training_step(PolicyBatch("batch", (sample,)), "sft", {}, 0)

    assert prepared.payload is not None
    assert prepared.payload["expected_runtime_load_id"] == "engine:7"
    assert prepared.payload["max_staleness"] == 0
    assert prepared.payload["producing_runtime_load_ids"] == [None]


@pytest.mark.unit
def test_ray_runtime_rejects_a_sao_batch_missing_provenance() -> None:
    runtime = RayRuntime(train_group_handle=FakeTrainGroupHandle(), inference_url="http://router")
    batch = PolicyBatch(
        "math:sao:1",
        (
            PolicySample(
                source_agent_record_id="i1",
                tokens=(5, 1),
                loss_mask=(1,),
                action_mask=(1,),
                rollout_log_probs=(-0.1,),
                reward=0.5,
            ),
        ),
    )

    with pytest.raises(RayRuntimeError, match="recorded producing runtime load ID"):
        runtime.prepare_training_step(batch, "sao", {}, 0)


@pytest.mark.unit
def test_enabled_sao_window_sends_missing_provenance_to_bridge_admission() -> None:
    class VersionedHandle(FakeTrainGroupHandle):
        def serving_runtime_load_id(self) -> str | None:
            return "engine:3"

    runtime = RayRuntime(
        train_group_handle=VersionedHandle(),
        inference_url="http://router",
        max_staleness=2,
    )

    prepared = runtime.prepare_training_step(sao_batch(None), "sao", {}, 0)

    assert prepared.payload is not None
    assert prepared.payload["producing_runtime_load_ids"] == [None]
    assert prepared.payload["samples"][0][6] is None
    assert prepared.payload["expected_runtime_load_id"] == "engine:3"


@pytest.mark.unit
def test_remote_handle_delegates_step_preparation_to_the_backend_actor(monkeypatch) -> None:
    expected = PreparedTrainingStep(
        action="skip",
        next_algorithm_state={"steps": 3},
        metrics={"warmup": True},
    )

    class RemoteMethod:
        def __init__(self):
            self.calls = []

        def remote(self, *args):
            self.calls.append(args)
            return expected

    class Bridge:
        prepare_training_step = RemoteMethod()

    class FakeRay:
        @staticmethod
        def get(value, **kwargs):
            return value

    monkeypatch.setattr(ray_runtime, "_require_ray", lambda: FakeRay)
    actor = Bridge()
    handle = RemoteRayTrainGroupHandle(train_group_actor=actor)

    prepared = handle.prepare_training_step(policy_batch(), "openclawrl", {"steps": 2})

    assert prepared is expected
    assert actor.prepare_training_step.calls == [(policy_batch(), "openclawrl", {"steps": 2})]


@pytest.mark.unit
def test_remote_handle_probes_the_named_serving_runtime_load_id_method(monkeypatch) -> None:
    class RemoteMethod:
        def remote(self):
            return "engine-incarnation:3"

    class Bridge:
        serving_runtime_load_id = RemoteMethod()

    class FakeRay:
        calls = []

        @staticmethod
        def get(value, **kwargs):
            FakeRay.calls.append(kwargs)
            return value

    monkeypatch.setattr(ray_runtime, "_require_ray", lambda: FakeRay)

    handle = RemoteRayTrainGroupHandle(train_group_actor=Bridge())

    assert handle.serving_runtime_load_id() == "engine-incarnation:3"
    assert FakeRay.calls == [{"timeout": 300.0}]


@pytest.mark.unit
def test_remote_handle_preserves_missing_serving_runtime_load_id(monkeypatch) -> None:
    class RemoteMethod:
        def remote(self):
            return None

    class Bridge:
        serving_runtime_load_id = RemoteMethod()

    class FakeRay:
        @staticmethod
        def get(value, **kwargs):
            return value

    monkeypatch.setattr(ray_runtime, "_require_ray", lambda: FakeRay)

    assert RemoteRayTrainGroupHandle(train_group_actor=Bridge()).serving_runtime_load_id() is None


@pytest.mark.unit
def test_remote_handle_rejects_an_old_actor_before_training_side_effects() -> None:
    handle = RemoteRayTrainGroupHandle(train_group_actor=object())

    with pytest.raises(RayRuntimeError, match="restart the Reef service and training actor together"):
        handle.health()


@pytest.mark.unit
def test_remote_handle_forwards_durable_training_payload(monkeypatch) -> None:
    class RemoteMethod:
        def __init__(self):
            self.calls = []

        def remote(self, *args):
            self.calls.append(args)
            return TrainingJobResult(outcome="stale", runtime_load_id="v1")

    class Bridge:
        execute_training_job = RemoteMethod()

    class FakeRay:
        @staticmethod
        def get(value, **kwargs):
            return value

    monkeypatch.setattr(ray_runtime, "_require_ray", lambda: FakeRay)

    actor = Bridge()
    handle = RemoteRayTrainGroupHandle(train_group_actor=actor)

    assert handle.execute_training_job({"loss": "pg"}) == TrainingJobResult(outcome="stale", runtime_load_id="v1")
    assert actor.execute_training_job.calls == [({"loss": "pg"},)]


@pytest.mark.unit
def test_inference_admission_controller_blocks_cancellation_without_leaking_a_handle() -> None:
    async def run() -> None:
        controller = InferenceAdmissionController()
        admitted = await controller.acquire()
        controller.close()
        queued = asyncio.create_task(controller.acquire())
        await asyncio.sleep(0)
        assert not queued.done()
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        assert controller.status == {"open": False, "active": 1}
        admitted.release()
        controller.open()
        next_handle = await asyncio.wait_for(controller.acquire(), 1)
        next_handle.release()
        assert controller.status == {"open": True, "active": 0}

    asyncio.run(run())


@pytest.mark.unit
def test_noncolocated_weight_update_preserves_inflight_and_queues_new_requests_until_commit() -> None:
    handle = DeferredWeightUpdateTrainGroupHandle()
    runtime = RayRuntime(train_group_handle=handle, inference_url="http://router")
    inflight = asyncio.run(runtime.acquire_inference())
    assert runtime.current_runtime_load_id() == "engine:0"

    result = runtime.execute_training_job({"rollout_id": 0})

    assert result.training_job_id == "job-0"
    assert handle.calls == ["execute", "update_weights"]
    assert runtime.serving_runtime_load_id() == "engine:1"
    assert runtime.current_runtime_load_id() == "engine:0"
    assert runtime.inference_admission_status == {"open": False, "active": 1}

    async def commit_and_admit() -> None:
        queued = asyncio.create_task(runtime.acquire_inference())
        await asyncio.sleep(0)
        assert not queued.done()
        runtime.reconcile_training_job(
            scenario_step=1,
            committed_training_job_id=result.training_job_id,
        )
        admitted = await asyncio.wait_for(queued, 1)
        admitted.release()

    asyncio.run(commit_and_admit())
    inflight.release()
    assert handle.calls == ["execute", "update_weights", "acknowledge"]
    assert runtime.current_runtime_load_id() == "engine:1"
    assert runtime.inference_admission_status == {"open": True, "active": 0}


@pytest.mark.unit
def test_candidate_rejection_leaves_serving_weights_unchanged() -> None:
    handle = DeferredWeightUpdateTrainGroupHandle()
    runtime = RayRuntime(train_group_handle=handle, inference_url="http://router")

    candidate = runtime.train_candidate({"rollout_id": 0})
    evaluation = EvaluationResult("test", "1", {})
    runtime.reject_candidate(
        candidate,
        SelectionDecision("reject", "test", "1", "rejected by test", evaluation),
    )

    assert handle.calls == ["execute", "reject"]
    assert runtime.serving_runtime_load_id() == "engine:0"
    assert runtime.current_runtime_load_id() == "engine:0"
    assert runtime.inference_admission_status["open"] is True


@pytest.mark.unit
def test_colocated_weight_update_retracts_without_draining_inflight() -> None:
    handle = DeferredWeightUpdateTrainGroupHandle(colocate=True)
    runtime = RayRuntime(train_group_handle=handle, inference_url="http://router")
    inflight = asyncio.run(runtime.acquire_inference())
    assert runtime.current_runtime_load_id() == "engine:0"

    result = runtime.execute_training_job({"rollout_id": 0})

    assert handle.calls == ["execute", "update_weights"]
    assert runtime.inference_admission_status == {"open": False, "active": 1}
    assert runtime.serving_runtime_load_id() == "engine:1"
    assert runtime.current_runtime_load_id() == "engine:0"
    runtime.reconcile_training_job(scenario_step=1, committed_training_job_id=result.training_job_id)
    assert runtime.inference_admission_status == {"open": True, "active": 1}
    assert runtime.current_runtime_load_id() == "engine:1"
    inflight.release()
    assert runtime.inference_admission_status == {"open": True, "active": 0}


@pytest.mark.unit
def test_colocated_weight_update_does_not_wait_for_inference_timeout() -> None:
    handle = DeferredWeightUpdateTrainGroupHandle(colocate=True)
    runtime = RayRuntime(
        train_group_handle=handle,
        inference_url="http://router",
        inference_timeout_s=0.01,
    )
    inflight = asyncio.run(runtime.acquire_inference())

    result = runtime.execute_training_job({"rollout_id": 0})

    assert result.training_job_id == handle.training_job_id
    assert handle.calls == ["execute", "update_weights"]
    assert runtime.inference_admission_status == {"open": False, "active": 1}
    runtime.reconcile_training_job(scenario_step=1, committed_training_job_id=result.training_job_id)
    inflight.release()


@pytest.mark.unit
def test_colocated_checkpoint_rejection_reopens_admission_when_backend_stayed_idle() -> None:
    class RejectingHandle(DeferredWeightUpdateTrainGroupHandle):
        def execute_training_job(self, payload: Mapping[str, Any]) -> TrainingJobResult:
            del payload
            self.calls.append("execute")
            raise RuntimeError("checkpoint rejected")

    handle = RejectingHandle(colocate=True)
    runtime = RayRuntime(train_group_handle=handle, inference_url="http://router")

    with pytest.raises(RuntimeError, match="checkpoint rejected"):
        runtime.execute_training_job({"rollout_id": 0})

    assert handle.calls == ["execute"]
    assert runtime.inference_admission_status == {"open": True, "active": 0}


@pytest.mark.unit
def test_running_training_closes_only_colocated_inference_admission() -> None:
    disjoint = RayRuntime(
        train_group_handle=DeferredWeightUpdateTrainGroupHandle(status="RUNNING"),
        inference_url="http://router",
    )
    colocated = RayRuntime(
        train_group_handle=DeferredWeightUpdateTrainGroupHandle(status="RUNNING", colocate=True),
        inference_url="http://router",
    )

    disjoint.reconcile_training_job(scenario_step=0)
    colocated.reconcile_training_job(scenario_step=0)

    assert disjoint.inference_admission_status == {"open": True, "active": 0}
    assert colocated.inference_admission_status == {"open": False, "active": 0}
    assert disjoint.current_runtime_load_id() == "engine:0"
    assert colocated.current_runtime_load_id() is None


@pytest.mark.unit
def test_ray_runtime_rejects_an_unhealthy_training_group() -> None:
    class UnhealthyHandle(DeferredWeightUpdateTrainGroupHandle):
        def health(self) -> Mapping[str, Any]:
            return {
                **super().health(),
                "ok": False,
                "phase": "training_failed",
            }

    with pytest.raises(RayRuntimeError, match=r"unhealthy.*training_failed"):
        RayRuntime(train_group_handle=UnhealthyHandle(), inference_url="http://router")


@pytest.mark.unit
def test_recovery_retries_a_failed_publication_instead_of_declaring_the_group_dead() -> None:
    # A wedged rollout engine fails the publication fan-out: the bridge
    # terminates its engines and reports itself unhealthy but recoverable —
    # the durable UPDATING_WEIGHTS marker keeps the publication replayable.
    class FailedPublicationHandle(DeferredWeightUpdateTrainGroupHandle):
        def health(self) -> Mapping[str, Any]:
            health = dict(super().health())
            if self.status == "UPDATING_WEIGHTS":
                health.update(ok=False, phase="weight_sync_failed", recoverable=True)
            return health

    handle = FailedPublicationHandle(status="UPDATING_WEIGHTS")
    runtime = RayRuntime(train_group_handle=handle, inference_url="http://router")

    runtime.reconcile_training_job(scenario_step=0)

    assert handle.calls == ["update_weights"]
    assert handle.status == "READY_TO_COMMIT"


@pytest.mark.unit
def test_a_failure_the_group_does_not_call_recoverable_is_still_unhealthy() -> None:
    # An older bridge (no recoverable field) degrades to the strict behavior.
    class DeadPublicationHandle(DeferredWeightUpdateTrainGroupHandle):
        def health(self) -> Mapping[str, Any]:
            return {**super().health(), "ok": False, "phase": "weight_sync_failed"}

    with pytest.raises(RayRuntimeError, match=r"unhealthy.*weight_sync_failed"):
        RayRuntime(train_group_handle=DeadPublicationHandle(status="COMPLETE"), inference_url="http://router")


@pytest.mark.unit
def test_colocated_completed_checkpoint_replay_reopens_admission() -> None:
    class CompletedReplay(DeferredWeightUpdateTrainGroupHandle):
        def execute_training_job(self, payload: Mapping[str, Any]) -> TrainingJobResult:
            del payload
            self.calls.append("execute")
            self.status = "COMPLETE"
            return TrainingJobResult(
                outcome="complete",
                runtime_load_id="engine:1",
                checkpoint_path="/checkpoint",
            )

    handle = CompletedReplay(colocate=True)
    runtime = RayRuntime(train_group_handle=handle, inference_url="http://router")

    result = runtime.execute_training_job({"rollout_id": 0})

    assert result.outcome == "complete"
    assert handle.calls == ["execute"]
    assert runtime.inference_admission_status == {"open": True, "active": 0}


@pytest.mark.unit
def test_recovery_acknowledges_a_training_job_after_the_scenario_commit() -> None:
    handle = DeferredWeightUpdateTrainGroupHandle(status="READY_TO_COMMIT", rollout_id=3)
    runtime = RayRuntime(train_group_handle=handle, inference_url="http://router")

    runtime.reconcile_training_job(
        scenario_step=4,
        committed_training_job_id=handle.training_job_id,
    )

    assert handle.calls == ["acknowledge"]
    assert runtime.inference_admission_status == {"open": True, "active": 0}


@pytest.mark.unit
def test_recovery_does_not_acknowledge_an_unrelated_later_commit() -> None:
    handle = DeferredWeightUpdateTrainGroupHandle(status="READY_TO_COMMIT", rollout_id=3)
    runtime = RayRuntime(train_group_handle=handle, inference_url="http://router")

    runtime.reconcile_training_job(
        scenario_step=5,
        committed_training_job_id="rollback-job",
    )

    assert handle.calls == []
    assert runtime.inference_admission_status == {"open": False, "active": 0}


@pytest.mark.unit
def test_legacy_complete_marker_stays_closed_until_reef_commit_is_proven() -> None:
    class LegacyCompleteHandle(DeferredWeightUpdateTrainGroupHandle):
        def health(self) -> Mapping[str, Any]:
            health = dict(super().health())
            health["training_job"] = {**health["training_job"], "commit_acknowledged": False}
            return health

    handle = LegacyCompleteHandle(status="COMPLETE", rollout_id=3)
    runtime = RayRuntime(train_group_handle=handle, inference_url="http://router")

    runtime.reconcile_training_job(scenario_step=3)
    assert handle.calls == []
    assert runtime.inference_admission_status == {"open": False, "active": 0}

    runtime.reconcile_training_job(scenario_step=4)
    assert handle.calls == []
    assert runtime.inference_admission_status == {"open": False, "active": 0}

    runtime.reconcile_training_job(
        scenario_step=4,
        committed_training_without_job_id=True,
    )
    assert handle.calls == ["acknowledge"]
    assert runtime.inference_admission_status == {"open": True, "active": 0}


@pytest.mark.unit
def test_legacy_complete_marker_does_not_trust_an_unrelated_later_training_commit() -> None:
    class LegacyCompleteHandle(DeferredWeightUpdateTrainGroupHandle):
        def health(self) -> Mapping[str, Any]:
            health = dict(super().health())
            health["training_job"] = {**health["training_job"], "commit_acknowledged": False}
            return health

    handle = LegacyCompleteHandle(status="COMPLETE", rollout_id=3)
    runtime = RayRuntime(train_group_handle=handle, inference_url="http://router")

    runtime.reconcile_training_job(
        scenario_step=5,
        committed_training_without_job_id=True,
    )

    assert handle.calls == []
    assert runtime.inference_admission_status == {"open": False, "active": 0}


@pytest.mark.unit
def test_recovery_republishes_an_uncertain_partial_weight_update_before_commit() -> None:
    handle = DeferredWeightUpdateTrainGroupHandle(status="UPDATING_WEIGHTS", rollout_id=3)
    runtime = RayRuntime(train_group_handle=handle, inference_url="http://router")

    runtime.reconcile_training_job(scenario_step=3)

    assert handle.calls == ["update_weights"]
    assert handle.status == "READY_TO_COMMIT"
    assert runtime.inference_admission_status == {"open": False, "active": 0}


@pytest.mark.unit
def test_queued_tttd_fanout_freezes_the_head_that_reopens_admission(monkeypatch) -> None:
    async def call_inline(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", call_inline)
    handle = DeferredWeightUpdateTrainGroupHandle()
    runtime = RayRuntime(train_group_handle=handle, inference_url="http://router")
    updated = runtime.execute_training_job({"rollout_id": 0})

    class Scenario:
        recipe = "test"
        repository = None
        surface = create_weight_surface()
        inference_backend = None

        def __init__(self) -> None:
            self.runtime = runtime
            self.ref = LiveWeightArtifactRef(
                content_id="live:old",
                release_id="live:proc:engine:0:0",
                parent_release_id=None,
                runtime_load_id="engine:0",
            )

        def current_artifact_ref(self):
            return self.ref

    class Dispatcher:
        def __init__(self, scenario) -> None:
            self.current = scenario

        def get_or_create_scenario(self, *args, **kwargs):
            del args, kwargs
            return self.current

        @staticmethod
        def accept_record(item, **kwargs):
            del kwargs
            return item

    class RecordingBackend(InferenceBackend):
        def __init__(self) -> None:
            self.versions: list[str] = []

        async def inference(self, artifact, path, payload):
            del path, payload
            self.versions.append(artifact.ref.runtime_load_id)
            return {"metadata": {"runtime_load_id": artifact.ref.runtime_load_id}}

    scenario = Scenario()
    backend = RecordingBackend()
    service = RequestService(Dispatcher(scenario))

    async def run() -> None:
        queued = [
            asyncio.create_task(
                service.infer(
                    {"x-reef-scenario": "math"},
                    {"messages": [{"role": "user", "content": f"rollout {rollout}"}]},
                    "/v1/chat/completions",
                    backend,
                )
            )
            for rollout in range(8 * 64)
        ]
        await asyncio.sleep(0)
        assert backend.versions == []
        scenario.ref = LiveWeightArtifactRef(
            content_id="live:new",
            release_id="live:proc:engine:1:1",
            parent_release_id=scenario.ref.release_id,
            runtime_load_id="engine:1",
        )
        runtime.reconcile_training_job(
            scenario_step=1,
            committed_training_job_id=updated.training_job_id,
        )
        responses = await asyncio.wait_for(asyncio.gather(*queued), 5)
        assert {response["metadata"]["runtime_load_id"] for response in responses} == {"engine:1"}

    asyncio.run(run())
    assert backend.versions == ["engine:1"] * (8 * 64)


@pytest.mark.unit
def test_stream_and_failure_release_their_inference_admission_handles() -> None:
    runtime = RayRuntime(train_group_handle=DeferredWeightUpdateTrainGroupHandle(), inference_url="http://router")
    ref = LiveWeightArtifactRef(
        content_id="live:current",
        release_id="live:proc:engine:0:0",
        parent_release_id=None,
        runtime_load_id="engine:0",
    )

    class Scenario:
        recipe = "test"
        repository = None
        surface = create_weight_surface()
        inference_backend = None

        @staticmethod
        def current_artifact_ref():
            return ref

        def __init__(self) -> None:
            self.runtime = runtime

    class Dispatcher:
        @staticmethod
        def get_or_create_scenario(*args, **kwargs):
            del args, kwargs
            return Scenario()

        @staticmethod
        def accept_record(item, **kwargs):
            del kwargs
            return item

    class StreamingBackend(InferenceBackend):
        async def inference(self, artifact, path, payload):
            raise AssertionError("streaming path expected")

        async def inference_stream(self, artifact, path, payload):
            del artifact, path, payload

            async def chunks():
                yield b"data: [DONE]\n\n"

            return InferenceStream(
                status=200,
                headers={"Content-Type": "text/event-stream"},
                chunks=chunks(),
                record_response={"metadata": {"runtime_load_id": "engine:0"}},
            )

    class FailingBackend(InferenceBackend):
        async def inference(self, artifact, path, payload):
            del artifact, path, payload
            raise RuntimeError("backend failed")

    class DeferredStreamingBackend(InferenceBackend):
        async def inference(self, artifact, path, payload):
            raise AssertionError("streaming path expected")

        async def inference_stream(self, artifact, path, payload):
            del artifact, path, payload
            holder = {}

            async def chunks():
                yield b'data: {"content":"first token"}\n\n'
                holder["stream"].record_response = {
                    "metadata": {"runtime_load_id": "engine:0"},
                    "training": {"runtime_load_id": "engine:0"},
                }
                yield b"data: [DONE]\n\n"

            stream = InferenceStream(
                status=200,
                headers={"Content-Type": "text/event-stream"},
                chunks=chunks(),
                record_response_pending=True,
            )
            holder["stream"] = stream
            return stream

    class UnverifiableStreamingBackend(InferenceBackend):
        async def inference(self, artifact, path, payload):
            raise AssertionError("streaming path expected")

        async def inference_stream(self, artifact, path, payload):
            del artifact, path, payload

            async def chunks():
                yield b"data: [DONE]\n\n"

            return InferenceStream(
                status=200,
                headers={"Content-Type": "text/event-stream"},
                chunks=chunks(),
            )

    service = RequestService(Dispatcher())

    async def run() -> None:
        stream, pending = await service.start_stream(
            {"x-reef-scenario": "math"},
            {"stream": True},
            "/v1/chat/completions",
            StreamingBackend(),
        )
        assert runtime.inference_admission_status == {"open": True, "active": 0}
        assert pending.admission is None
        await stream.close()
        service.record_stream(pending, {"stream": True, "complete": True})
        assert runtime.inference_admission_status == {"open": True, "active": 0}

        deferred, deferred_pending = await service.start_stream(
            {"x-reef-scenario": "math"},
            {"stream": True},
            "/v1/chat/completions",
            DeferredStreamingBackend(),
        )
        assert runtime.inference_admission_status == {"open": True, "active": 1}
        body = b"".join([chunk async for chunk in deferred.chunks])
        recorded = stream_record(deferred, body, complete=True)
        item = service.record_stream(deferred_pending, recorded)
        assert item.payload["runtime_load_id"] == "engine:0"
        assert runtime.inference_admission_status == {"open": True, "active": 0}

        with pytest.raises(RuntimeLoadMismatch, match="atomic record_response"):
            await service.start_stream(
                {"x-reef-scenario": "math"},
                {"stream": True},
                "/v1/chat/completions",
                UnverifiableStreamingBackend(),
            )
        assert runtime.inference_admission_status == {"open": True, "active": 0}

        with pytest.raises(RuntimeError, match="backend failed"):
            await service.infer(
                {"x-reef-scenario": "math"},
                {},
                "/v1/chat/completions",
                FailingBackend(),
            )
        assert runtime.inference_admission_status == {"open": True, "active": 0}

    asyncio.run(run())


def _two_epoch_sample_preparer(batch, state):
    """A cookbook-shaped preparer: per-sample rollouts, two passes over the batch."""
    del state
    return StepSignal(
        action="train",
        loss_family="pg",
        next_algorithm_state={},
        advantages=(-1.0, 1.0),
        scheduling=StepScheduling(unit="sample", batch_size="actual", epochs=2),
    )


@pytest.mark.unit
def test_ray_runtime_provenance_follows_the_step_schedule() -> None:
    # Epochs repeat every wire row; the producing runtime load IDs (and spans)
    # must repeat with them, in the schedule's order, or bounded-staleness
    # admission counts one version per batch row against two rows per sample.
    class VersionedHandle(FakeTrainGroupHandle):
        def serving_runtime_load_id(self) -> str | None:
            return "engine:1"

    runtime = RayRuntime(train_group_handle=VersionedHandle(), inference_url="http://router")
    prepared = runtime.prepare_training_step(
        grouped_policy_batch(versions=("engine:0", "engine:1")),
        "reef_service.test_ray_runtime:_two_epoch_sample_preparer",
        {},
        0,
    )

    assert prepared.payload is not None
    assert [row[0] for row in prepared.payload["samples"]] == ["g1", "g2", "g1", "g2"]
    assert prepared.payload["producing_runtime_load_ids"] == ["engine:0", "engine:1", "engine:0", "engine:1"]
    assert "source_rows" not in prepared.payload


@pytest.mark.unit
def test_runtime_takes_inference_url_from_the_training_actor_when_unset() -> None:
    class ReportingHandle(FakeTrainGroupHandle):
        def health(self) -> Mapping[str, Any]:
            return {**super().health(), "inference_url": "http://10.0.0.7:30000/"}

    runtime = RayRuntime(train_group_handle=ReportingHandle())

    assert runtime.base_url == "http://10.0.0.7:30000"


@pytest.mark.unit
def test_runtime_configured_inference_url_wins_over_the_reported_one() -> None:
    class ReportingHandle(FakeTrainGroupHandle):
        def health(self) -> Mapping[str, Any]:
            return {**super().health(), "inference_url": "http://10.0.0.7:30000"}

    runtime = RayRuntime(train_group_handle=ReportingHandle(), inference_url="http://router")

    assert runtime.base_url == "http://router"


@pytest.mark.unit
def test_runtime_without_any_inference_url_fails_clearly() -> None:
    with pytest.raises(RayRuntimeError, match="inference_url is unset"):
        RayRuntime(train_group_handle=FakeTrainGroupHandle())
