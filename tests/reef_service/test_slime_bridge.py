"""CPU contract tests for the Reef ↔ Slime Ray bridge.

These tests intentionally exercise the plain implementation class rather than
starting a Ray actor or a Megatron worker.  The bridge's wire-format and
object-store hand-off are deterministic Python operations; GPU/Ray-cluster
tests belong to a separate deployment suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from slime.utils.misc import Box

from reef.runtime.base import TrainingJobResult
from reef.train.algos import StepScheduling
from reef.train.slime_backend.data_builder import to_slime_rollout_data
from reef.train.slime_backend.loss_families import resolve_loss_family
from reef.train.slime_backend.reef_adapters import bridge
from reef.train.slime_backend.reef_adapters.preparation import _build_payload
from reef.train.slime_backend.reef_adapters.training_job.marker import read_marker, transition_marker, write_marker
from reef.train.slime_backend.reef_adapters.training_job.storage import RetentionConfig, _allocated_bytes
from reef.train.types import GroupedPolicyBatch, PolicyBatch, PolicySample


@pytest.fixture(autouse=True)
def _local_ray_get(monkeypatch):
    monkeypatch.setattr(bridge.ray, "get", lambda value, **kwargs: value)


def _row(
    source_id: str,
    *,
    tokens: tuple[int, ...] = (10, 11, 12),
    loss_mask: tuple[int, ...] = (1, 1),
    log_probs: tuple[float, ...] = (-0.1, -0.2),
    reward: float = 0.5,
) -> list[object]:
    """Build one framework-agnostic row emitted by Reef's remote handle."""

    return [source_id, list(tokens), list(loss_mask), list(log_probs), reward]


def _payload(
    *,
    log_probs: object = "default",
    advantages: object = "default",
    loss: str = "pg",
) -> dict:
    if log_probs == "default":
        rows = [_row("a"), _row("b", reward=1.0), _row("c", reward=-1.0)]
    else:
        rows = [
            _row("a", log_probs=tuple(log_probs[0])),
            _row("b", log_probs=tuple(log_probs[1]), reward=1.0),
            _row("c", log_probs=tuple(log_probs[2]), reward=-1.0),
        ]
    data: dict[str, object] = {
        "samples": rows,
        # Two samples from one comparison set and one from another.  IDs are
        # deliberately explicit: the bridge must not derive them from a hash
        # or from the order in which a dict happens to be traversed.
        "rollout_ids": [0, 0, 1],
        "loss": loss,
    }
    if advantages != "default":
        data["advantages"] = advantages
    return data


def _execute_and_update_weights(
    actor: bridge.TrainBridgeActorImpl,
    payload: dict,
) -> TrainingJobResult:
    result = actor.execute_training_job(payload)
    if result.outcome != "checkpoint":
        return result
    assert result.training_job_id is not None
    return actor.update_serving_weights(result.training_job_id)


@pytest.mark.unit
def test_to_slime_rollout_data_converts_non_empty_payload_and_preserves_groups() -> None:
    converted = to_slime_rollout_data(
        {
            "samples": [_row("first"), _row("second", reward=1.0)],
            "rollout_ids": [7, 7],
            "loss": "pg",
            "advantages": [0.25, -0.25],
        }
    )

    assert converted["tokens"] == [[10, 11, 12], [10, 11, 12]]
    assert converted["loss_masks"] == [[1, 1], [1, 1]]
    assert converted["response_lengths"] == [2, 2]
    assert converted["rewards"] == [0.5, 1.0]
    assert converted["rollout_ids"] == [7, 7]
    assert converted["sample_indices"] == [0, 1]
    assert converted["loss"] == "pg"
    # Reef supplies one scalar advantage per sample; Slime consumes one value
    # per response token.
    assert converted["advantages"] == [[0.25, 0.25], [-0.25, -0.25]]


@pytest.mark.unit
def test_to_slime_rollout_data_aggregates_rollout_mask_sums() -> None:
    payload = {
        "samples": [
            _row("a", loss_mask=(1, 1)),
            _row("b", loss_mask=(1,), log_probs=(-0.1,)),
            _row(
                "c",
                tokens=(10, 11, 12, 13, 14),
                loss_mask=(1, 1, 1, 1),
                log_probs=(-0.1, -0.2, -0.3, -0.4),
            ),
        ],
        "rollout_ids": [0, 0, 1],
        "loss": "pg",
        "advantages": [0.25, -0.25, 0.0],
    }

    converted = to_slime_rollout_data(payload)

    # IDs [0, 0, 1], per-sample mask totals [2, 1, 4] => aggregate and
    # broadcast [3, 3, 4].
    assert converted["rollout_mask_sums"] == [3, 3, 4]


@pytest.mark.unit
def test_to_slime_rollout_data_omits_optional_empty_log_probs() -> None:
    converted = to_slime_rollout_data(_payload(log_probs=[[], [], []], loss="sft"))

    assert "rollout_log_probs" not in converted


@pytest.mark.unit
def test_to_slime_rollout_data_keeps_non_empty_log_probs() -> None:
    converted = to_slime_rollout_data(
        _payload(
            log_probs=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            loss="sft",
        )
    )

    assert converted["rollout_log_probs"] == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]


@pytest.mark.unit
def test_to_slime_rollout_data_rejects_non_empty_log_prob_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="log_prob"):
        to_slime_rollout_data(
            _payload(
                log_probs=[[0.1], [0.3, 0.4], [0.5, 0.6]],
                loss="sft",
            )
        )


@pytest.mark.unit
def test_to_slime_rollout_data_rejects_mixed_empty_and_non_empty_log_probs() -> None:
    with pytest.raises(ValueError, match="log_prob"):
        to_slime_rollout_data(
            _payload(
                log_probs=[[], [0.3, 0.4], [0.5, 0.6]],
                loss="sft",
            )
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "payload, message",
    [
        ({"samples": [], "rollout_ids": [], "loss": "pg"}, "non-empty"),
        ({"samples": [_row("a")], "rollout_ids": [], "loss": "pg"}, "rollout_ids"),
        (
            {"samples": [_row("a")], "rollout_ids": [0], "loss": "pg", "advantages": []},
            "advantages",
        ),
        (
            {
                "samples": [
                    ["a", [1], [1]],
                ],
                "rollout_ids": [0],
                "loss": "pg",
            },
            "sample",
        ),
    ],
)
def test_to_slime_rollout_data_validates_non_empty_and_parallel_shapes(payload, message) -> None:
    with pytest.raises(ValueError, match=message):
        to_slime_rollout_data(payload)


@pytest.mark.unit
def test_slime_preparation_emits_stable_rollout_ids_for_comparison_sets() -> None:
    batch = GroupedPolicyBatch(
        "batch",
        (
            (
                PolicySample("a", (1, 2), (1, 1), (-0.1, -0.2), 0.2),
                PolicySample("b", (3, 4), (1, 1), (-0.3, -0.4), 0.8),
            ),
            (PolicySample("c", (5, 6), (1, 1), (-0.5, -0.6), 0.4),),
        ),
    )

    payload = _build_payload(batch, "pg", (0.1, -0.1, 0.0), StepScheduling())

    assert payload["rollout_ids"] == [0, 0, 1]
    assert len(payload["rollout_ids"]) == len(payload["samples"]) == 3


@pytest.mark.unit
def test_slime_preparation_honors_sample_scheduling_and_actual_batch_size() -> None:
    batch = GroupedPolicyBatch(
        "batch",
        (
            (
                PolicySample("a", (1, 2), (1,), (-0.1,), 0.2),
                PolicySample("b", (3, 4), (1,), (-0.2,), 0.8),
            ),
        ),
    )

    payload = _build_payload(
        batch,
        "tttd",
        (0.1, -0.1),
        StepScheduling(unit="sample", batch_size="actual"),
    )

    assert payload["rollout_ids"] == [0, 1]
    assert payload["external_step_sizes"] == [2]
    assert "external_remainder" not in payload


@pytest.mark.unit
def test_slime_preparation_assigns_distinct_stable_ids_to_policy_samples() -> None:
    batch = PolicyBatch(
        "batch",
        (
            PolicySample("a", (1,), (1,), (-0.1,), 0.2),
            PolicySample("b", (2,), (1,), (-0.2,), 0.8),
        ),
    )

    payload = _build_payload(batch, "sft", None, StepScheduling())

    assert payload["rollout_ids"] == [0, 1]


@pytest.mark.unit
def test_assembled_multi_turn_sample_uses_existing_slime_payload_path() -> None:
    sample = PolicySample(
        "harbor-report",
        (10, 20, 11, 21),
        (1, 0, 1),
        (-0.1, 0.0, -0.2),
        0.75,
        "wv-1",
        turn_count=2,
    )
    assert sample.is_multi_turn
    payload = _build_payload(PolicyBatch("batch", (sample,)), "sft", None, StepScheduling())

    assert payload["samples"] == [["harbor-report", [10, 20, 11, 21], [1, 0, 1], [-0.1, 0.0, -0.2], 0.75]]
    converted = to_slime_rollout_data(payload)

    assert converted["tokens"] == [[10, 20, 11, 21]]
    assert converted["loss_masks"] == [[1, 0, 1]]
    assert converted["rollout_log_probs"] == [[-0.1, 0.0, -0.2]]
    assert converted["rewards"] == [0.75]


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _FakeRolloutManager:
    def __init__(self, packed, timeline: list[str] | None = None):
        self.packed = packed
        self.calls: list[dict] = []
        self.version = "v1"
        self.versions: list[str] | None = None
        # ``timeline`` is the shared event log a group double can also write to,
        # so a test that claims one side happens before the other compares them
        # in a single ordered sequence instead of in two lists that cannot be
        # interleaved. ``lifecycle_calls`` stays the rollout only view.
        self.lifecycle_calls: list[str] = []
        self.timeline = timeline if timeline is not None else []
        self.prepare_external_train_data = _RemoteMethod(self._prepare)
        self.inference_url = _RemoteMethod(lambda: "http://10.0.0.7:30000")
        self.get_runtime_load_ids = _RemoteMethod(
            lambda: self.versions if self.versions is not None else [self.version]
        )
        self.terminate_updatable_engines = _RemoteMethod(lambda: 1)
        self.recover_updatable_engines = _RemoteMethod(lambda: self._record("recover_engines"))
        self.pause_generation_for_update = _RemoteMethod(lambda: self._record("pause_generation"))
        self.continue_generation_after_update = _RemoteMethod(lambda: self._record("continue_generation"))
        self.offload = _RemoteMethod(lambda: self._record("offload"))
        self.onload_weights = _RemoteMethod(lambda: self._record("onload_weights"))
        self.onload_kv = _RemoteMethod(lambda: self._record("onload_kv"))

    def _record(self, event: str) -> None:
        self.lifecycle_calls.append(event)
        self.timeline.append(event)

    def _prepare(self, data):
        self.calls.append(data)
        return "packed-ref"


class _FakeRank:
    def __init__(self, version="v1", metrics=None):
        self.version = version
        self.metrics = {} if metrics is None else metrics
        self.get_runtime_load_id = _RemoteMethod(lambda: self.version)
        self.pop_metrics = _RemoteMethod(lambda: self.metrics)


class _FakeGroup:
    def __init__(self, timeline: list[str] | None = None):
        self.train_calls: list[tuple[int, object]] = []
        self.update_calls = 0
        self.update_generation_management: list[bool] = []
        self.update_force_full: list[bool] = []
        self.save_calls: list[tuple[int, bool]] = []
        self.republication_calls: list[str] = []
        self.timeline = timeline if timeline is not None else []
        self._actor_handlers = [_FakeRank()]

    def async_train(self, rollout_id, rollout_data_ref, external_data=None):
        self.train_calls.append((rollout_id, rollout_data_ref))
        return ["train-ref"]

    def async_pop_rank0_metrics(self):
        return self._actor_handlers[0].pop_metrics.remote()

    def async_get_rank0_runtime_load_id(self):
        return self._actor_handlers[0].get_runtime_load_id.remote()

    def update_weights(self, *, manage_generation: bool = True, force_full: bool = False):
        self.update_calls += 1
        self.update_generation_management.append(manage_generation)
        self.update_force_full.append(force_full)

    def restore_runtime_load_id_for_republication(self, runtime_load_id):
        self.republication_calls.append(runtime_load_id)

    def save_model(self, rollout_id, force_sync=False):
        self.save_calls.append((rollout_id, force_sync))
        self.timeline.append("save_model")


class _DurableGroup(_FakeGroup):
    def __init__(self, template: str, megatron_root: Path | None = None, timeline: list[str] | None = None):
        super().__init__(timeline)
        self.template = template
        self.megatron_root = megatron_root

    def save_model(self, rollout_id, force_sync=False):
        super().save_model(rollout_id, force_sync)
        checkpoint = Path(self.template.format(rollout_id=rollout_id))
        checkpoint.mkdir(parents=True)
        (checkpoint / "weights").write_text("hf", encoding="utf-8")
        if self.megatron_root is not None:
            iteration = self.megatron_root / f"iter_{rollout_id:07d}"
            iteration.mkdir(parents=True)
            (iteration / "state").write_text("megatron", encoding="utf-8")
            (self.megatron_root / "latest_checkpointed_iteration.txt").write_text(str(rollout_id), encoding="utf-8")


def _durable_actor(tmp_path):
    template = str(tmp_path / "checkpoint-{rollout_id}")
    group = _DurableGroup(template)
    actor = bridge.TrainBridgeActorImpl(group, _FakeRolloutManager(["packed"]), save_hf_template=template)
    payload = _payload(loss="sft")
    payload.update(rollout_id=0, expected_runtime_load_id="v1", parent_release_id="parent-0")
    return actor, group, payload


def _sao_durable_actor(
    tmp_path: Path,
    *,
    producing_versions: tuple[str | None, ...] = ("engine:1",),
    serving_version: str = "engine:3",
    max_staleness: int = 2,
):
    template = str(tmp_path / "checkpoint-{rollout_id}")
    group = _DurableGroup(template)
    group._actor_handlers[0].version = serving_version
    manager = _FakeRolloutManager(["packed"])
    manager.version = serving_version
    actor = bridge.TrainBridgeActorImpl(
        group,
        manager,
        save_hf_template=template,
        loss_runtime=resolve_loss_family("sao").bind(),
        critic_group=_FakeGroup(),
    )
    rows = [
        [
            f"source-{index}",
            [10, 11, 12],
            [1, 1],
            [-0.1, -0.2],
            0.5,
            [1, 1],
            producing_version,
            1234.5,
        ]
        for index, producing_version in enumerate(producing_versions)
    ]
    payload = {
        "samples": rows,
        "rollout_ids": list(range(len(rows))),
        "loss": "sao",
        "rollout_id": 0,
        "expected_runtime_load_id": serving_version,
        "max_staleness": max_staleness,
        "producing_runtime_load_ids": list(producing_versions),
    }
    return actor, group, manager, payload


def _loss_family_durable_actor(
    tmp_path: Path,
    loss_family: str,
    *,
    producing_version: str = "engine:1",
    serving_version: str = "engine:2",
):
    if loss_family == "sao":
        return _sao_durable_actor(
            tmp_path,
            producing_versions=(producing_version,),
            serving_version=serving_version,
        )
    template = str(tmp_path / "checkpoint-{rollout_id}")
    group = _DurableGroup(template)
    group._actor_handlers[0].version = serving_version
    manager = _FakeRolloutManager(["packed"])
    manager.version = serving_version
    loss_runtime = resolve_loss_family(loss_family).bind()
    actor = bridge.TrainBridgeActorImpl(
        group,
        manager,
        save_hf_template=template,
        loss_runtime=loss_runtime,
    )
    payload = {
        "samples": [_row("source-0")],
        "rollout_ids": [0],
        "loss": loss_family,
        "rollout_id": 0,
        "expected_runtime_load_id": serving_version,
        "max_staleness": 2,
        "producing_runtime_load_ids": [producing_version],
    }
    if loss_family == "openclawrl":
        payload["samples"] = [
            [
                "source-0",
                [10, 11, 12],
                [1, 1],
                [-0.1, -0.2],
                0.5,
                [[10, 20], [11, 21]],
                [[-0.1, -2.0], [-0.2, -2.0]],
                # Every trained sample carries >= 1 candidate: the un-enhanced
                # anchor is the native ids verbatim (tokens[-2:] is the
                # response tail the validator checks).
                [{"hint": "", "teacher_tokens": [10, 11, 12]}],
            ]
        ]
    if loss_family in {"openclawrl", "pg", "tttd"}:
        payload["advantages"] = [1.0]
    return actor, group, manager, payload


def _bridge_args(**overrides) -> SimpleNamespace:
    values = {
        "num_rollout": 1,
        "save_hf": "/checkpoints/hf/{rollout_id}",
        "save": "/checkpoints/megatron",
        "compute_advantages_and_returns": False,
        "debug_train_only": False,
        "debug_rollout_only": False,
        "rollout_num_gpus": 1,
        "rollout_external": False,
        "colocate": False,
        "offload_rollout": False,
        "offload_train": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.unit
def test_bridge_health_reports_start_rollout_id() -> None:
    group = _FakeGroup()
    actor = bridge.TrainBridgeActorImpl(
        group, _FakeRolloutManager(["packed"]), save_hf_template=None, start_rollout_id=3
    )

    health = actor.health()
    assert health["ok"] is True
    assert health["start_rollout_id"] == 3


@pytest.mark.unit
def test_bridge_reports_the_serving_runtime_load_id() -> None:
    group = _FakeGroup()
    actor = bridge.TrainBridgeActorImpl(group, _FakeRolloutManager([]), save_hf_template=None)

    assert actor.serving_runtime_load_id() == "v1"


@pytest.mark.unit
def test_durable_bridge_syncs_training_weights_before_reporting_ready(tmp_path) -> None:
    template = str(tmp_path / "checkpoint-{rollout_id}")
    group = _DurableGroup(template)

    actor = bridge.TrainBridgeActorImpl(group, _FakeRolloutManager([]), save_hf_template=template)

    assert group.update_calls == 1
    assert actor.health()["ok"] is True


@pytest.mark.unit
def test_bridge_packs_with_rollout_manager_then_passes_list_of_boxes(tmp_path, monkeypatch) -> None:
    packed = [Box("dp-0"), Box("dp-1")]
    manager = _FakeRolloutManager(packed)
    template = str(tmp_path / "checkpoint-{rollout_id}")
    group = _DurableGroup(template)

    def fake_ray_get(value, **kwargs):
        del kwargs
        return packed if value == "packed-ref" else value

    monkeypatch.setattr(bridge.ray, "get", fake_ray_get)
    actor = bridge.TrainBridgeActorImpl(group, manager, save_hf_template=template)
    payload = _payload(loss="sft")
    payload.update(rollout_id=0, expected_runtime_load_id="v1", parent_release_id="parent-0")

    result = _execute_and_update_weights(actor, payload)

    assert result.outcome == "complete"
    assert len(manager.calls) == 1
    assert manager.calls[0]["rollout_ids"] == [0, 0, 1]
    assert group.train_calls == [(0, packed)]
    assert isinstance(group.train_calls[0][1], list)
    assert all(isinstance(item, Box) for item in group.train_calls[0][1])


@pytest.mark.unit
def test_colocated_durable_job_offloads_then_publishes_before_completion(tmp_path) -> None:
    template = str(tmp_path / "checkpoint-{rollout_id}")
    # One shared log for both doubles: asserting the rollout lifecycle alone
    # cannot see where the checkpoint save lands relative to it, so a build that
    # published before offloading serving memory used to pass this test.
    timeline: list[str] = []
    group = _DurableGroup(template, timeline=timeline)
    manager = _FakeRolloutManager(["packed"], timeline=timeline)
    actor = bridge.TrainBridgeActorImpl(group, manager, save_hf_template=template, colocate=True)
    payload = _payload(loss="tttd", advantages=[0.25, -0.25, 0.0])
    payload.update(rollout_id=0, expected_runtime_load_id="v1")

    result = _execute_and_update_weights(actor, payload)

    assert result.outcome == "complete"
    assert timeline == [
        "onload_weights",
        "onload_kv",
        "pause_generation",
        "offload",
        "save_model",
        "onload_weights",
        "onload_kv",
    ]
    assert manager.lifecycle_calls == [
        "onload_weights",
        "onload_kv",
        "pause_generation",
        "offload",
        "onload_weights",
        "onload_kv",
    ]
    assert actor.health()["phase"] == "awaiting_commit"
    assert actor.health()["completed_train_steps"] == 1
    assert actor.health()["last_train_rollout_id"] == 0
    assert result.training_job_id is not None

    actor.acknowledge_training_commit(result.training_job_id)

    assert timeline[-1] == "continue_generation"
    assert actor.health()["phase"] == "serving"


@pytest.mark.unit
def test_bridge_checkpoint_requires_save_template() -> None:
    actor = bridge.TrainBridgeActorImpl(_FakeGroup(), _FakeRolloutManager([]), save_hf_template=None)
    payload = _payload(loss="sft")
    payload.update(rollout_id=0, expected_runtime_load_id="v1")

    with pytest.raises(RuntimeError, match="save_hf"):
        _execute_and_update_weights(actor, payload)


JOB_ID = "0956bba7f3b4ab2e268625c28e716cdd589bbf0d7b3a787b00311267aa0ff2e7"


@pytest.mark.unit
def test_durable_bridge_is_idempotent_and_rejects_stale_samples(tmp_path) -> None:
    actor, group, payload = _durable_actor(tmp_path)
    stale = {**payload, "expected_runtime_load_id": "old"}
    assert _execute_and_update_weights(actor, stale) == TrainingJobResult(outcome="stale", runtime_load_id="v1")

    first = _execute_and_update_weights(actor, payload)
    assert first.outcome == "complete"
    assert read_marker(tmp_path / ".reef-latest-job.json")["job_id"] == JOB_ID
    assert _execute_and_update_weights(actor, payload) == first
    assert len(group.train_calls) == 1
    assert group.save_calls == [(0, True)]


@pytest.mark.unit
def test_bridge_defers_resume_until_reef_acknowledges_the_commit(tmp_path) -> None:
    template = str(tmp_path / "checkpoint-{rollout_id}")
    group = _DurableGroup(template)
    manager = _FakeRolloutManager(["packed"])
    actor = bridge.TrainBridgeActorImpl(group, manager, save_hf_template=template)
    payload = _payload(loss="sft")
    payload.update(rollout_id=0, expected_runtime_load_id="v1")

    checkpoint = actor.execute_training_job(payload)

    assert checkpoint.outcome == "checkpoint"
    marker = read_marker(tmp_path / ".reef-latest-job.json")
    assert checkpoint.training_job_id == marker["job_id"]
    assert marker["status"] == "CHECKPOINT"
    assert manager.lifecycle_calls == []
    assert group.update_calls == 1  # startup publication only
    assert group.update_generation_management == [True]

    updated = actor.update_serving_weights(checkpoint.training_job_id)

    assert updated.outcome == "complete"
    assert read_marker(tmp_path / ".reef-latest-job.json")["status"] == "READY_TO_COMMIT"
    assert manager.lifecycle_calls == ["pause_generation"]
    assert group.update_generation_management == [True, False]
    assert actor.health()["training_job"]["status"] == "READY_TO_COMMIT"

    actor.acknowledge_training_commit(checkpoint.training_job_id)

    assert read_marker(tmp_path / ".reef-latest-job.json")["status"] == "COMPLETE"
    assert manager.lifecycle_calls == ["pause_generation", "continue_generation"]


@pytest.mark.unit
def test_pause_barrier_failure_leaves_checkpoint_replayable(tmp_path) -> None:
    template = str(tmp_path / "checkpoint-{rollout_id}")
    group = _DurableGroup(template)
    manager = _FakeRolloutManager(["packed"])
    actor = bridge.TrainBridgeActorImpl(group, manager, save_hf_template=template)
    payload = _payload(loss="sft")
    payload.update(rollout_id=0, expected_runtime_load_id="v1")
    checkpoint = actor.execute_training_job(payload)
    terminated: list[str] = []

    def fail_pause() -> None:
        raise RuntimeError("pause failed")

    manager.pause_generation_for_update = _RemoteMethod(fail_pause)
    manager.terminate_updatable_engines = _RemoteMethod(lambda: terminated.append("terminated"))

    with pytest.raises(RuntimeError, match="pause failed"):
        actor.update_serving_weights(checkpoint.training_job_id)

    assert read_marker(tmp_path / ".reef-latest-job.json")["status"] == "CHECKPOINT"
    assert terminated == ["terminated"]
    assert group.update_calls == 1  # startup publication only


@pytest.mark.unit
def test_bridge_admits_bounded_sao_lag_and_persists_exact_metrics(tmp_path) -> None:
    actor, group, manager, payload = _sao_durable_actor(tmp_path)

    result = _execute_and_update_weights(actor, payload)
    marker = read_marker(tmp_path / ".reef-latest-job.json")

    assert result.outcome == "complete"
    assert result.metrics["staleness/samples_fresh"] == 0
    assert result.metrics["staleness/samples_admitted_stale"] == 1
    assert marker["metrics"]["staleness/samples_admitted_stale"] == 1
    assert manager.calls[0]["rollout_log_probs"] == [[-0.1, -0.2]]
    assert "importance_weight" not in payload
    assert "importance_weight" not in manager.calls[0]
    assert len(group.train_calls) == 1


@pytest.mark.unit
def test_bridge_admits_and_preserves_mixed_token_runtime_load_ids(tmp_path) -> None:
    template = str(tmp_path / "checkpoint-{rollout_id}")
    group = _DurableGroup(template)
    group._actor_handlers[0].version = "engine:7"
    manager = _FakeRolloutManager(["packed"])
    manager.version = "engine:7"
    actor = bridge.TrainBridgeActorImpl(group, manager, save_hf_template=template)
    payload = {
        "samples": [
            _row(
                "mixed",
                tokens=(10, 20, 21, 22),
                loss_mask=(1, 1, 1),
                log_probs=(-0.1, -0.2, -0.3),
            )
        ],
        "rollout_ids": [0],
        "loss": "sft",
        "rollout_id": 0,
        "expected_runtime_load_id": "engine:7",
        "max_staleness": 2,
        "producing_runtime_load_ids": [None],
        "producing_runtime_load_spans": [
            [
                {"start": 0, "end": 1, "runtime_load_id": "engine:6"},
                {"start": 1, "end": 3, "runtime_load_id": "engine:7"},
            ]
        ],
    }

    result = _execute_and_update_weights(actor, payload)

    assert result.outcome == "complete"
    assert result.metrics["staleness/samples_fresh"] == 0
    assert result.metrics["staleness/samples_admitted_stale"] == 1
    assert manager.calls[0]["producing_runtime_load_spans"] == payload["producing_runtime_load_spans"]


@pytest.mark.unit
def test_bridge_rejects_mixed_token_versions_at_exact_admission_without_running_training(tmp_path) -> None:
    template = str(tmp_path / "checkpoint-{rollout_id}")
    group = _DurableGroup(template)
    group._actor_handlers[0].version = "engine:7"
    manager = _FakeRolloutManager(["packed"])
    manager.version = "engine:7"
    actor = bridge.TrainBridgeActorImpl(group, manager, save_hf_template=template)
    payload = {
        "samples": [_row("mixed", tokens=(10, 20, 21), loss_mask=(1, 1), log_probs=(-0.1, -0.2))],
        "rollout_ids": [0],
        "loss": "sft",
        "rollout_id": 0,
        "expected_runtime_load_id": "engine:7",
        "max_staleness": 0,
        "producing_runtime_load_ids": [None],
        "producing_runtime_load_spans": [
            [
                {"start": 0, "end": 1, "runtime_load_id": "engine:6"},
                {"start": 1, "end": 2, "runtime_load_id": "engine:7"},
            ]
        ],
    }

    result = _execute_and_update_weights(actor, payload)

    assert result.outcome == "stale"
    assert result.metrics["staleness/drop_reason"] == "policy_lag_exceeded"
    assert not group.train_calls
    assert not manager.calls


@pytest.mark.unit
@pytest.mark.parametrize(
    ("max_staleness", "lag", "expected_outcome"),
    [
        (0, 0, "complete"),
        (0, 1, "stale"),
        (1, 0, "complete"),
        (1, 1, "complete"),
        (1, 2, "stale"),
        (2, 0, "complete"),
        (2, 1, "complete"),
        (2, 2, "complete"),
        (2, 3, "stale"),
    ],
)
def test_bridge_enforces_the_inclusive_max_staleness_boundary(
    tmp_path,
    max_staleness,
    lag,
    expected_outcome,
) -> None:
    serving_sequence = 3
    producing_version = f"engine:{serving_sequence - lag}"
    actor, group, manager, payload = _sao_durable_actor(
        tmp_path,
        producing_versions=(producing_version,),
        serving_version=f"engine:{serving_sequence}",
        max_staleness=max_staleness,
    )
    if max_staleness == 0:
        payload.pop("max_staleness")
        payload.pop("producing_runtime_load_ids")
        payload["expected_runtime_load_id"] = producing_version

    result = _execute_and_update_weights(actor, payload)

    assert result.outcome == expected_outcome
    if expected_outcome == "complete":
        assert len(group.train_calls) == 1
        assert len(manager.calls) == 1
        if max_staleness > 0:
            assert result.metrics["staleness/samples_fresh"] == int(lag == 0)
            assert result.metrics["staleness/samples_admitted_stale"] == int(lag > 0)
    else:
        assert not group.train_calls
        assert not manager.calls
        if max_staleness > 0:
            assert result.metrics["staleness/drop_reason"] == "policy_lag_exceeded"


@pytest.mark.unit
@pytest.mark.parametrize("loss_family", ["openclawrl", "sao", "tttd"])
def test_bridge_admits_bounded_lag_for_every_cookbook_loss_family(tmp_path, loss_family) -> None:
    actor, group, manager, payload = _loss_family_durable_actor(tmp_path, loss_family)

    result = _execute_and_update_weights(actor, payload)

    assert result.outcome == "complete"
    assert result.metrics["staleness/samples_fresh"] == 0
    assert result.metrics["staleness/samples_admitted_stale"] == 1
    assert len(group.train_calls) == 1
    assert len(manager.calls) == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("producing_version", "reason", "policy_lags"),
    [
        ("engine:0", "policy_lag_exceeded", [3]),
        ("other:3", "cross_incarnation", None),
        ("engine:4", "future_producing_runtime_load_id", [-1]),
        ("malformed", "malformed_producing_runtime_load_id", None),
        ("engine:01", "malformed_producing_runtime_load_id", None),
        (None, "missing_producing_runtime_load_id", None),
    ],
)
def test_bridge_drops_inadmissible_sao_provenance_without_consuming_rollout(
    tmp_path,
    producing_version,
    reason,
    policy_lags,
) -> None:
    actor, group, manager, payload = _sao_durable_actor(
        tmp_path,
        producing_versions=(producing_version,),
    )

    result = _execute_and_update_weights(actor, payload)

    assert result.outcome == "stale"
    assert result.metrics["staleness/samples_dropped"] == 1
    assert result.metrics["staleness/drop_reason"] == reason
    assert result.metrics.get("staleness/drop_policy_lags") == policy_lags
    assert result.metrics["staleness/source_agent_record_ids"] == ["source-0"]
    assert actor.start_rollout_id() == 0
    assert not group.train_calls
    assert not group.save_calls
    assert not manager.calls


@pytest.mark.unit
def test_bridge_classifies_each_sao_sample_in_a_mixed_version_batch(tmp_path) -> None:
    actor, _, manager, payload = _sao_durable_actor(
        tmp_path,
        producing_versions=("engine:3", "engine:1"),
    )

    result = _execute_and_update_weights(actor, payload)

    assert result.outcome == "complete"
    assert result.metrics["staleness/samples_fresh"] == 1
    assert result.metrics["staleness/samples_admitted_stale"] == 1
    assert manager.calls[0]["producing_runtime_load_ids"] == ["engine:3", "engine:1"]


@pytest.mark.unit
def test_bridge_drops_heterogeneous_samples_in_exact_mode(tmp_path) -> None:
    actor, group, manager, payload = _sao_durable_actor(
        tmp_path,
        producing_versions=("engine:3", "engine:2"),
        max_staleness=0,
    )

    result = _execute_and_update_weights(actor, payload)

    assert result.outcome == "stale"
    assert result.metrics["staleness/drop_reason"] == "policy_lag_exceeded"
    assert not group.train_calls
    assert not manager.calls


@pytest.mark.unit
def test_sao_row_runtime_load_id_must_match_the_shared_training_payload(tmp_path) -> None:
    actor, group, manager, payload = _sao_durable_actor(tmp_path)
    payload["producing_runtime_load_ids"] = ["engine:2"]

    with pytest.raises(ValueError, match="row runtime load IDs do not match"):
        _execute_and_update_weights(actor, payload)

    assert not group.train_calls
    assert not manager.calls


@pytest.mark.unit
def test_bridge_accepts_the_initial_sao_weight_token(tmp_path) -> None:
    actor, _, _, payload = _sao_durable_actor(
        tmp_path,
        producing_versions=("engine:0",),
        serving_version="engine:0",
    )

    result = _execute_and_update_weights(actor, payload)

    assert result.outcome == "complete"
    assert result.metrics["staleness/samples_fresh"] == 1
    assert result.metrics["staleness/samples_admitted_stale"] == 0


@pytest.mark.unit
def test_enabled_sao_job_replays_after_fence_and_window_changes(tmp_path) -> None:
    actor, group, _, payload = _sao_durable_actor(tmp_path)

    first = _execute_and_update_weights(actor, payload)
    replayed = _execute_and_update_weights(
        actor,
        {
            **payload,
            "expected_runtime_load_id": "engine:4",
            "max_staleness": 1,
        },
    )

    assert replayed == first
    assert replayed.metrics["staleness/samples_admitted_stale"] == 1
    assert len(group.train_calls) == 1


@pytest.mark.unit
def test_enabled_sao_execution_fence_still_rejects_fresh_job(tmp_path) -> None:
    actor, group, _, payload = _sao_durable_actor(tmp_path)

    result = _execute_and_update_weights(actor, {**payload, "expected_runtime_load_id": "engine:2"})

    assert result.outcome == "stale"
    assert result.metrics["staleness/drop_reason"] == "execution_fence_mismatch"
    assert result.metrics["staleness/samples_dropped"] == 1
    assert not group.train_calls


@pytest.mark.unit
def test_enabled_sao_window_refuses_uncanonical_serving_version_without_dropping(tmp_path) -> None:
    actor, group, _, payload = _sao_durable_actor(
        tmp_path,
        producing_versions=("engine:1",),
        serving_version="v1",
    )

    with pytest.raises(RuntimeError, match="cannot classify staleness"):
        _execute_and_update_weights(actor, payload)

    assert actor.start_rollout_id() == 0
    assert not group.train_calls


@pytest.mark.unit
def test_bridge_rejects_symlinked_checkpoint(tmp_path, monkeypatch) -> None:
    actor, group, payload = _durable_actor(tmp_path)
    target = tmp_path / "checkpoint-target"
    target.mkdir()

    def save_model(rollout_id, force_sync=False):
        group.save_calls.append((rollout_id, force_sync))
        Path(group.template.format(rollout_id=rollout_id)).symlink_to(target, target_is_directory=True)

    monkeypatch.setattr(group, "save_model", save_model)

    with pytest.raises(RuntimeError, match="missing or unsafe"):
        _execute_and_update_weights(actor, payload)


@pytest.mark.unit
def test_bridge_catalogs_paired_checkpoint_metrics_and_blocks_before_second_optimizer(tmp_path) -> None:
    root = tmp_path / "checkpoints"
    template = str(root / "hf" / "{rollout_id}")
    megatron_root = root / "megatron"
    source_hf, source_megatron = tmp_path / "source-hf", tmp_path / "source-megatron"
    (source_hf / "weights").parent.mkdir(parents=True)
    (source_hf / "weights").write_text("hf", encoding="utf-8")
    (source_megatron / "iter_0000000").mkdir(parents=True)
    (source_megatron / "iter_0000000" / "state").write_text("megatron", encoding="utf-8")
    (source_megatron / "latest_checkpointed_iteration.txt").write_text("0", encoding="utf-8")
    hf_bytes = _allocated_bytes(source_hf)
    pair_bytes = max(hf_bytes + _allocated_bytes(source_megatron / "iter_0000000"), 8 * hf_bytes)
    group = _DurableGroup(template, megatron_root)
    manager = _FakeRolloutManager(["packed"])
    actor = bridge.TrainBridgeActorImpl(
        group,
        manager,
        save_hf_template=template,
        storage_config=RetentionConfig(max_storage_bytes=pair_bytes),
        megatron_save_root=str(megatron_root),
        source_hf=str(source_hf),
        source_megatron=str(source_megatron),
    )
    payload = _payload(loss="sft")
    payload.update(rollout_id=0, expected_runtime_load_id="v1", parent_release_id="parent-0")

    first = _execute_and_update_weights(actor, payload)
    assert first.training_job_id is not None
    actor.acknowledge_training_commit(first.training_job_id)
    record = read_marker(root / "hf" / ".reef-latest-job.json")
    catalog = actor._storage._record_path(0)
    stored = json.loads(catalog.read_text(encoding="utf-8"))

    assert first.outcome == "complete"
    assert record["rollout_id"] == stored["rollout_id"] == 0
    assert "loss" not in stored
    assert stored["reward"] == pytest.approx(sum(row[4] for row in payload["samples"]) / 3)

    second = {**payload, "rollout_id": 1}
    blocked = _execute_and_update_weights(actor, second)
    assert blocked.outcome == "storage_blocked"
    assert blocked.storage["blocked"] is True
    assert len(manager.calls) == len(group.train_calls) == 1
    assert len(group.save_calls) == 1
    assert read_marker(root / "hf" / ".reef-latest-job.json")["status"] == "COMPLETE"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "checkpoint_exists", "error"),
    [
        ("COMPLETE", True, None),
        ("CHECKPOINT", True, None),
        ("UPDATING_WEIGHTS", True, None),
        ("RUNNING", False, "ambiguous"),
        ("COMPLETE", False, "no checkpoint"),
    ],
)
def test_bridge_marker_recovery_is_fail_closed(tmp_path, status, checkpoint_exists, error) -> None:
    template = str(tmp_path / "checkpoint-{rollout_id}")
    checkpoint = Path(template.format(rollout_id=0))
    marker = {"status": status, "job_id": JOB_ID, "rollout_id": 0, "checkpoint_path": str(checkpoint)}
    if status == "RUNNING":
        marker.pop("checkpoint_path")
    elif status == "COMPLETE":
        marker["runtime_load_id"] = "v1"
    if checkpoint_exists:
        checkpoint.mkdir()
    write_marker(tmp_path / ".reef-latest-job.json", marker)
    group = _DurableGroup(template)
    payload = _payload(loss="sft")
    payload.update(rollout_id=0, expected_runtime_load_id="v1", parent_release_id="parent-0")

    if error is not None:
        with pytest.raises(RuntimeError, match=error):
            bridge.TrainBridgeActorImpl(group, _FakeRolloutManager(["packed"]), save_hf_template=template)
    else:
        manager = _FakeRolloutManager(["packed"])
        actor = bridge.TrainBridgeActorImpl(group, manager, save_hf_template=template)
        result = _execute_and_update_weights(actor, payload)
        assert result.outcome == "complete"
        assert result.training_job_id is not None
        actor.acknowledge_training_commit(result.training_job_id)
        recovered = read_marker(tmp_path / ".reef-latest-job.json")
        assert recovered["status"] == "COMPLETE"
        assert recovered["commit_acknowledged"] is True
        assert group.update_force_full == [True]
        if status == "UPDATING_WEIGHTS":
            assert manager.lifecycle_calls[:2] == ["recover_engines", "pause_generation"]
    assert not group.train_calls


@pytest.mark.unit
def test_unacknowledged_head_committed_marker_is_not_commit_proof(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-0"
    checkpoint.mkdir()
    path = tmp_path / ".reef-latest-job.json"
    write_marker(
        path,
        {
            "status": "HEAD_COMMITTED",
            "job_id": JOB_ID,
            "rollout_id": 0,
            "checkpoint_path": str(checkpoint),
            "runtime_load_id": "engine:1",
        },
    )

    with pytest.raises(RuntimeError, match="requires a commit acknowledgement"):
        read_marker(path)


@pytest.mark.unit
def test_marker_transition_cannot_skip_the_weight_update_phase(tmp_path) -> None:
    path = tmp_path / ".reef-latest-job.json"
    marker = {"status": "CHECKPOINT", "job_id": JOB_ID, "rollout_id": 0}

    with pytest.raises(RuntimeError, match=r"CHECKPOINT.*READY_TO_COMMIT"):
        transition_marker(path, marker, "READY_TO_COMMIT", runtime_load_id="engine:1")

    assert marker["status"] == "CHECKPOINT"
    assert not path.exists()


@pytest.mark.unit
def test_weight_update_recovery_converges_disagreeing_engines_before_startup_validation(tmp_path) -> None:
    template = str(tmp_path / "checkpoint-{rollout_id}")
    checkpoint = Path(template.format(rollout_id=0))
    checkpoint.mkdir()
    write_marker(
        tmp_path / ".reef-latest-job.json",
        {
            "status": "UPDATING_WEIGHTS",
            "job_id": JOB_ID,
            "rollout_id": 0,
            "checkpoint_path": str(checkpoint),
        },
    )
    manager = _FakeRolloutManager(["packed"])
    manager.versions = ["engine:1", "engine:2"]

    class RecoveryGroup(_DurableGroup):
        def update_weights(self, *, manage_generation: bool = True, force_full: bool = False):
            super().update_weights(manage_generation=manage_generation, force_full=force_full)
            self._actor_handlers[0].version = "engine:3"
            manager.versions = ["engine:3", "engine:3"]

    group = RecoveryGroup(template)
    actor = bridge.TrainBridgeActorImpl(group, manager, save_hf_template=template)

    assert manager.lifecycle_calls[:2] == ["recover_engines", "pause_generation"]
    assert group.update_force_full == [True]
    assert actor.health()["training_job"]["status"] == "READY_TO_COMMIT"
    assert actor.serving_runtime_load_id() == "engine:3"


@pytest.mark.unit
def test_complete_marker_republishes_checkpoint_with_its_original_runtime_load_id(tmp_path) -> None:
    template = str(tmp_path / "checkpoint-{rollout_id}")
    checkpoint = Path(template.format(rollout_id=0))
    checkpoint.mkdir()
    recovered_version = "checkpoint-incarnation:2"
    write_marker(
        tmp_path / ".reef-latest-job.json",
        {
            "status": "COMPLETE",
            "job_id": JOB_ID,
            "rollout_id": 0,
            "checkpoint_path": str(checkpoint),
            "runtime_load_id": recovered_version,
        },
    )
    manager = _FakeRolloutManager(["packed"])
    group = _DurableGroup(template)

    def restore_runtime_load_id(runtime_load_id):
        group.republication_calls.append(runtime_load_id)
        group._actor_handlers[0].version = runtime_load_id
        manager.version = runtime_load_id

    group.restore_runtime_load_id_for_republication = restore_runtime_load_id

    actor = bridge.TrainBridgeActorImpl(
        group,
        manager,
        save_hf_template=template,
        loss_runtime=resolve_loss_family("sao").bind(),
        critic_group=_FakeGroup(),
    )

    assert group.republication_calls == [recovered_version]
    assert group.update_calls == 1
    assert actor.serving_runtime_load_id() == recovered_version
    assert read_marker(tmp_path / ".reef-latest-job.json")["runtime_load_id"] == recovered_version

    payload = {
        "samples": [
            [
                "recovered-source",
                [10, 11, 12],
                [1, 1],
                [-0.1, -0.2],
                0.5,
                [1, 1],
                "checkpoint-incarnation:1",
                1234.5,
            ]
        ],
        "rollout_ids": [0],
        "loss": "sao",
        "rollout_id": 1,
        "expected_runtime_load_id": recovered_version,
        "max_staleness": 2,
        "producing_runtime_load_ids": ["checkpoint-incarnation:1"],
    }
    result = _execute_and_update_weights(actor, payload)

    assert result.outcome == "complete"
    assert result.metrics["staleness/samples_admitted_stale"] == 1


@pytest.mark.unit
def test_serving_republication_preserves_current_runtime_load_id() -> None:
    manager = _FakeRolloutManager([])
    group = _FakeGroup()
    actor = bridge.TrainBridgeActorImpl(group, manager, save_hf_template=None)

    def restore_runtime_load_id(runtime_load_id):
        group.republication_calls.append(runtime_load_id)
        group._actor_handlers[0].version = runtime_load_id
        manager.version = runtime_load_id

    group.restore_runtime_load_id_for_republication = restore_runtime_load_id

    assert actor.republish_serving() == "v1"
    assert group.republication_calls == ["v1"]
    assert group.update_calls == 1
    assert actor.serving_runtime_load_id() == "v1"


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["empty", "symlink"])
def test_bridge_marker_rejects_unsafe_checkpoint_path(tmp_path, kind) -> None:
    target = tmp_path / "checkpoint"
    target.mkdir()
    link = tmp_path / "checkpoint-link"
    link.symlink_to(target, target_is_directory=True)
    marker_path = tmp_path / ".reef-latest-job.json"
    write_marker(
        marker_path,
        {
            "status": "COMPLETE",
            "job_id": JOB_ID,
            "rollout_id": 0,
            "checkpoint_path": "" if kind == "empty" else str(link),
            "runtime_load_id": "v1",
        },
    )

    with pytest.raises(RuntimeError, match="no checkpoint"):
        read_marker(marker_path)


@pytest.mark.unit
def test_load_args_file_expands_variables_and_uses_shell_like_quotes(tmp_path: Path, monkeypatch) -> None:
    from reef.train.slime_backend.reef_adapters.driver import load_args_file

    monkeypatch.setenv("BRIDGE_MODEL", "/models/demo model")
    args_file = tmp_path / "args.txt"
    args_file.write_text(
        '--hf-checkpoint "$BRIDGE_MODEL"\n'
        "--save-hf '/checkpoints/step-{rollout_id}'\n"
        "# comments are ignored by shlex\n",
        encoding="utf-8",
    )

    assert load_args_file(args_file) == [
        "--hf-checkpoint",
        "/models/demo model",
        "--save-hf",
        "/checkpoints/step-{rollout_id}",
    ]


@pytest.mark.unit
def test_driver_ready_file_is_atomic_and_driver_option_is_not_forwarded(tmp_path: Path) -> None:
    from reef.train.slime_backend.reef_adapters.driver import (
        READY_MARKER,
        _driver_options,
        _retention_options,
        _write_ready_file,
    )

    ready_file = tmp_path / "state" / "bridge.ready"
    parsed_ready_file, remaining = _driver_options([f"--ready-file={ready_file}", "--loss-type", "sft_loss"])

    assert parsed_ready_file == ready_file
    assert remaining == ["--loss-type", "sft_loss"]
    retention, remaining = _retention_options(
        [
            "--reef-checkpoint-policy=best_reward",
            "--reef-checkpoint-max-storage=100GiB",
            "--loss-type",
            "sft_loss",
        ]
    )
    assert retention == RetentionConfig(policy="best_reward", max_storage_bytes=100 * 1024**3)
    assert remaining == ["--loss-type", "sft_loss"]

    _write_ready_file(ready_file)

    assert ready_file.read_text(encoding="utf-8") == f"{READY_MARKER}\n"
    assert list(ready_file.parent.glob(f".{ready_file.name}.*.tmp")) == []


@pytest.mark.unit
def test_slime_rejects_its_legacy_raw_wandb_flags() -> None:
    from reef.train.slime_backend.reef_adapters.driver import _validate_tracking_args

    _validate_tracking_args(SimpleNamespace(use_wandb=False, wandb_key=None))
    with pytest.raises(RuntimeError, match=r"observability\.wandb"):
        _validate_tracking_args(SimpleNamespace(use_wandb=True, wandb_key=None))
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        _validate_tracking_args(SimpleNamespace(use_wandb=False, wandb_key="secret"))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("loss_family", "loss_type", "use_rollout_logprobs"),
    [
        ("sft", "sft_loss", False),
        ("pg", "policy_loss", True),
    ],
)
def test_driver_accepts_matching_reef_and_slime_objectives(loss_family, loss_type, use_rollout_logprobs) -> None:
    from reef.train.slime_backend.loss_families import resolve_loss_family

    resolve_loss_family(loss_family).validate_backend_args(
        SimpleNamespace(loss_type=loss_type, use_rollout_logprobs=use_rollout_logprobs)
    )


@pytest.mark.unit
def test_driver_derives_the_loss_family_from_the_configured_recipe() -> None:
    from reef.train.slime_backend.loss_families import resolve_loss_family
    from reef.train.slime_backend.reef_adapters.driver import _resolve_training_recipe

    recipe = "recipes.sao.recipe:SAORecipe"
    assert _resolve_training_recipe({"reef": {"recipe": recipe}}) == (
        "sao",
        recipe,
        resolve_loss_family("sao"),
    )
    with pytest.raises(RuntimeError, match=r"must define reef\.recipe"):
        _resolve_training_recipe({})
    with pytest.raises(RuntimeError, match="WeightTrainingRecipe"):
        _resolve_training_recipe({"reef": {"recipe": "reef.recipe.base:Recipe"}})


@pytest.mark.unit
def test_driver_stamps_a_dotted_family_reference_for_the_workers() -> None:
    from types import SimpleNamespace

    from reef.train.slime_backend.reef_adapters.driver import _stamp_loss_family_reference

    dotted = SimpleNamespace(loss_family="toy")
    _stamp_loss_family_reference(dotted, "toy_pkg.family:ToyAlgorithm")
    assert dotted.loss_family_ref == "toy_pkg.family:ToyAlgorithm"

    cookbook = SimpleNamespace(loss_family="sao")
    _stamp_loss_family_reference(cookbook, "sao")
    assert cookbook.loss_family_ref == "recipes.sao.slime:SaoAlgorithm"

    unreferenced = SimpleNamespace(loss_family="pg")
    _stamp_loss_family_reference(unreferenced, "pg")
    assert not hasattr(unreferenced, "loss_family_ref")


@pytest.mark.unit
def test_driver_rejects_a_recipe_with_an_unknown_loss_family(monkeypatch) -> None:
    import sys
    from types import ModuleType

    from reef.recipe import WeightTrainingRecipe, WeightTrainingSpec
    from reef.train.slime_backend.reef_adapters.driver import _resolve_training_recipe

    class UnknownLossRecipe(WeightTrainingRecipe):
        @classmethod
        def training_spec(cls):
            return WeightTrainingSpec(step_preparer="unused", loss_family="grpo")

    module = ModuleType("unknown_loss_recipe")
    module.UnknownLossRecipe = UnknownLossRecipe
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(RuntimeError, match="declares unsupported loss family 'grpo'"):
        _resolve_training_recipe({"reef": {"recipe": "unknown_loss_recipe:UnknownLossRecipe"}})


@pytest.mark.unit
def test_driver_accepts_tttd_custom_objective() -> None:
    from reef.train.slime_backend.loss_families import resolve_loss_family

    resolve_loss_family("tttd").validate_backend_args(
        SimpleNamespace(
            loss_type="custom_loss",
            use_rollout_logprobs=True,
            compute_advantages_and_returns=True,
            kl_coef=0.1,
        ),
    )


def _sao_backend_namespace(**overrides) -> SimpleNamespace:
    values = {
        "loss_type": "policy_loss",
        "use_rollout_logprobs": True,
        "custom_advantage_function_path": "recipes.sao.slime.objective.sao_advantages",
        "custom_pg_loss_function_path": "recipes.sao.slime.objective.sao_loss",
        "use_critic": True,
        "eps_clip": 0.3,
        "eps_clip_high": 5.0,
        "critic_steps_per_actor": 2,
        "num_critic_only_steps": 0,
        "sao_length_adaptive_lambda": True,
        "sao_lambda_alpha": 1.5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.unit
def test_driver_accepts_sao_objective() -> None:
    from reef.train.slime_backend.loss_families import resolve_loss_family

    resolve_loss_family("sao").validate_backend_args(_sao_backend_namespace())


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("eps_clip", 1.0, "eps-clip in"),
        ("eps_clip_high", -0.1, "eps-clip-high"),
        ("critic_steps_per_actor", 0, "critic-steps-per-actor"),
        ("num_critic_only_steps", -1, "num-critic-only-steps"),
        ("sao_lambda_alpha", 0.0, "sao-lambda-alpha"),
    ],
)
def test_driver_rejects_invalid_sao_backend_contract(field, value, message) -> None:
    from reef.train.slime_backend.loss_families import resolve_loss_family

    with pytest.raises(RuntimeError, match=message):
        resolve_loss_family("sao").validate_backend_args(_sao_backend_namespace(**{field: value}))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        # A false use_critic means the run would train with no value model:
        # SAO opts in via the explicit --use-critic flag on the Slime driver.
        ({"use_critic": False}, "requires a value model"),
    ],
)
def test_driver_rejects_sao_without_its_custom_objective_or_value_model(overrides, message) -> None:
    from reef.train.slime_backend.loss_families import resolve_loss_family

    with pytest.raises(RuntimeError, match=message):
        resolve_loss_family("sao").validate_backend_args(_sao_backend_namespace(**overrides))


@pytest.mark.parametrize(
    ("loss_family", "loss_type", "use_rollout_logprobs", "message"),
    [
        ("sft", "policy_loss", False, "sft_loss"),
        ("pg", "sft_loss", True, "policy_loss"),
        ("pg", "policy_loss", False, "use-rollout-logprobs"),
    ],
)
def test_driver_rejects_mismatched_reef_and_slime_objectives(
    loss_family,
    loss_type,
    use_rollout_logprobs,
    message,
) -> None:
    from reef.train.slime_backend.loss_families import resolve_loss_family

    with pytest.raises(RuntimeError, match=message):
        resolve_loss_family(loss_family).validate_backend_args(
            SimpleNamespace(loss_type=loss_type, use_rollout_logprobs=use_rollout_logprobs)
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("ref_load", "expected"),
    [
        ("/checkpoints/initial", "/checkpoints/initial"),
        (None, "/models/hf"),
    ],
)
def test_bridge_mode_first_start_falls_back_to_initial_checkpoint(
    tmp_path: Path, ref_load: str | None, expected: str
) -> None:
    from reef.train.slime_backend.reef_adapters.driver import _apply_bridge_resume_fallback

    args = SimpleNamespace(
        megatron_to_hf_mode="bridge",
        load=str(tmp_path / "empty-resume-directory"),
        ref_load=ref_load,
        hf_checkpoint="/models/hf",
        start_rollout_id=9,
    )

    _apply_bridge_resume_fallback(args)

    assert args.load == expected
    assert args.start_rollout_id == 0


@pytest.mark.unit
def test_bridge_mode_restart_keeps_resumable_checkpoint(tmp_path: Path) -> None:
    from reef.train.slime_backend.reef_adapters.driver import _apply_bridge_resume_fallback

    resume = tmp_path / "resume"
    resume.mkdir()
    (resume / "latest_checkpointed_iteration.txt").write_text("3", encoding="utf-8")
    args = SimpleNamespace(
        megatron_to_hf_mode="bridge",
        load=str(resume),
        ref_load="/checkpoints/initial",
        hf_checkpoint="/models/hf",
        start_rollout_id=4,
    )

    _apply_bridge_resume_fallback(args)

    assert args.load == str(resume)
    assert args.start_rollout_id == 4


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"debug_train_only": True}, "debug-train-only"),
        ({"debug_rollout_only": True}, "debug-rollout-only"),
        ({"rollout_num_gpus": 0}, "rollout-num-gpus"),
    ],
)
def test_start_bridge_rejects_configuration_without_local_inference(
    overrides,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        bridge.start_bridge(_bridge_args(**overrides))


@pytest.mark.unit
def test_start_bridge_rejects_ambiguous_marker_before_creating_workers(tmp_path, monkeypatch) -> None:
    root = tmp_path / "checkpoints"
    template = str(root / "hf" / "{rollout_id}")
    write_marker(
        root / "hf" / ".reef-latest-job.json",
        {"status": "RUNNING", "job_id": "job", "rollout_id": 0},
    )
    monkeypatch.setattr(
        bridge,
        "create_placement_groups",
        lambda args: pytest.fail("workers started before marker validation"),
    )
    args = _bridge_args(save_hf=template, save=str(root / "megatron"))

    with pytest.raises(RuntimeError, match="ambiguous"):
        bridge.start_bridge(args, retention=RetentionConfig(max_storage_bytes=100))


@pytest.mark.unit
def test_start_bridge_rejects_blocked_storage_before_creating_workers(tmp_path, monkeypatch) -> None:
    root = tmp_path / "checkpoints"
    monkeypatch.setattr(
        bridge.CheckpointStorage,
        "validate_capacity",
        lambda self, **kwargs: {
            "blocked": True,
            "reasons": ["checkpoint reservation would violate the filesystem free-space floor"],
        },
    )
    monkeypatch.setattr(
        bridge,
        "create_placement_groups",
        lambda args: pytest.fail("workers started before storage validation"),
    )
    args = _bridge_args(
        save_hf=str(root / "hf" / "{rollout_id}"),
        save=str(root / "megatron"),
    )

    with pytest.raises(RuntimeError, match=r"storage preflight.*free-space floor"):
        bridge.start_bridge(args)


@pytest.mark.unit
def test_start_bridge_delegates_initial_sync_to_actor(tmp_path, monkeypatch) -> None:
    events = []
    rollout_manager = object()
    monkeypatch.setenv("HOME", str(tmp_path))

    class Group:
        def update_weights(self):
            events.append("update_weights")

    group = Group()

    class ActorClass:
        def options(self, **kwargs):
            events.append(("options", kwargs))
            return self

        def remote(self, *args, **kwargs):
            events.append(("remote", args, kwargs))
            return "bridge-handle"

    monkeypatch.setattr(bridge.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(bridge, "create_placement_groups", lambda args: {"rollout": "rollout-pg"})
    monkeypatch.setattr(
        bridge,
        "create_rollout_manager",
        lambda args, pg: rollout_manager,
    )
    monkeypatch.setattr(
        bridge,
        "create_train_groups",
        lambda args, pgs, manager: (group, None),
    )
    monkeypatch.setattr(bridge, "TrainBridgeActor", ActorClass())
    monkeypatch.setattr(
        bridge.CheckpointStorage,
        "validate_capacity",
        lambda self, **kwargs: {"blocked": False, "reasons": []},
    )
    args = _bridge_args(
        save_hf="~/checkpoints/hf/{rollout_id}",
        save="~/checkpoints/megatron",
        hf_checkpoint="/models/hf",
        load="/models/megatron",
        start_rollout_id=2,
    )

    result = bridge.start_bridge(args)

    assert result == "bridge-handle"
    assert events[0] == ("options", {"name": "reef-train-bridge", "namespace": "reef"})
    assert events[1][0] == "remote"
    assert events[1][1][0] is group
    assert isinstance(events[1][2]["storage_config"], RetentionConfig)
    assert events[1][2]["save_hf_template"] == args.save_hf == str(tmp_path / "checkpoints/hf/{rollout_id}")
    assert events[1][2]["megatron_save_root"] == args.save == str(tmp_path / "checkpoints/megatron")
    # The critic group rides the keyword contract: recipes without a value
    # model (this one) hand the bridge None rather than omitting the argument.
    assert events[1][2]["critic_group"] is None


@pytest.mark.unit
def test_start_bridge_ships_a_resolvable_loss_family_reference(tmp_path, monkeypatch) -> None:
    """An external family must reach the actor as its dotted reference.

    The bridge actor re-resolves ``loss_family`` in a fresh Ray process whose
    registry is empty, so the plain name of a family registered via
    ``register_loss_family_ref`` only resolves there as its dotted reference
    (``resolve`` imports it and registers the spec). Names without a
    registered reference ship unchanged.
    """
    import sys
    from types import ModuleType

    from reef.train.algos.registry import register_loss_family_ref
    from reef.train.slime_backend.algorithm import SlimeAlgorithm
    from reef.train.slime_backend.loss_families import unregister_loss_family

    class _ExternalAlgorithm(SlimeAlgorithm):
        loss_family = "external_family"
        loss_type = "sft_loss"

        def validate_specific_args(self, args, source):
            pass

    module = ModuleType("external_family_pkg")
    module.ALGORITHM = _ExternalAlgorithm()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "external_family_pkg", module)
    register_loss_family_ref("external_family", "external_family_pkg:ALGORITHM")

    events = []
    monkeypatch.setenv("HOME", str(tmp_path))

    class ActorClass:
        def options(self, **kwargs):
            return self

        def remote(self, *args, **kwargs):
            events.append(kwargs)
            return "bridge-handle"

    monkeypatch.setattr(bridge.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(bridge, "create_placement_groups", lambda args: {"rollout": "rollout-pg"})
    monkeypatch.setattr(bridge, "create_rollout_manager", lambda args, pg: object())
    monkeypatch.setattr(bridge, "create_train_groups", lambda args, pgs, manager: (object(), None))
    monkeypatch.setattr(bridge, "TrainBridgeActor", ActorClass())
    monkeypatch.setattr(
        bridge.CheckpointStorage,
        "validate_capacity",
        lambda self, **kwargs: {"blocked": False, "reasons": []},
    )
    args = _bridge_args(
        save_hf="~/checkpoints/hf/{rollout_id}",
        save="~/checkpoints/megatron",
    )

    try:
        assert bridge.start_bridge(args, loss_family="external_family") == "bridge-handle"
        assert events[0]["loss_family"] == "external_family_pkg:ALGORITHM"
    finally:
        unregister_loss_family("external_family")

    # A registered family without a dotted reference ships under its own
    # name, and an explicit dotted reference passes through untouched.
    bridge.start_bridge(args, loss_family="sft")
    assert events[1]["loss_family"] == "sft"


@pytest.mark.unit
def test_driver_sao_options_strip_sao_flags_and_project_them_onto_args() -> None:
    # The --sao-* family is owned by the sao plugin: the driver strips the
    # flags before Slime's parser sees them and projects the values back onto
    # args so the Megatron-side advantage function can observe them.
    from recipes.sao.slime import SaoSettings

    spec = resolve_loss_family("sao")
    settings, remaining = spec.parse_specific_options(
        ["--sao-length-adaptive-lambda", "--sao-lambda-alpha", "2.0", "--loss-type", "policy_loss"]
    )

    assert settings == SaoSettings(length_adaptive_lambda=True, lambda_alpha=2.0)
    assert remaining == ["--loss-type", "policy_loss"]

    args = SimpleNamespace()
    spec.apply_driver_options(args, settings)
    assert args.sao_length_adaptive_lambda is True
    assert args.sao_lambda_alpha == 2.0

    defaults = SimpleNamespace()
    spec.apply_driver_options(defaults, None)
    assert defaults.sao_length_adaptive_lambda is False
    assert defaults.sao_lambda_alpha == 1.5


def _grouped_batch(groups: int, size: int, batch_id: str = "batch") -> GroupedPolicyBatch:
    return GroupedPolicyBatch(
        batch_id,
        tuple(
            tuple(
                PolicySample(f"g{group}-s{index}", (group, index), (1,), (-0.1,), float(index))
                for index in range(size)
            )
            for group in range(groups)
        ),
    )


@pytest.mark.unit
def test_slime_preparation_explicit_batch_size_cuts_one_batch_into_steps() -> None:
    batch = _grouped_batch(groups=4, size=2)

    payload = _build_payload(batch, "pg", tuple(range(8)), StepScheduling(batch_size=2))

    assert payload["rollout_ids"] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert payload["external_step_sizes"] == [2, 2]
    assert payload["advantages"] == list(range(8))
    converted = to_slime_rollout_data(payload)
    assert converted["external_step_sizes"] == [2, 2]


@pytest.mark.unit
def test_slime_preparation_epochs_repeat_rollouts_with_distinct_ids() -> None:
    batch = _grouped_batch(groups=2, size=2)

    payload = _build_payload(batch, "pg", (1.0, 2.0, 3.0, 4.0), StepScheduling(batch_size=1, epochs=3))

    assert [row[0] for row in payload["samples"]] == ["g0-s0", "g0-s1", "g1-s0", "g1-s1"] * 3
    assert payload["rollout_ids"] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert payload["advantages"] == [1.0, 2.0, 3.0, 4.0] * 3
    assert payload["external_step_sizes"] == [1] * 6


@pytest.mark.unit
def test_slime_preparation_shuffle_is_deterministic_per_batch_and_keeps_groups_contiguous() -> None:
    scheduling = StepScheduling(batch_size=2, epochs=2, shuffle=True)
    first = _build_payload(_grouped_batch(8, 2, "b-1"), "pg", tuple(range(16)), scheduling)
    again = _build_payload(_grouped_batch(8, 2, "b-1"), "pg", tuple(range(16)), scheduling)
    other = _build_payload(_grouped_batch(8, 2, "b-2"), "pg", tuple(range(16)), scheduling)

    assert first == again
    assert first["samples"] != other["samples"]
    names = [row[0] for row in first["samples"]]
    assert names != [
        row[0]
        for row in _build_payload(_grouped_batch(8, 2, "b-1"), "pg", None, StepScheduling(batch_size=2, epochs=2))[
            "samples"
        ]
    ]
    for epoch in range(2):
        epoch_names = names[epoch * 16 : (epoch + 1) * 16]
        assert sorted(epoch_names) == sorted(f"g{g}-s{s}" for g in range(8) for s in range(2))
        # every comparison set stays contiguous and in source order
        for start in range(0, 16, 2):
            g0, g1 = epoch_names[start], epoch_names[start + 1]
            assert g0[:2] == g1[:2] and g0.endswith("s0") and g1.endswith("s1")
    # advantages travel with their rows
    for row, advantage in zip(first["samples"], first["advantages"], strict=True):
        g, s = int(row[0][1]), int(row[0][-1])
        assert advantage == g * 2 + s


@pytest.mark.unit
def test_slime_preparation_remainder_policy() -> None:
    batch = _grouped_batch(groups=5, size=1)

    with pytest.raises(ValueError, match="5 rollouts do not form complete optimizer steps of 2"):
        _build_payload(batch, "pg", tuple(range(5)), StepScheduling(batch_size=2, remainder="error"))
    with pytest.raises(ValueError, match="exceeds the 5 rollouts"):
        _build_payload(batch, "pg", tuple(range(5)), StepScheduling(batch_size=8))

    dropped = _build_payload(batch, "pg", tuple(range(5)), StepScheduling(batch_size=2, remainder="drop"))
    assert [row[0] for row in dropped["samples"]] == ["g0-s0", "g1-s0", "g2-s0", "g3-s0"]
    assert dropped["advantages"] == [0, 1, 2, 3]
    assert dropped["external_step_sizes"] == [2, 2]

    partial = _build_payload(batch, "pg", tuple(range(5)), StepScheduling(batch_size=2, epochs=2))
    assert len(partial["samples"]) == 10
    assert partial["external_step_sizes"] == [2, 2, 1, 2, 2, 1]


@pytest.mark.unit
def test_slime_preparation_actual_batch_size_applies_to_policy_batches() -> None:
    batch = PolicyBatch(
        "batch",
        (PolicySample("a", (1,), (1,), (-0.1,), 0.2), PolicySample("b", (2,), (1,), (-0.2,), 0.8)),
    )

    payload = _build_payload(batch, "sft", None, StepScheduling(batch_size="actual"))

    assert payload["external_step_sizes"] == [2]


@pytest.mark.unit
def test_slime_preparation_configured_batch_size_forwards_remainder_policy() -> None:
    batch = _grouped_batch(groups=2, size=1)

    assert _build_payload(batch, "pg", (0.0, 1.0), StepScheduling())["external_remainder"] == "partial"
    payload = _build_payload(batch, "pg", (0.0, 1.0), StepScheduling(remainder="error"))
    assert payload["external_remainder"] == "error"
    assert "external_step_sizes" not in payload
    assert to_slime_rollout_data(payload)["external_remainder"] == "error"


@pytest.mark.unit
def test_to_slime_rollout_data_validates_step_layout() -> None:
    sample = ["a", [1, 2], [1], [-0.1], 0.5]
    payload = {"samples": [sample, sample, sample, sample], "rollout_ids": [0, 1, 2, 3], "loss": "sft"}

    assert to_slime_rollout_data({**payload, "external_step_sizes": [3, 1]})["external_step_sizes"] == [3, 1]
    with pytest.raises(ValueError, match="sum 3 must equal the 4 distinct rollout_ids"):
        to_slime_rollout_data({**payload, "external_step_sizes": [3]})
    with pytest.raises(ValueError, match="positive integers"):
        to_slime_rollout_data({**payload, "external_step_sizes": [4, 0]})
    with pytest.raises(ValueError, match="only when external_step_sizes is absent"):
        to_slime_rollout_data({**payload, "external_step_sizes": [4], "external_remainder": "partial"})
    with pytest.raises(ValueError, match="external_remainder must be"):
        to_slime_rollout_data({**payload, "external_remainder": "drop"})


@pytest.mark.unit
def test_prepare_slime_step_reports_schedule_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    from reef.train.algos import StepSignal
    from reef.train.slime_backend.reef_adapters import preparation

    def preparer(batch, state):
        return StepSignal(
            "train", "pg", {}, {"steps": 1}, tuple(range(8)), StepScheduling(batch_size=3, epochs=2, remainder="drop")
        )

    monkeypatch.setattr(preparation, "resolve_preparer", lambda _name: preparer)

    result = preparation.prepare_slime_step(_grouped_batch(8, 1), "any", {})

    assert result.metrics == {"steps": 1, "epochs": 2, "optimizer_steps": 4, "dropped_rollouts": 2}
    assert result.payload is not None and len(result.payload["samples"]) == 12
