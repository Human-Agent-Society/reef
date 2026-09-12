"""Named Ray actor that exposes slime's training actor group to a remote reef.

reef and slime run as separate services connected to one Ray cluster. This
module boots the slime training stack and publishes it behind a named
:class:`TrainBridgeActor`; the reef service then looks the actor up by name
(``reef.runtime.adapters.ray_runtime.connect_ray_runtime``) and drives
training through it.

Keeping the bridge here lets slime own its wire format: reef sends
framework-agnostic sample rows and the bridge converts them into slime's
rollout payload, so reef carries no slime-specific payload knowledge.

This module deliberately lives in ``reef.train.slime_backend`` and
imports its sibling runtime, keeping the reef package free of any Slime
dependency.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Literal

import ray

from reef.core.artifact_ref import parse_runtime_load_spans
from reef.runtime.adapter_residency import AdapterCapacityExhausted, AdapterEvictionFailed, AdapterResidencyManager
from reef.runtime.base import PreparedTrainingStep, TrainingJobResult
from reef.runtime.executor import Executor, resolve
from reef.runtime.executor.ray import RayExecutor
from reef.runtime.names import DEFAULT_ACTOR_NAME, DEFAULT_NAMESPACE
from reef.runtime.training_job.execution import (
    PreparedTrainingJob,
    TrainingCheckpoint,
    TrainingExecution,
    TrainingMetrics,
)
from reef.runtime.training_job.execution import max_staleness as _max_staleness
from reef.runtime.training_job.execution import uses_staleness_admission as _uses_staleness_admission
from reef.runtime.training_job.marker import marker_path, marker_result, marker_rollouts, read_marker
from reef.runtime.training_job.publication import TrainingPublication
from reef.surface.adapter import parse_adapter_name
from reef.train.algos.registry import loss_family_refs
from reef.train.slime_backend.algorithm import SlimeAlgorithm
from reef.train.slime_backend.data_builder import to_slime_rollout_data
from reef.train.slime_backend.loss_families import resolve_loss_family
from reef.train.slime_backend.reef_adapters.preflight import (
    configure_megatron_runtime,
    configure_rollout_runtime,
    configure_sglang_runtime,
    prepare_checkpoint_storage,
    validate_bridge_args,
)
from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_step
from reef.train.slime_backend.reef_adapters.training_job.scenarios import ScenarioHistory, history_path
from reef.train.slime_backend.reef_adapters.training_job.storage import CheckpointStorage, RetentionConfig

DEFAULT_BRIDGE_ACTOR_NAME = DEFAULT_ACTOR_NAME

# One training step (train + checkpoint + publish) legitimately takes hours;
# this bounds a single Ray RPC from the bridge to its workers.
_TRAIN_RPC_TIMEOUT_S = 14_400


@dataclass(frozen=True, slots=True)
class _StalenessDecision:
    action: Literal["admit", "drop"]
    metrics: Mapping[str, Any]


def _source_agent_record_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
    samples = payload.get("samples")
    if not isinstance(samples, Sequence) or isinstance(samples, str | bytes):
        return ()
    return tuple(
        str(row[0]) for row in samples if isinstance(row, Sequence) and not isinstance(row, str | bytes) and row
    )


def _producing_runtime_load_ids(payload: Mapping[str, Any]) -> Sequence[Any]:
    versions = payload.get("producing_runtime_load_ids")
    if not isinstance(versions, Sequence) or isinstance(versions, str | bytes) or not versions:
        raise ValueError("bounded staleness admission requires producing_runtime_load_ids")
    source_ids = _source_agent_record_ids(payload)
    if len(versions) != len(source_ids):
        raise ValueError(
            "bounded staleness admission requires one producing runtime load ID "
            f"per sample: {len(versions)} versions for {len(source_ids)} samples"
        )
    return versions


def _admission_runtime_load_id_groups(payload: Mapping[str, Any]) -> list[list[Any]]:
    """Return each sample's exact span versions for bounded admission."""
    versions = _producing_runtime_load_ids(payload)
    raw_groups = payload.get("producing_runtime_load_spans")
    if raw_groups is None:
        return [[version] for version in versions]
    if not isinstance(raw_groups, Sequence) or isinstance(raw_groups, str | bytes) or len(raw_groups) != len(versions):
        raise ValueError("producing_runtime_load_spans must contain one span list per sample")
    samples = payload["samples"]
    groups: list[list[Any]] = []
    for sample_index, (raw_spans, scalar) in enumerate(zip(raw_groups, versions, strict=True)):
        if not raw_spans:
            groups.append([scalar])
            continue
        row = samples[sample_index]
        response_length = len(row[2]) if isinstance(row, Sequence) and len(row) > 2 else None
        spans = parse_runtime_load_spans(
            raw_spans,
            field_name=f"producing_runtime_load_spans[{sample_index}]",
            response_length=response_length,
        )
        group = [span.runtime_load_id for span in spans]
        span_versions = set(group)
        if scalar is not None and span_versions != {scalar}:
            raise ValueError(f"producing runtime load ID for sample {sample_index} disagrees with its token spans")
        groups.append(group)
    return groups


def _stale_drop_decision(
    payload: Mapping[str, Any],
    *,
    serving_runtime_load_id: str,
    producing_runtime_load_ids: Sequence[Any],
    reason: str,
    policy_lags: Sequence[int] = (),
) -> _StalenessDecision:
    source_ids = _source_agent_record_ids(payload)
    metrics: dict[str, Any] = {
        "staleness/samples_dropped": len(source_ids) or len(producing_runtime_load_ids),
        "staleness/drop_reason": reason,
        "staleness/source_agent_record_ids": list(source_ids),
        "staleness/producing_runtime_load_ids": [
            None if version is None else str(version) for version in producing_runtime_load_ids
        ],
        "staleness/serving_runtime_load_id": serving_runtime_load_id,
    }
    if policy_lags:
        metrics["staleness/drop_policy_lags"] = list(policy_lags)
    return _StalenessDecision(action="drop", metrics=metrics)


def _staleness_admission(
    payload: Mapping[str, Any],
    *,
    serving_runtime_load_id: str,
    max_staleness: int,
) -> _StalenessDecision:
    # The reef wheel ships without the slime distribution; staleness admission
    # only runs inside a live bridge, where slime is installed.
    from reef.train.slime_backend.reef_adapters.runtime_load_id import RuntimeLoadId

    try:
        serving = RuntimeLoadId.parse(serving_runtime_load_id)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"cannot classify staleness from serving runtime load ID {serving_runtime_load_id!r}"
        ) from exc
    if str(serving) != serving_runtime_load_id:
        raise RuntimeError(f"cannot classify staleness from non-canonical serving version {serving_runtime_load_id!r}")
    producing_groups = _admission_runtime_load_id_groups(payload)
    producing_versions = [version for group in producing_groups for version in group]

    lags: list[int] = []
    sample_lags: list[int] = []

    def drop(reason: str) -> _StalenessDecision:
        return _stale_drop_decision(
            payload,
            serving_runtime_load_id=serving_runtime_load_id,
            producing_runtime_load_ids=producing_versions,
            reason=reason,
            policy_lags=lags,
        )

    for group in producing_groups:
        group_lags: list[int] = []
        previous_sequence: int | None = None
        for value in group:
            if not isinstance(value, str) or not value:
                return drop("missing_producing_runtime_load_id")
            try:
                producing = RuntimeLoadId.parse(value)
            except (TypeError, ValueError):
                return drop("malformed_producing_runtime_load_id")
            if str(producing) != value:
                return drop("malformed_producing_runtime_load_id")
            if producing.incarnation != serving.incarnation:
                return drop("cross_incarnation")
            if previous_sequence is not None and producing.sequence <= previous_sequence:
                return drop("non_monotonic_producing_runtime_load_ids")
            previous_sequence = producing.sequence
            lag = serving.sequence - producing.sequence
            lags.append(lag)
            group_lags.append(lag)
            if lag < 0:
                return drop("future_producing_runtime_load_id")
            if lag > max_staleness:
                return drop("policy_lag_exceeded")
        sample_lags.append(max(group_lags))
    return _StalenessDecision(
        action="admit",
        metrics={
            "staleness/samples_fresh": sum(lag == 0 for lag in sample_lags),
            "staleness/samples_admitted_stale": sum(lag > 0 for lag in sample_lags),
        },
    )


def _scenario_staleness_admission(
    payload: Mapping[str, Any],
    *,
    scenario: str,
    history: ScenarioHistory,
    serving_runtime_load_id: str,
    max_staleness: int,
) -> _StalenessDecision:
    """Bounded admission against one scenario's own publication history.

    The engine's runtime load ID advances on every scenario's publication, so
    the global sequence gap overstates this scenario's staleness. A sample's
    lag is the number of *this* scenario's publications that postdate the
    version its tokens were produced under.
    """
    from reef.train.slime_backend.reef_adapters.runtime_load_id import RuntimeLoadId

    if _uses_staleness_admission(payload):
        producing_groups = _admission_runtime_load_id_groups(payload)
    else:
        expected = payload.get("expected_runtime_load_id")
        producing_groups = [[expected]]
    producing_versions = [version for group in producing_groups for version in group]
    lags: list[int] = []
    sample_lags: list[int] = []

    def drop(reason: str) -> _StalenessDecision:
        return _stale_drop_decision(
            payload,
            serving_runtime_load_id=serving_runtime_load_id,
            producing_runtime_load_ids=producing_versions,
            reason=reason,
            policy_lags=lags,
        )

    for group in producing_groups:
        group_lags: list[int] = []
        for value in group:
            if not isinstance(value, str) or not value:
                return drop("missing_producing_runtime_load_id")
            try:
                producing = RuntimeLoadId.parse(value)
            except (TypeError, ValueError):
                return drop("malformed_producing_runtime_load_id")
            lag = history.lag(scenario, producing)
            if lag is None:
                return drop("cross_incarnation")
            lags.append(lag)
            group_lags.append(lag)
            if lag > max_staleness:
                return drop("policy_lag_exceeded")
        sample_lags.append(max(group_lags))
    return _StalenessDecision(
        action="admit",
        metrics={
            "staleness/samples_fresh": sum(lag == 0 for lag in sample_lags),
            "staleness/samples_admitted_stale": sum(lag > 0 for lag in sample_lags),
            "staleness/scenario": scenario,
        },
    )


def create_placement_groups(args):
    """Load Slime's heavyweight placement-group module only in the driver."""
    from slime.ray.placement_group import create_placement_groups as implementation

    return implementation(args)


def create_rollout_manager(args, placement_group, *, serving: Executor | None = None):
    from reef.train.slime_backend.reef_adapters.rollout.manager import create_rollout_manager as implementation

    return implementation(args, placement_group, serving=serving)


def create_train_groups(args, placement_groups, rollout_manager):
    from reef.train.slime_backend.reef_adapters.megatron.train_actor import ReefMegatronTrainRayActor
    from reef.train.slime_backend.reef_adapters.train_groups import create_train_groups as implementation

    return implementation(
        args,
        placement_groups,
        rollout_manager,
        actor_cls=ReefMegatronTrainRayActor,
    )


class _NullAlgorithm(SlimeAlgorithm):
    """Stateless no-op algorithm for bridges started without a loss family."""

    loss_family = ""  # type: ignore[assignment]
    loss_type = ""

    def validate_specific_args(self, args, source):
        pass


class _RolloutAdapterEngine:
    """The rollout engines as one :class:`~reef.runtime.adapter_residency.AdapterEngine`.

    Loads are whatever publication the bridge hands over as the payload (a
    callable that pushes the adapter through the trainer's transport);
    unloads go to every engine directly. An unload any engine refuses raises,
    so the residency manager records the slot as leaked instead of assuming
    it is free.
    """

    def __init__(self, rollout_manager: Any) -> None:
        self._rollout_manager = rollout_manager

    def load_adapter(self, name: str, payload: Any) -> None:
        if not callable(payload):
            raise TypeError(
                f"adapter {name!r} needs a publication callable to load from, got {type(payload).__name__}"
            )
        payload()

    def unload_adapter(self, name: str) -> None:
        manager = RayExecutor.from_workers([self._rollout_manager])
        engines, *_ = manager.rpc(0, "get_updatable_engines_and_lock", timeout=_TRAIN_RPC_TIMEOUT_S)
        results = RayExecutor.from_workers(engines).collective_rpc(
            "unload_lora_adapter", kwargs={"lora_name": name}, timeout=_TRAIN_RPC_TIMEOUT_S
        )
        for result in results:
            if result is not None and (not isinstance(result, Mapping) or result.get("success") is not True):
                raise RuntimeError(f"engine kept adapter {name!r}: {result!r}")


#: What a training step invalidates when the base model is frozen. SGLang's
#: release and resume sides take the same tag names, and its resume removes
#: each tag from the released set, so a tag left out here must also be left
#: out of the matching onload.
_KV_AND_GRAPH_TAGS = ("kv_cache", "cuda_graph")


class TrainBridgeActorImpl:
    """Named actor holding a slime ``RayTrainGroup`` for a remote reef runtime.

    Methods mirror what Reef's ``RayTrainGroupHandle`` needs. They run in one
    bridge actor process, where the plain ``RayTrainGroup`` wrapper remains
    local while its worker actor handles stay in Ray.
    """

    def __init__(
        self,
        actor_group,
        rollout_manager,
        *,
        save_hf_template: str | None,
        start_rollout_id: int = 0,
        storage_config: RetentionConfig | None = None,
        megatron_save_root: str | None = None,
        critic_save_root: str | None = None,
        source_hf: str | None = None,
        source_megatron: str | None = None,
        colocate: bool = False,
        lora: bool = False,
        adapter_capacity: int | None = None,
        keep_lora_base_resident: bool = False,
        critic_group=None,
        critic_steps_per_actor: int | None = None,
        critic_only_steps: int = 0,
        loss_family: str | None = None,
        loss_family_config: object | None = None,
        loss_runtime: SlimeAlgorithm | None = None,
    ) -> None:
        self._group = actor_group
        self._critic_group = critic_group
        # The critic checkpoints only when it has its own root: without one its
        # saves would land in the actor's Megatron tree (path collision), so
        # the bridge falls back to the historical save-actor-only behavior.
        self._critic_save_root = critic_save_root if critic_group is not None else None
        self._rollout_manager = rollout_manager
        self._manager_executor = RayExecutor.from_workers([rollout_manager])
        self._save_hf_template = save_hf_template
        self._colocate = colocate
        # A LoRA deployment serves one adapter per scenario: the scenarios
        # time-slice the group's one adapter slot, each publishes under its
        # own scenario-qualified versioned name, and ScenarioHistory records
        # the per-scenario publications the engine-global runtime load ID
        # cannot express. Reporting it in health is what lets the serving side
        # address each scenario's adapter.
        self._lora = lora
        self._history = (
            ScenarioHistory(history_path(save_hf_template)) if lora and save_hf_template is not None else None
        )
        # One engine, one accounting point for the adapters it holds:
        # publication makes room through it, restart recovery reloads through
        # it, and its status is what the serving side reports.
        self._residency = AdapterResidencyManager(adapter_capacity) if lora else None
        self._adapter_engine = _RolloutAdapterEngine(rollout_manager) if lora else None
        # A LoRA run never rewrites the base, so releasing it copies identical
        # bytes to the host and back on every step. Opt in and the training
        # step releases only what it invalidates. Whether the base can stay
        # resident is a memory question, so this stays off by default.
        self._release_tags = _KV_AND_GRAPH_TAGS if (lora and colocate and keep_lora_base_resident) else None
        self._generation_paused = False
        if loss_runtime is not None:
            self._algo = loss_runtime
        elif loss_family is not None:
            self._algo = resolve_loss_family(loss_family).bind(
                loss_family_config,
                critic_steps_per_actor=critic_steps_per_actor,
                critic_only_steps=critic_only_steps,
            )
        else:
            self._algo = _NullAlgorithm()
        self._publication = TrainingPublication(
            marker_path(save_hf_template) if save_hf_template is not None else None,
            _SlimeWeightPublisher(self),
        )
        self._execution = TrainingExecution(
            marker_path(save_hf_template) if save_hf_template is not None else None,
            _SlimeTrainingBackend(self),
            self._publication.state,
        )
        self._closed = False
        self._completed_train_steps = 0
        self._last_train_rollout_id: int | None = None
        self._last_train_metrics: dict[str, Any] = {}
        self._operation_lock = Lock()
        self._next_rollout_id = start_rollout_id
        self._storage = (
            CheckpointStorage(
                storage_config,
                hf_template=save_hf_template,
                megatron_root=megatron_save_root,
                critic_root=self._critic_save_root,
                source_hf=source_hf,
                source_megatron=source_megatron,
                lora=lora,
            )
            if storage_config is not None and save_hf_template is not None and megatron_save_root is not None
            else None
        )
        marker = self._recover_marker() if self._save_hf_template is not None else None
        marker_status = None if marker is None else str(marker["status"])
        if marker_status == "UPDATING_WEIGHTS":
            # The previous fan-out may have updated only some engines. Recover
            # dead actors first, keep every engine paused, and force a complete
            # tensor transfer from the durable checkpoint-backed actor state.
            self._manager_call("recover_updatable_engines")
        if marker_status == "REJECTING":
            if marker is None:
                raise RuntimeError("REJECTING marker status has no marker payload")
            self._publication.reject(str(marker["job_id"]))
            marker["status"] = marker_status = "REJECTED"
        self._inference_url = self._manager_call("inference_url")
        versions = self._manager_call("get_runtime_load_ids")
        if not versions or (marker_status != "UPDATING_WEIGHTS" and len({str(version) for version in versions}) != 1):
            raise RuntimeError(f"serving engines disagree at bridge startup: {versions!r}")
        # An UPDATING_WEIGHTS marker explicitly means this observation may be mixed.
        # It is only a temporary seed; the forced full publication below must
        # converge every engine before construction succeeds.
        self._runtime_load_id = str(versions[0])
        recovered_runtime_load_id = None
        if marker_status in {"READY_TO_COMMIT", "HEAD_COMMITTED", "COMPLETE"}:
            if marker is None:
                raise RuntimeError(f"{marker_status} marker status has no marker payload")
            recovered_runtime_load_id = str(marker["runtime_load_id"])
        if recovered_runtime_load_id is not None:
            # The paired checkpoint contains exactly the weights that were
            # published under this token before the restart. Seed every
            # updater with its predecessor so the mandatory startup publish
            # recreates the same serving identity and the durable Reef head
            # remains tied to the correct serving version.
            self._group.restore_runtime_load_id_for_republication(recovered_runtime_load_id)
        if self._history is not None:
            self._recover_scenario_adapters(marker)
        self._publication.prepare_recovery(marker)
        if self._save_hf_template is not None and marker_status != "REJECTED" and not (self._lora and marker is None):
            # The Megatron checkpoint can be newer than the HF checkpoint used
            # to boot SGLang. Publish actor weights before construction returns
            # so the first Reef inference uses the actual training version. A
            # LoRA bridge that never trained has nothing to publish: the frozen
            # base SGLang booted from is exactly what every fresh adapter
            # computes, and the history replay above restored trained ones.
            self._runtime_load_id = self._update_serving(
                force_full=marker is not None,
                scenario=self._marker_scenario(marker),
            )
        elif self._lora and marker is None:
            # Nothing to publish, but the engines still need Reef's canonical
            # version token (they boot with SGLang's "default"), and colocated
            # engines boot released: give them their weights and KV back
            # before the first request.
            if self._colocate:
                self._manager_call("onload_weights")
                self._manager_call("onload_kv")
            self._runtime_load_id = str(self._group.sync_serving_runtime_load_id())
            observed = [str(value) for value in self._manager_call("get_runtime_load_ids")]
            if not observed or set(observed) != {self._runtime_load_id}:
                raise RuntimeError(f"serving engines disagree after version sync: {observed!r}")
        self._publication.finish_recovery(marker, self._runtime_load_id)

    @property
    def _phase(self) -> str:
        return self._publication.phase

    @_phase.setter
    def _phase(self, phase: str) -> None:
        self._publication.phase = phase

    def _recover_scenario_adapters(self, marker: Mapping[str, Any] | None) -> None:
        """Re-register every scenario's committed adapter after a restart.

        The Megatron checkpoint restores only the slot's last occupant; the
        other scenarios come back from their persisted slot snapshots. Each
        is loaded under the name its last publication recorded, so Reef's
        routing for that scenario keeps resolving. The marker's scenario is
        activated last: the regular startup republication then publishes it
        under the recovered runtime load ID.
        """
        history = self._require_history()
        residency = self._require_residency()
        active = marker.get("scenario") if marker is not None else None
        pending = [
            (scenario, adapter)
            for scenario in history.scenarios
            if (adapter := history.adapter(scenario)) is not None and scenario != active
        ]
        if not pending and active is None:
            return
        self._pause_generation()
        for scenario, adapter in pending:
            _, version = parse_adapter_name(adapter)
            residency.activate(
                scenario,
                version,
                self._adapter_engine,
                payload=lambda scenario=scenario, adapter=adapter: self._group.publish_adapter(scenario, adapter),
            )
        if active is not None:
            self._group.activate_scenario(active)

    def _require_history(self) -> ScenarioHistory:
        """The per-scenario history; only LoRA runs with a checkpoint save path keep one."""
        if self._history is None:
            raise RuntimeError("scenario bookkeeping requires LoRA training with a checkpoint save path")
        return self._history

    def _require_residency(self) -> AdapterResidencyManager:
        if self._residency is None:
            raise RuntimeError("adapter residency requires a LoRA bridge")
        return self._residency

    def _marker_scenario(self, marker: Mapping[str, Any] | None) -> str | None:
        """The scenario a marker's publication belongs to, when the bridge trains per scenario."""
        if self._history is None or marker is None:
            return None
        scenario = marker.get("scenario")
        return str(scenario) if isinstance(scenario, str) and scenario else None

    def _job_scenario(self, payload: Mapping[str, Any]) -> str | None:
        scenario = payload.get("scenario")
        if self._history is None:
            return None
        if not isinstance(scenario, str) or not scenario:
            raise ValueError("per-scenario LoRA training jobs must name their scenario")
        return scenario

    def shutdown(self) -> None:
        """Release training and rollout workers before retiring their bridge."""
        with self._operation_lock:
            if self._closed:
                return
            self._closed = True
            self._phase = "stopped"
            errors = []
            for group in (self._critic_group, self._group):
                if group is not None:
                    try:
                        group.release()
                    except Exception as exc:
                        errors.append(exc)
            try:
                self._manager_executor.rpc(0, "dispose", timeout=60)
            except Exception as exc:
                errors.append(exc)
            finally:
                RayExecutor.from_workers([self._rollout_manager], owned=True).shutdown()
            if errors:
                raise errors[0]

    def health(self) -> dict[str, Any]:
        """Return a lightweight liveness marker for container health checks."""
        training_job: dict[str, Any] = {
            "deferred_weight_update": self._save_hf_template is not None,
            "status": "COMPLETE" if self._save_hf_template is None else "IDLE",
        }
        if self._save_hf_template is not None and (marker := read_marker(self._marker_path())) is not None:
            training_job.update(
                status=marker["status"],
                training_job_id=marker["job_id"],
                # Reef reasons in scenario steps; in per-scenario mode the
                # marker's rollout id is the bridge-global checkpoint index.
                rollout_id=marker.get("scenario_step", marker["rollout_id"]),
                runtime_load_id=marker.get("runtime_load_id"),
                commit_acknowledged=marker.get("commit_acknowledged", False),
            )
            if "scenario" in marker:
                training_job["scenario"] = marker["scenario"]
        ok = self._phase not in {"training_failed", "checkpoint_failed", "weight_sync_failed", "stopped"}
        return {
            "ok": ok,
            # A publication failure with a durable UPDATING_WEIGHTS marker is
            # replayable in place: ``update_serving_weights`` recovers the
            # engines and republishes from the checkpoint. The bridge decides
            # which failures are retryable so callers never re-derive it from
            # phase and marker.
            "recoverable": not ok
            and self._phase == "weight_sync_failed"
            and training_job.get("status") == "UPDATING_WEIGHTS",
            "start_rollout_id": self._next_rollout_id,
            "phase": self._phase,
            "colocate": self._colocate,
            # Where the serving engines answer; Reef dials this when the
            # deployment leaves ``reef.inference_url`` unset.
            "inference_url": self._inference_url,
            "lora_adapter": None,
            "lora_mode": "scenario" if self._lora else None,
            "lora_adapters": {} if self._history is None else self._history.status(),
            "adapter_residency": None if self._residency is None else self._residency.status(),
            "completed_train_steps": self._completed_train_steps,
            "last_train_rollout_id": self._last_train_rollout_id,
            "last_train_metrics": dict(self._last_train_metrics),
            "training_job": training_job,
        }

    def start_rollout_id(self) -> int:
        return self._next_rollout_id

    def republish_serving(self) -> str:
        """Recover serving actors and republish unchanged weights in place.

        This path is for an inference-engine replacement, not a training step.
        Keep the current token because the checkpoint/model tensors have not
        changed; the next optimizer-backed publication advances it normally.
        """
        with self._operation_lock:
            return self._publication.republish(self._runtime_load_id)

    def prepare_training_step(
        self,
        batch,
        step_preparer: str,
        algorithm_state: Mapping[str, Any],
    ) -> PreparedTrainingStep:
        """Prepare a framework-neutral Reef batch with Slime-owned logic."""
        prepared = prepare_slime_step(batch, step_preparer, algorithm_state)
        if prepared.payload is not None:
            self._algo.validate_payload(prepared.payload)
        return prepared

    def execute_training_job(self, payload: Mapping[str, Any]) -> TrainingJobResult:
        """Delegate job replay, train ordering and checkpoint recording to Reef."""
        if self._save_hf_template is None:
            raise RuntimeError("slime args.save_hf is not set")
        with self._operation_lock:
            return self._execution.execute(payload)

    def update_serving_weights(self, training_job_id: str) -> TrainingJobResult:
        """Delegate durable publication ordering to Reef's shared coordinator."""
        with self._operation_lock:
            publication = self._publication.publish(training_job_id)
            marker = publication.marker
            if publication.published:
                rollout_id = int(marker["rollout_id"])
                self._next_rollout_id = max(self._next_rollout_id, rollout_id + 1)
                self._completed_train_steps += 1
                self._last_train_rollout_id = rollout_id
                recorded_train_metrics = marker.get("train_metrics")
                self._last_train_metrics = (
                    dict(recorded_train_metrics) if isinstance(recorded_train_metrics, Mapping) else {}
                )
            return marker_result(marker)

    def reject_training_candidate(self, training_job_id: str) -> None:
        """Finish a checkpointed job without changing the serving weights."""
        with self._operation_lock:
            marker = self._publication.reject(training_job_id)
            self._next_rollout_id = max(self._next_rollout_id, int(marker["rollout_id"]) + 1)

    def _restore_incumbent_serving(self) -> None:
        if not self._colocate:
            return
        # Pairs with the training step's offload: resuming a region that was
        # never released fails, because SGLang resumes by removing the tag
        # from the set release added it to.
        if self._release_tags is None:
            self._manager_call("onload_weights")
        self._manager_call("onload_kv")
        self._continue_generation()

    def acknowledge_training_commit(self, training_job_id: str) -> None:
        """Resume requests only through Reef's durable commit gate."""
        with self._operation_lock:
            self._publication.acknowledge(training_job_id)

    def serving_runtime_load_id(self) -> str:
        """Return the last successfully published serving-runtime load ID.

        Failed swaps can consume a backend counter before raising, so this
        caches only completed publications. Reef recovery uses the value to
        reconcile the serving engine with its recovered head.
        """
        if self._runtime_load_id == "0":
            self._runtime_load_id = str(self._get(self._group.async_get_rank0_runtime_load_id()))
        return self._runtime_load_id

    def _update_serving(self, *, force_full: bool = False, scenario: str | None = None) -> str:
        """Publish the group's weights; ``scenario`` names the adapter a LoRA publication belongs to.

        A per-scenario adapter publication loads a new versioned name into
        every engine, so the residency manager frees a slot first (evicting
        the publishing scenario's own current revision when nothing else
        fits: generation is paused, so no request observes the gap) and
        records the published revision afterwards.

        Admission runs before any weight leaves the trainer. A capacity
        rejection therefore means nothing was published and every engine still
        serves what it served, so it must not terminate them — that took down
        scenarios which were never part of the publication (#65). An eviction
        the engine refused is the opposite: its state is uncertain, so the
        terminate-and-recover path stays (#61).
        """
        residency = self._residency if scenario is not None else None
        try:
            self._phase = "publishing"
            if self._colocate and self._release_tags is None:
                self._manager_call("onload_weights")
            if residency is not None and scenario is not None:
                residency.make_room(scenario, self._adapter_engine, supersede=True)
            self._group.update_weights(
                manage_generation=not self._generation_paused,
                force_full=force_full,
            )
            if self._colocate:
                self._manager_call("onload_kv")
            raw_version = str(self._get(self._group.async_get_rank0_runtime_load_id()))
            observed = [str(value) for value in self._manager_call("get_runtime_load_ids")]
            if not observed or set(observed) != {raw_version}:
                raise RuntimeError(f"serving engines disagree after update: {observed!r}")
            if residency is not None and scenario is not None:
                residency.register(scenario, raw_version)
        except AdapterEvictionFailed:
            self._phase = "weight_sync_failed"
            with suppress(Exception):
                self._manager_call("terminate_updatable_engines")
            raise
        except AdapterCapacityExhausted:
            # Admission was refused before any weight left the trainer.
            self._phase = "serving"
            raise
        except BaseException:
            self._phase = "weight_sync_failed"
            with suppress(Exception):
                self._manager_call("terminate_updatable_engines")
            raise
        self._runtime_load_id = raw_version
        return raw_version

    def _manager_call(self, method: str, *args: Any) -> Any:
        return self._manager_executor.rpc(0, method, args=args, timeout=_TRAIN_RPC_TIMEOUT_S)

    def _pause_generation(self, *, reconcile: bool = False) -> None:
        if self._generation_paused and not reconcile:
            return
        self._manager_call("pause_generation_for_update")
        self._generation_paused = True

    def _continue_generation(self) -> None:
        if not self._generation_paused:
            return
        self._manager_call("continue_generation_after_update")
        self._generation_paused = False

    @staticmethod
    def _get(value: Any) -> Any:
        return resolve(value, timeout=_TRAIN_RPC_TIMEOUT_S)

    def _checkpoint_path(self, rollout_id: int) -> str:
        if self._save_hf_template is None:
            raise RuntimeError("slime args.save_hf is not set")
        return self._save_hf_template.format(rollout_id=rollout_id)

    def _marker_path(self) -> Path:
        if self._save_hf_template is None:
            raise RuntimeError("slime args.save_hf is not set")
        return marker_path(self._save_hf_template)

    def _recover_marker(self) -> dict[str, Any] | None:
        marker = self._execution.recover()
        if marker is None:
            return None
        self._next_rollout_id = max(self._next_rollout_id, marker["rollout_id"] + 1)
        return marker


class _SlimeTrainingBackend:
    """Slime admission, scoring, tensorization and checkpoint reservations."""

    def __init__(self, bridge: TrainBridgeActorImpl) -> None:
        self._bridge = bridge

    @contextmanager
    def prepare(
        self,
        payload: Mapping[str, Any],
        *,
        job_id: str,
        rollout_id: int,
        prior_marker: Mapping[str, Any] | None,
    ) -> Iterator[PreparedTrainingJob | TrainingJobResult]:
        bridge = self._bridge
        scenario = bridge._job_scenario(payload)
        scenario_step = rollout_id
        if scenario is not None:
            # Scenario steps are per scenario; the bridge's checkpoint index
            # stays one monotonic sequence across all of them.
            rollout_id = bridge._next_rollout_id
        elif rollout_id != bridge._next_rollout_id:
            raise RuntimeError(f"expected rollout {bridge._next_rollout_id}, got {rollout_id}")
        max_staleness = _max_staleness(payload)
        durable_metrics: dict[str, Any] = {}
        if scenario is not None:
            admission = _scenario_staleness_admission(
                payload,
                scenario=scenario,
                history=bridge._require_history(),
                serving_runtime_load_id=bridge._runtime_load_id,
                max_staleness=max_staleness,
            )
            if admission.action == "drop":
                yield TrainingJobResult(
                    outcome="stale",
                    runtime_load_id=bridge._runtime_load_id,
                    metrics=admission.metrics,
                )
                return
            durable_metrics.update(admission.metrics)
        elif _uses_staleness_admission(payload):
            producing_versions = [version for group in _admission_runtime_load_id_groups(payload) for version in group]
            if payload.get("expected_runtime_load_id") != bridge._runtime_load_id:
                admission = _stale_drop_decision(
                    payload,
                    serving_runtime_load_id=bridge._runtime_load_id,
                    producing_runtime_load_ids=producing_versions,
                    reason="execution_fence_mismatch",
                )
            else:
                admission = _staleness_admission(
                    payload,
                    serving_runtime_load_id=bridge._runtime_load_id,
                    max_staleness=max_staleness,
                )
            if admission.action == "drop":
                yield TrainingJobResult(
                    outcome="stale",
                    runtime_load_id=bridge._runtime_load_id,
                    metrics=admission.metrics,
                )
                return
            durable_metrics.update(admission.metrics)
        elif payload.get("expected_runtime_load_id") != bridge._runtime_load_id:
            yield TrainingJobResult(outcome="stale", runtime_load_id=bridge._runtime_load_id)
            return
        checkpoint = Path(bridge._checkpoint_path(rollout_id))
        if bridge._storage is None and (checkpoint.exists() or checkpoint.is_symlink()):
            raise RuntimeError(f"checkpoint target already exists: {checkpoint}")
        rollout_data = to_slime_rollout_data(dict(payload))
        rollout_versions = rollout_data.get("producing_runtime_load_ids")
        if (
            max_staleness > 0
            and rollout_versions is not None
            and list(rollout_versions) != list(_producing_runtime_load_ids(payload))
        ):
            raise ValueError("loss-family row producing versions do not match the shared training payload")
        bridge._algo.validate_payload(rollout_data)
        context: Any = nullcontext(None)
        if bridge._storage is not None:
            protected = marker_rollouts(prior_marker)
            if bridge._history is not None:
                # Every scenario's latest checkpoint is its restart source.
                protected |= bridge._history.protected_rollouts()
            context = bridge._storage.admit(
                rollout_id=rollout_id,
                active_rollouts=protected,
            )
        with context as storage_plan:
            if storage_plan is not None and storage_plan["blocked"]:
                yield TrainingJobResult(
                    outcome="storage_blocked",
                    storage=storage_plan,
                    runtime_load_id=bridge._runtime_load_id,
                )
                return
            # The teacher is scored before the RUNNING marker so a
            # scoring failure leaves no partial state: the job
            # stays retryable under the same identity.
            algorithm_metrics = bridge._algo.prepare_rollout(rollout_data)
            # RolloutManager owns Slime's DP schedule and
            # object-store transport contract. It returns one Box
            # per DP rank, exactly what the training actors expect.
            packed = bridge._manager_call("prepare_external_train_data", rollout_data)
            yield _SlimePreparedTrainingJob(
                bridge,
                checkpoint=TrainingCheckpoint(rollout_id, checkpoint, scenario, scenario_step if scenario else None),
                job_id=job_id,
                rollout_data=rollout_data,
                packed=packed,
                algorithm_metrics=algorithm_metrics,
                durable_metrics=durable_metrics,
            )


class _SlimePreparedTrainingJob:
    """One prepared Slime step; Reef controls when training and saving run."""

    def __init__(
        self,
        bridge: TrainBridgeActorImpl,
        *,
        checkpoint: TrainingCheckpoint,
        job_id: str,
        rollout_data: dict[str, Any],
        packed: Any,
        algorithm_metrics: Mapping[str, Any],
        durable_metrics: Mapping[str, Any],
    ) -> None:
        self._bridge = bridge
        self._checkpoint = checkpoint
        self._job_id = job_id
        self._rollout_data = rollout_data
        self._packed = packed
        self._algorithm_metrics = algorithm_metrics
        self._durable_metrics = durable_metrics

    @property
    def checkpoint(self) -> TrainingCheckpoint:
        return self._checkpoint

    def train(self) -> TrainingMetrics:
        bridge = self._bridge
        if bridge._colocate:
            # Retract requests before releasing weights/KV/graphs. Reef's
            # publication commit gate decides when those requests can resume.
            bridge._pause_generation()
            bridge._manager_call("offload", bridge._release_tags)
        if self.checkpoint.scenario is not None:
            bridge._group.activate_scenario(self.checkpoint.scenario)
        training = bridge._algo.train(
            self.checkpoint.rollout_id,
            self._packed,
            actor_group=bridge._group,
            critic_group=bridge._critic_group,
            resolve=bridge._get,
        )
        durable_metrics = dict(self._durable_metrics)
        durable_metrics.update(training.durable_metrics)
        durable_metrics.update(bridge._algo.rollout_metrics(self._rollout_data, bridge.serving_runtime_load_id()))
        worker_metrics = dict(bridge._get(bridge._group.async_pop_rank0_metrics()))
        train_metrics = next(
            (dict(result) for result in training.worker_results if isinstance(result, Mapping) and result),
            {},
        )
        train_metrics.update(worker_metrics)
        train_metrics.update(self._algorithm_metrics)
        return TrainingMetrics(training=train_metrics, durable=durable_metrics)

    def save_checkpoint(self) -> None:
        bridge = self._bridge
        rollout_id = self.checkpoint.rollout_id
        bridge._group.save_model(rollout_id, force_sync=True)
        if bridge._critic_save_root is not None:
            # Critic-only warmup also needs paired optimizer recovery; the
            # critic checkpoint never becomes a serving model/HF export.
            bridge._critic_group.save_model(rollout_id, force_sync=True)
        checkpoint = self.checkpoint.path
        if checkpoint.is_symlink() or not checkpoint.is_dir():
            raise RuntimeError(f"checkpoint is missing or unsafe: {checkpoint}")
        if bridge._storage is not None:
            rewards = self._rollout_data["rewards"]
            bridge._storage.complete(self._job_id, rollout_id, reward=math.fsum(rewards) / len(rewards))
        if self.checkpoint.scenario is not None:
            bridge._require_history().record_checkpoint(self.checkpoint.scenario, rollout_id)


class _SlimeWeightPublisher:
    """Slime transport and colocated/LoRA engine operations, without commit policy."""

    def __init__(self, bridge: TrainBridgeActorImpl) -> None:
        self._bridge = bridge

    def recover(self, marker: Mapping[str, Any] | None) -> None:
        bridge = self._bridge
        bridge._manager_call("recover_updatable_engines")
        if bridge._history is not None:
            # Replacement engines boot without adapters. Restore other scenarios
            # before this job's complete transfer, and release dead residency slots.
            bridge._require_residency().reconcile((), bridge._adapter_engine)
            bridge._recover_scenario_adapters(marker)

    def pause(self) -> None:
        # Reassert the owner barrier even if the bridge cached a prior pause;
        # replacement controllers/engines may not have observed that RPC.
        self._bridge._pause_generation(reconcile=True)

    def republish(self, runtime_load_id: str, marker: Mapping[str, Any] | None) -> str:
        bridge = self._bridge
        bridge._group.restore_runtime_load_id_for_republication(runtime_load_id)
        try:
            return bridge._update_serving(force_full=True, scenario=bridge._marker_scenario(marker))
        finally:
            # A failed/mismatched transfer must not replace the retry identity.
            bridge._runtime_load_id = runtime_load_id

    def publish(self, marker: Mapping[str, Any], *, force_full: bool) -> str:
        bridge = self._bridge
        if bridge._history is not None:
            bridge._group.activate_scenario(str(marker["scenario"]))
        published = bridge._update_serving(force_full=force_full, scenario=bridge._marker_scenario(marker))
        if bridge._history is not None:
            from reef.train.slime_backend.reef_adapters.megatron.lora import scenario_adapter_name

            scenario = str(marker["scenario"])
            bridge._history.record_publication(scenario, published, scenario_adapter_name(scenario, published))
        return published

    def resume(self) -> None:
        self._bridge._continue_generation()

    def restore_incumbent(self) -> None:
        self._bridge._restore_incumbent_serving()

    def abort(self) -> None:
        self._bridge._manager_call("terminate_updatable_engines")


# A concurrent health call must remain responsive while a training RPC is
# waiting for all workers.  Keeping the implementation plain also makes the
# payload contract unit-testable without starting a Ray cluster.
# type ignore: ray's remote() overloads do not declare actor-only options
# such as max_concurrency, but the runtime accepts them for classes.
TrainBridgeActor = ray.remote(max_concurrency=64)(TrainBridgeActorImpl)  # type: ignore[call-overload]


@dataclass(frozen=True)
class BridgePreparation:
    """Validated bridge choices, resolved before model resources are allocated."""

    retention: RetentionConfig
    loss_family: str | None
    lora: bool


def prepare_bridge(
    args: Any, *, retention: RetentionConfig | None = None, loss_family: str | None = None
) -> BridgePreparation:
    """Validate and configure training/inference invariants before either starts."""
    retention = retention or RetentionConfig()
    spec = resolve_loss_family(loss_family) if loss_family is not None else None
    validate_bridge_args(args, spec)
    if loss_family is not None and ":" not in loss_family:
        loss_family = loss_family_refs().get(loss_family) or loss_family
    configure_sglang_runtime(args)
    configure_megatron_runtime(args)
    configure_rollout_runtime(args)
    from reef.train.slime_backend.reef_adapters.executors.config import slime_executor_class

    slime_executor_class(getattr(args, "reef_executor_backend", "auto"), role="training")
    from reef.train.slime_backend.reef_adapters.executors.rollout import rollout_executor_class

    rollout_executor_class(args)
    # Imported here, not at module scope: the LoRA module reaches the Megatron
    # stack, and importing the bridge actor must not drag that in (see
    # tests/reef_service/test_dependency_boundaries.py).
    from reef.train.slime_backend.reef_adapters.megatron.lora import megatron_lora_enabled

    lora = megatron_lora_enabled(args)
    prepare_checkpoint_storage(args, retention)
    return BridgePreparation(retention=retention, loss_family=loss_family, lora=lora)


def start_bridge(
    args,
    *,
    retention: RetentionConfig | None = None,
    loss_family: str | None = None,
    loss_family_config: object | None = None,
    actor_name: str = DEFAULT_BRIDGE_ACTOR_NAME,
    namespace: str = DEFAULT_NAMESPACE,
    serving: Executor | None = None,
    placement_groups: Mapping[str, Any] | None = None,
    preparation: BridgePreparation | None = None,
):
    """Boot the slime training stack and publish it as a named bridge actor.

    Run this in the slime driver process. ``args`` is slime's
    ``argparse.Namespace`` (see ``slime.utils.arguments.parse_args``);
    ``num_rollout`` is the number of externally supplied Reef training steps
    and ``save_hf`` is a checkpoint path template. The Reef service connects
    with the same ``actor_name`` and ``namespace``.

    ``serving`` and ``placement_groups`` are supplied together by the deployment
    owner. The training manager borrows both; bridge failure/shutdown releases
    training workers and the batch manager only. Without them, direct callers
    retain the existing combined lifecycle. ``preparation`` is the result of
    ``prepare_bridge`` for these same arguments before resource allocation.
    """
    prepared = preparation or prepare_bridge(args, retention=retention, loss_family=loss_family)
    retention = prepared.retention
    loss_family = prepared.loss_family
    lora = prepared.lora
    colocate = bool(getattr(args, "colocate", False))
    from reef.train.slime_backend.reef_adapters.megatron.lora import lora_engine_slots

    if (serving is None) != (placement_groups is None):
        raise ValueError("supplied inference requires its coordinated placement groups")
    if not ray.is_initialized():
        ray.init(namespace=namespace)
    owns_placement_groups = placement_groups is None
    pgs = create_placement_groups(args) if placement_groups is None else placement_groups
    rollout_manager = None
    actor_group = None
    critic_group = None
    try:
        rollout_manager = (
            create_rollout_manager(args, pgs["rollout"])
            if serving is None
            else create_rollout_manager(args, pgs["rollout"], serving=serving)
        )
        # Loss families that train a value model need the critic actor group;
        # the others discard it. ``args.use_critic`` comes from the explicit
        # --use-critic driver flag (or implicitly from --advantage-estimator ppo),
        # so keeping the group here is what wires the value model into the bridge
        # schedule rather than leaving it uninitialized.
        actor_group, critic_group = create_train_groups(args, pgs, rollout_manager)
        return TrainBridgeActor.options(name=actor_name, namespace=namespace).remote(
            actor_group,
            rollout_manager,
            save_hf_template=args.save_hf,
            start_rollout_id=getattr(args, "start_rollout_id", 0) or 0,
            storage_config=retention,
            megatron_save_root=args.save,
            critic_save_root=getattr(args, "critic_save", None),
            source_hf=getattr(args, "hf_checkpoint", None),
            source_megatron=getattr(args, "load", None),
            colocate=colocate,
            lora=lora,
            adapter_capacity=lora_engine_slots(args) if lora else None,
            keep_lora_base_resident=bool(getattr(args, "keep_lora_base_resident", False)),
            critic_group=critic_group,
            critic_steps_per_actor=getattr(args, "critic_steps_per_actor", None),
            critic_only_steps=getattr(args, "num_critic_only_steps", 0),
            loss_family=loss_family,
            loss_family_config=loss_family_config,
        )
    except BaseException:
        for group in (critic_group, actor_group):
            if group is not None:
                with suppress(Exception):
                    group.release()
        if rollout_manager is not None:
            with suppress(Exception):
                RayExecutor.from_workers([rollout_manager]).rpc(0, "dispose", timeout=60)
            with suppress(Exception):
                RayExecutor.from_workers([rollout_manager], owned=True).shutdown()
        # Supplied reservations belong to the deployment owner.
        if owns_placement_groups:
            with suppress(Exception):
                from ray.util.placement_group import remove_placement_group

                released = set()
                for placement in pgs.values():
                    if placement is not None and placement[0] is not None and placement[0].id not in released:
                        released.add(placement[0].id)
                        remove_placement_group(placement[0])
        raise
