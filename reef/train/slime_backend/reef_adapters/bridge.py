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

import hashlib
import json
import math
import sys
import traceback
from collections.abc import Mapping, Sequence
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Literal

import ray

from reef.core.artifact_ref import parse_weight_version_spans
from reef.runtime.adapter_residency import AdapterResidencyManager
from reef.runtime.base import PreparedTrainingStep, TrainingJobResult
from reef.runtime.names import DEFAULT_ACTOR_NAME, DEFAULT_NAMESPACE
from reef.surface.adapter import parse_adapter_name
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
from reef.train.slime_backend.reef_adapters.training_job.marker import (
    marker_checkpoint_result,
    marker_disposition,
    marker_path,
    marker_result,
    marker_rollouts,
    read_marker,
    transition_marker,
    write_marker,
)
from reef.train.slime_backend.reef_adapters.training_job.scenarios import ScenarioLedger, ledger_path
from reef.train.slime_backend.reef_adapters.training_job.storage import CheckpointStorage, RetentionConfig

DEFAULT_BRIDGE_ACTOR_NAME = DEFAULT_ACTOR_NAME

# One training step (train + checkpoint + publish) legitimately takes hours;
# this bounds a single Ray RPC from the bridge to its workers.
_TRAIN_RPC_TIMEOUT_S = 14_400


@dataclass(frozen=True, slots=True)
class _StalenessDecision:
    action: Literal["admit", "drop"]
    metrics: Mapping[str, Any]


def _max_staleness(payload: Mapping[str, Any]) -> int:
    value = payload.get("max_staleness", 0)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("training job max_staleness must be a non-negative integer")
    return value


def _uses_staleness_admission(payload: Mapping[str, Any]) -> bool:
    """Whether the serving version is an admission fence, not job identity."""
    return _max_staleness(payload) > 0 or "producing_weight_versions" in payload


def _training_job_id(payload: Mapping[str, Any]) -> str:
    identity = dict(payload)
    # Admission policy controls whether a fresh execution may start; changing
    # that policy must not turn a completed logical job into different work.
    identity.pop("max_staleness", None)
    if _uses_staleness_admission(payload):
        # The serving fence is sampled at preparation and can legitimately be
        # newer on a lost-ack replay after this job published. It fences only a
        # fresh execution; sample rows and rollout id are the retry-stable
        # logical job identity.
        identity.pop("expected_weight_version", None)
    encoded = json.dumps(identity, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_agent_record_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
    samples = payload.get("samples")
    if not isinstance(samples, Sequence) or isinstance(samples, str | bytes):
        return ()
    return tuple(
        str(row[0]) for row in samples if isinstance(row, Sequence) and not isinstance(row, str | bytes) and row
    )


def _producing_weight_versions(payload: Mapping[str, Any]) -> Sequence[Any]:
    versions = payload.get("producing_weight_versions")
    if not isinstance(versions, Sequence) or isinstance(versions, str | bytes) or not versions:
        raise ValueError("bounded staleness admission requires producing_weight_versions")
    source_ids = _source_agent_record_ids(payload)
    if len(versions) != len(source_ids):
        raise ValueError(
            "bounded staleness admission requires one producing weight version "
            f"per sample: {len(versions)} versions for {len(source_ids)} samples"
        )
    return versions


def _admission_weight_version_groups(payload: Mapping[str, Any]) -> list[list[Any]]:
    """Return each sample's exact span versions for bounded admission."""
    versions = _producing_weight_versions(payload)
    raw_groups = payload.get("producing_weight_version_spans")
    if raw_groups is None:
        return [[version] for version in versions]
    if not isinstance(raw_groups, Sequence) or isinstance(raw_groups, str | bytes) or len(raw_groups) != len(versions):
        raise ValueError("producing_weight_version_spans must contain one span list per sample")
    samples = payload["samples"]
    groups: list[list[Any]] = []
    for sample_index, (raw_spans, scalar) in enumerate(zip(raw_groups, versions, strict=True)):
        if not raw_spans:
            groups.append([scalar])
            continue
        row = samples[sample_index]
        response_length = len(row[2]) if isinstance(row, Sequence) and len(row) > 2 else None
        spans = parse_weight_version_spans(
            raw_spans,
            field_name=f"producing_weight_version_spans[{sample_index}]",
            response_length=response_length,
        )
        group = [span.weight_version for span in spans]
        span_versions = set(group)
        if scalar is not None and span_versions != {scalar}:
            raise ValueError(f"producing weight version for sample {sample_index} disagrees with its token spans")
        groups.append(group)
    return groups


def _stale_drop_decision(
    payload: Mapping[str, Any],
    *,
    serving_weight_version: str,
    producing_weight_versions: Sequence[Any],
    reason: str,
    policy_lags: Sequence[int] = (),
) -> _StalenessDecision:
    source_ids = _source_agent_record_ids(payload)
    metrics: dict[str, Any] = {
        "staleness/samples_dropped": len(source_ids) or len(producing_weight_versions),
        "staleness/drop_reason": reason,
        "staleness/source_agent_record_ids": list(source_ids),
        "staleness/producing_weight_versions": [
            None if version is None else str(version) for version in producing_weight_versions
        ],
        "staleness/serving_weight_version": serving_weight_version,
    }
    if policy_lags:
        metrics["staleness/drop_policy_lags"] = list(policy_lags)
    return _StalenessDecision(action="drop", metrics=metrics)


def _staleness_admission(
    payload: Mapping[str, Any],
    *,
    serving_weight_version: str,
    max_staleness: int,
) -> _StalenessDecision:
    # The reef wheel ships without the slime distribution; staleness admission
    # only runs inside a live bridge, where slime is installed.
    from reef.train.slime_backend.reef_adapters.weight_version import WeightVersion

    try:
        serving = WeightVersion.parse(serving_weight_version)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"cannot classify staleness from serving weight version {serving_weight_version!r}"
        ) from exc
    if str(serving) != serving_weight_version:
        raise RuntimeError(f"cannot classify staleness from non-canonical serving version {serving_weight_version!r}")
    producing_groups = _admission_weight_version_groups(payload)
    producing_versions = [version for group in producing_groups for version in group]

    lags: list[int] = []
    sample_lags: list[int] = []

    def drop(reason: str) -> _StalenessDecision:
        return _stale_drop_decision(
            payload,
            serving_weight_version=serving_weight_version,
            producing_weight_versions=producing_versions,
            reason=reason,
            policy_lags=lags,
        )

    for group in producing_groups:
        group_lags: list[int] = []
        previous_sequence: int | None = None
        for value in group:
            if not isinstance(value, str) or not value:
                return drop("missing_producing_weight_version")
            try:
                producing = WeightVersion.parse(value)
            except (TypeError, ValueError):
                return drop("malformed_producing_weight_version")
            if str(producing) != value:
                return drop("malformed_producing_weight_version")
            if producing.incarnation != serving.incarnation:
                return drop("cross_incarnation")
            if previous_sequence is not None and producing.sequence <= previous_sequence:
                return drop("non_monotonic_producing_weight_versions")
            previous_sequence = producing.sequence
            lag = serving.sequence - producing.sequence
            lags.append(lag)
            group_lags.append(lag)
            if lag < 0:
                return drop("future_producing_weight_version")
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
    ledger: ScenarioLedger,
    serving_weight_version: str,
    max_staleness: int,
) -> _StalenessDecision:
    """Bounded admission against one scenario's own publication history.

    The engine's weight version advances on every scenario's publication, so
    the global sequence gap overstates this scenario's staleness. A sample's
    lag is the number of *this* scenario's publications that postdate the
    version its tokens were produced under.
    """
    from reef.train.slime_backend.reef_adapters.weight_version import WeightVersion

    if _uses_staleness_admission(payload):
        producing_groups = _admission_weight_version_groups(payload)
    else:
        expected = payload.get("expected_weight_version")
        producing_groups = [[expected]]
    producing_versions = [version for group in producing_groups for version in group]
    lags: list[int] = []
    sample_lags: list[int] = []

    def drop(reason: str) -> _StalenessDecision:
        return _stale_drop_decision(
            payload,
            serving_weight_version=serving_weight_version,
            producing_weight_versions=producing_versions,
            reason=reason,
            policy_lags=lags,
        )

    for group in producing_groups:
        group_lags: list[int] = []
        for value in group:
            if not isinstance(value, str) or not value:
                return drop("missing_producing_weight_version")
            try:
                producing = WeightVersion.parse(value)
            except (TypeError, ValueError):
                return drop("malformed_producing_weight_version")
            lag = ledger.lag(scenario, producing)
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


def create_rollout_manager(args, placement_group):
    from reef.train.slime_backend.reef_adapters.rollout.manager import create_rollout_manager as implementation

    return implementation(args, placement_group)


def create_train_groups(args, placement_groups, rollout_manager):
    from reef.train.slime_backend.reef_adapters.megatron.train_actor import ReefMegatronTrainRayActor
    from reef.train.slime_backend.reef_adapters.ray_train_groups import create_train_groups as implementation

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
        engines, *_ = ray.get(
            self._rollout_manager.get_updatable_engines_and_lock.remote(), timeout=_TRAIN_RPC_TIMEOUT_S
        )
        results = ray.get(
            [engine.unload_lora_adapter.remote(lora_name=name) for engine in engines], timeout=_TRAIN_RPC_TIMEOUT_S
        )
        for result in results:
            if result is not None and (not isinstance(result, Mapping) or result.get("success") is not True):
                raise RuntimeError(f"engine kept adapter {name!r}: {result!r}")


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
        self._save_hf_template = save_hf_template
        self._colocate = colocate
        # A LoRA deployment serves one adapter per scenario: the scenarios
        # time-slice the group's one adapter slot, each publishes under its
        # own scenario-qualified versioned name, and the ledger keeps the
        # per-scenario history the engine-global weight version cannot
        # express. Reporting it in health is what lets the serving side
        # address each scenario's adapter.
        self._lora = lora
        self._ledger = ScenarioLedger(ledger_path(save_hf_template)) if lora and save_hf_template is not None else None
        # One engine, one accounting point for the adapters it holds:
        # publication makes room through it, restart recovery reloads through
        # it, and its status is what the serving side reports.
        self._residency = AdapterResidencyManager(adapter_capacity) if lora else None
        self._adapter_engine = _RolloutAdapterEngine(rollout_manager) if lora else None
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
        self._phase = "serving"
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
            self._restore_incumbent_serving()
            transition_marker(self._marker_path(), marker, "REJECTED")
            marker_status = "REJECTED"
        self._inference_url = self._manager_call("inference_url")
        versions = self._manager_call("get_weight_versions")
        if not versions or (marker_status != "UPDATING_WEIGHTS" and len({str(version) for version in versions}) != 1):
            raise RuntimeError(f"serving engines disagree at bridge startup: {versions!r}")
        # An UPDATING_WEIGHTS marker explicitly means this observation may be mixed.
        # It is only a temporary seed; the forced full publication below must
        # converge every engine before construction succeeds.
        self._weight_version = str(versions[0])
        recovered_weight_version = None
        if marker_status in {"READY_TO_COMMIT", "HEAD_COMMITTED", "COMPLETE"}:
            if marker is None:
                raise RuntimeError(f"{marker_status} marker status has no marker payload")
            recovered_weight_version = str(marker["weight_version"])
        if recovered_weight_version is not None:
            # The paired checkpoint contains exactly the weights that were
            # published under this token before the restart. Seed every
            # updater with its predecessor so the mandatory startup publish
            # recreates the same serving identity and the durable Reef head
            # remains tied to the correct serving version.
            self._group.restore_weight_version_for_republication(recovered_weight_version)
        if self._ledger is not None:
            self._recover_scenario_adapters(marker)
        if marker_status in {"CHECKPOINT", "UPDATING_WEIGHTS", "READY_TO_COMMIT", "HEAD_COMMITTED"}:
            try:
                self._pause_generation()
            except BaseException:
                with suppress(Exception):
                    self._manager_call("terminate_updatable_engines")
                raise
        if marker_status == "CHECKPOINT":
            if marker is None:
                raise RuntimeError("CHECKPOINT marker status has no marker payload")
            transition_marker(self._marker_path(), marker, "UPDATING_WEIGHTS")
        if self._save_hf_template is not None and marker_status != "REJECTED" and not (self._lora and marker is None):
            # The Megatron checkpoint can be newer than the HF checkpoint used
            # to boot SGLang. Publish actor weights before construction returns
            # so the first Reef inference uses the actual training version. A
            # LoRA bridge that never trained has nothing to publish: the frozen
            # base SGLang booted from is exactly what every fresh adapter
            # computes, and the ledger replay above restored trained ones.
            self._weight_version = self._update_serving(
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
            self._weight_version = str(self._group.sync_serving_weight_version())
            observed = [str(value) for value in self._manager_call("get_weight_versions")]
            if not observed or set(observed) != {self._weight_version}:
                raise RuntimeError(f"serving engines disagree after version sync: {observed!r}")
        if recovered_weight_version is not None and self._weight_version != recovered_weight_version:
            raise RuntimeError(
                "checkpoint republication changed weight version "
                f"{recovered_weight_version!r} to {self._weight_version!r}"
            )
        if marker is not None and marker_status in {"CHECKPOINT", "UPDATING_WEIGHTS"}:
            transition_marker(
                self._marker_path(),
                marker,
                "READY_TO_COMMIT",
                weight_version=self._weight_version,
            )
            self._phase = "awaiting_commit"
        elif marker_status == "READY_TO_COMMIT":
            self._phase = "awaiting_commit"
        elif marker is not None and marker_status == "HEAD_COMMITTED":
            self._continue_generation()
            transition_marker(
                self._marker_path(),
                marker,
                "COMPLETE",
                commit_acknowledged=True,
            )
            self._phase = "serving"
        else:
            self._phase = "serving"
            # Per-scenario recovery paused the engines itself; a serving
            # bridge must not leave them paused.
            self._continue_generation()

    def _recover_scenario_adapters(self, marker: dict[str, Any] | None) -> None:
        """Re-register every scenario's committed adapter after a restart.

        The Megatron checkpoint restores only the slot's last occupant; the
        other scenarios come back from their persisted slot snapshots. Each
        is loaded under the name its last publication recorded, so Reef's
        routing for that scenario keeps resolving. The marker's scenario is
        activated last: the regular startup republication then publishes it
        under the recovered weight version.
        """
        ledger = self._require_ledger()
        residency = self._require_residency()
        active = marker.get("scenario") if marker is not None else None
        pending = [
            (scenario, adapter)
            for scenario in ledger.scenarios
            if (adapter := ledger.adapter(scenario)) is not None and scenario != active
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

    def _require_ledger(self) -> ScenarioLedger:
        """The per-scenario ledger; only LoRA runs with a checkpoint save path keep one."""
        if self._ledger is None:
            raise RuntimeError("scenario bookkeeping requires LoRA training with a checkpoint save path")
        return self._ledger

    def _require_residency(self) -> AdapterResidencyManager:
        if self._residency is None:
            raise RuntimeError("adapter residency requires a LoRA bridge")
        return self._residency

    def _marker_scenario(self, marker: Mapping[str, Any] | None) -> str | None:
        """The scenario a marker's publication belongs to, when the bridge trains per scenario."""
        if self._ledger is None or marker is None:
            return None
        scenario = marker.get("scenario")
        return str(scenario) if isinstance(scenario, str) and scenario else None

    def _job_scenario(self, payload: Mapping[str, Any]) -> str | None:
        scenario = payload.get("scenario")
        if self._ledger is None:
            return None
        if not isinstance(scenario, str) or not scenario:
            raise ValueError("per-scenario LoRA training jobs must name their scenario")
        return scenario

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
                weight_version=marker.get("weight_version"),
                commit_acknowledged=marker.get("commit_acknowledged", False),
            )
            if "scenario" in marker:
                training_job["scenario"] = marker["scenario"]
        return {
            "ok": self._phase not in {"training_failed", "checkpoint_failed", "weight_sync_failed"},
            "start_rollout_id": self._next_rollout_id,
            "phase": self._phase,
            "colocate": self._colocate,
            # Where the serving engines answer; Reef dials this when the
            # deployment leaves ``reef.inference_url`` unset.
            "inference_url": self._inference_url,
            "lora_adapter": None,
            "lora_mode": "scenario" if self._lora else None,
            "lora_adapters": {} if self._ledger is None else self._ledger.status(),
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
            expected = self._weight_version
            self._group.restore_weight_version_for_republication(expected)
            marker = read_marker(self._marker_path()) if self._save_hf_template is not None else None
            published = self._update_serving(scenario=self._marker_scenario(marker))
            if published != expected:
                raise RuntimeError(f"serving republication changed weight version {expected!r} to {published!r}")
            self._phase = "serving"
            return published

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
        """Train and checkpoint one idempotent job without updating serving weights."""
        job_id = _training_job_id(payload)
        rollout_id = payload.get("rollout_id")
        if not isinstance(rollout_id, int) or isinstance(rollout_id, bool) or rollout_id < 0:
            raise ValueError("training job rollout_id must be non-negative")
        scenario = self._job_scenario(payload)
        marker_file = self._marker_path()
        with self._operation_lock:
            marker = read_marker(marker_file)
            disposition = marker_disposition(marker, job_id)
            if disposition == "conflict":
                if marker is None:
                    raise RuntimeError("conflicting training disposition has no marker")
                raise RuntimeError(f"training marker is {marker['status']}; operator recovery required")
            if disposition == "replay":
                if marker is None:
                    raise RuntimeError("replayed training disposition has no marker")
                if marker["status"] == "COMPLETE":
                    return marker_result(marker)
                return marker_checkpoint_result(marker)
            # Worker and loss-family telemetry is captured in the marker and
            # returned through TrainingJobResult.metrics. The generic Reef
            # training observer can then track any backend without knowing
            # Slime's worker topology. A CHECKPOINT resume reuses the recorded
            # metrics instead of draining workers again.
            train_metrics: dict[str, Any] = {}
            if disposition == "fresh":
                step = self._run_train_step(payload, job_id, rollout_id, prior_marker=marker, scenario=scenario)
                if isinstance(step, TrainingJobResult):
                    return step
                marker, train_metrics, _ = step
            if marker is None:
                raise RuntimeError("training job produced no checkpoint marker")
            if train_metrics:
                marker["train_metrics"] = dict(train_metrics)
                write_marker(marker_file, marker)
            return marker_checkpoint_result(marker)

    def update_serving_weights(self, training_job_id: str) -> TrainingJobResult:
        """Publish one checkpointed job while keeping generation paused."""
        if not training_job_id:
            raise ValueError("training_job_id must be non-empty")
        with self._operation_lock:
            marker = read_marker(self._marker_path())
            if marker is None or marker["job_id"] != training_job_id:
                raise RuntimeError(f"unknown training job {training_job_id!r}")
            status = marker["status"]
            if status in {"READY_TO_COMMIT", "HEAD_COMMITTED", "COMPLETE"}:
                return marker_result(marker)
            if status not in {"CHECKPOINT", "UPDATING_WEIGHTS"}:
                raise RuntimeError(f"training job is {status}; operator recovery required")

            recovering = status == "UPDATING_WEIGHTS"
            try:
                if recovering:
                    self._manager_call("recover_updatable_engines")
                self._pause_generation()
                if self._ledger is not None:
                    # A restart between checkpoint and publication may have
                    # left another scenario in the slot.
                    self._group.activate_scenario(str(marker["scenario"]))
                if not recovering:
                    # Persist uncertainty only after every engine has crossed
                    # the pause barrier. A pause failure has not changed any
                    # weights and remains safely replayable from CHECKPOINT.
                    transition_marker(self._marker_path(), marker, "UPDATING_WEIGHTS")
                published = self._update_serving(force_full=recovering, scenario=self._marker_scenario(marker))
                if self._ledger is not None:
                    from reef.train.slime_backend.reef_adapters.megatron.lora import scenario_adapter_name

                    scenario = str(marker["scenario"])
                    self._ledger.record_publication(scenario, published, scenario_adapter_name(scenario, published))
                transition_marker(
                    self._marker_path(),
                    marker,
                    "READY_TO_COMMIT",
                    weight_version=published,
                )
                self._phase = "awaiting_commit"
            except BaseException:
                self._phase = "weight_sync_failed"
                with suppress(Exception):
                    self._manager_call("terminate_updatable_engines")
                traceback.print_exc(file=sys.stderr)
                raise

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
        if not training_job_id:
            raise ValueError("training_job_id must be non-empty")
        with self._operation_lock:
            marker = read_marker(self._marker_path())
            if marker is None or marker["job_id"] != training_job_id:
                raise RuntimeError(f"unknown training job {training_job_id!r}")
            status = marker["status"]
            if status == "REJECTED":
                return
            if status == "CHECKPOINT":
                transition_marker(self._marker_path(), marker, "REJECTING")
            elif status != "REJECTING":
                raise RuntimeError(f"cannot reject training job {training_job_id!r} from {status}")
            self._restore_incumbent_serving()
            transition_marker(self._marker_path(), marker, "REJECTED")
            rollout_id = int(marker["rollout_id"])
            self._next_rollout_id = max(self._next_rollout_id, rollout_id + 1)
            self._phase = "serving"

    def _restore_incumbent_serving(self) -> None:
        if not self._colocate:
            return
        self._manager_call("onload_weights")
        self._manager_call("onload_kv")
        self._continue_generation()

    def acknowledge_training_commit(self, training_job_id: str) -> None:
        """Resume paused requests only after Reef durably commits the new head."""
        if not training_job_id:
            raise ValueError("training_job_id must be non-empty")
        with self._operation_lock:
            marker = read_marker(self._marker_path())
            if marker is None or marker["job_id"] != training_job_id:
                raise RuntimeError(f"unknown training job {training_job_id!r}")
            if marker["status"] == "COMPLETE":
                if marker.get("commit_acknowledged") is not True:
                    marker["commit_acknowledged"] = True
                    write_marker(self._marker_path(), marker)
                return
            if marker["status"] == "READY_TO_COMMIT":
                transition_marker(
                    self._marker_path(),
                    marker,
                    "HEAD_COMMITTED",
                    commit_acknowledged=True,
                )
            if marker["status"] != "HEAD_COMMITTED":
                raise RuntimeError(f"cannot acknowledge training job {training_job_id!r} from {marker['status']}")
            self._continue_generation()
            transition_marker(
                self._marker_path(),
                marker,
                "COMPLETE",
                commit_acknowledged=True,
            )
            self._phase = "serving"

    def _run_train_step(
        self,
        payload: Mapping[str, Any],
        job_id: str,
        rollout_id: int,
        *,
        prior_marker: dict[str, Any] | None,
        scenario: str | None = None,
    ) -> TrainingJobResult | tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Run train + checkpoint for a fresh job, up to the CHECKPOINT marker.

        Returns an early ``TrainingJobResult`` (stale or storage-blocked) or
        the CHECKPOINT-state marker with the drained worker metrics and the
        loss family's durable telemetry.
        """
        scenario_step = rollout_id
        if scenario is not None:
            # Scenario steps are per scenario; the bridge's checkpoint index
            # stays one monotonic sequence across all of them.
            rollout_id = self._next_rollout_id
        elif rollout_id != self._next_rollout_id:
            raise RuntimeError(f"expected rollout {self._next_rollout_id}, got {rollout_id}")
        max_staleness = _max_staleness(payload)
        train_metrics: dict[str, Any] = {}
        durable_metrics: dict[str, Any] = {}
        if scenario is not None:
            admission = _scenario_staleness_admission(
                payload,
                scenario=scenario,
                ledger=self._require_ledger(),
                serving_weight_version=self._weight_version,
                max_staleness=max_staleness,
            )
            if admission.action == "drop":
                return TrainingJobResult(
                    outcome="stale",
                    weight_version=self._weight_version,
                    metrics=admission.metrics,
                )
            durable_metrics.update(admission.metrics)
        elif _uses_staleness_admission(payload):
            producing_versions = [version for group in _admission_weight_version_groups(payload) for version in group]
            if payload.get("expected_weight_version") != self._weight_version:
                admission = _stale_drop_decision(
                    payload,
                    serving_weight_version=self._weight_version,
                    producing_weight_versions=producing_versions,
                    reason="execution_fence_mismatch",
                )
            else:
                admission = _staleness_admission(
                    payload,
                    serving_weight_version=self._weight_version,
                    max_staleness=max_staleness,
                )
            if admission.action == "drop":
                return TrainingJobResult(
                    outcome="stale",
                    weight_version=self._weight_version,
                    metrics=admission.metrics,
                )
            durable_metrics.update(admission.metrics)
        elif payload.get("expected_weight_version") != self._weight_version:
            return TrainingJobResult(outcome="stale", weight_version=self._weight_version)
        checkpoint = Path(self._checkpoint_path(rollout_id))
        if self._storage is None and (checkpoint.exists() or checkpoint.is_symlink()):
            raise RuntimeError(f"checkpoint target already exists: {checkpoint}")
        rollout_data = to_slime_rollout_data(dict(payload))
        rollout_versions = rollout_data.get("producing_weight_versions")
        if (
            max_staleness > 0
            and rollout_versions is not None
            and list(rollout_versions) != list(_producing_weight_versions(payload))
        ):
            raise ValueError("loss-family row provenance does not match the shared training payload")
        self._algo.validate_payload(rollout_data)
        context: Any = nullcontext(None)
        if self._storage is not None:
            protected = marker_rollouts(prior_marker)
            if self._ledger is not None:
                # Every scenario's latest checkpoint is its restart source.
                protected |= self._ledger.protected_rollouts()
            context = self._storage.admit(
                rollout_id=rollout_id,
                active_rollouts=protected,
            )
        with context as storage_plan:
            if storage_plan is not None and storage_plan["blocked"]:
                return TrainingJobResult(
                    outcome="storage_blocked",
                    storage=storage_plan,
                    weight_version=self._weight_version,
                )
            # The teacher is scored before the RUNNING marker so a
            # scoring failure leaves no partial state: the job
            # stays retryable under the same identity.
            algorithm_metrics = self._algo.prepare_rollout(rollout_data)
            # RolloutManager owns Slime's DP schedule and
            # object-store transport contract. It returns one Box
            # per DP rank, exactly what the training actors expect.
            packed = self._get(self._rollout_manager.prepare_external_train_data.remote(rollout_data))
            marker = {
                "status": "RUNNING",
                "job_id": job_id,
                "rollout_id": rollout_id,
            }
            if scenario is not None:
                marker.update(scenario=scenario, scenario_step=scenario_step)
            write_marker(self._marker_path(), marker)
            try:
                if self._colocate:
                    # Retract active requests before SGLang releases its KV,
                    # weights, and CUDA graphs. Their CPU request state stays
                    # queued and is re-prefilled with the committed model when
                    # generation resumes.
                    self._pause_generation()
                    self._manager_call("offload")
                if scenario is not None:
                    # Put this scenario's adapter and optimizer state into the
                    # slot; a first-time scenario starts from the pristine one.
                    self._group.activate_scenario(scenario)
                self._phase = "training"
                training = self._algo.train(
                    rollout_id,
                    packed,
                    actor_group=self._group,
                    critic_group=self._critic_group,
                    resolve=self._get,
                )
                train_results = training.worker_results
                durable_metrics.update(training.durable_metrics)
                durable_metrics.update(self._algo.provenance_metrics(rollout_data, self.serving_weight_version()))
                worker_metrics = dict(self._get(self._group.async_pop_rank0_metrics()))
                train_metrics = next(
                    (dict(result) for result in train_results if isinstance(result, Mapping) and result),
                    {},
                )
                train_metrics.update(worker_metrics)
                train_metrics.update(algorithm_metrics)
                self._phase = "checkpointing"
                self._group.save_model(rollout_id, force_sync=True)
                if self._critic_save_root is not None:
                    # Every commit — critic-only warmup included — persists the
                    # critic's weights and optimizer alongside the actor pair;
                    # otherwise the value head cold-starts on every reboot
                    # (SAO's stated cold-start concern). No HF export: the
                    # critic never serves.
                    self._critic_group.save_model(rollout_id, force_sync=True)
                if checkpoint.is_symlink() or not checkpoint.is_dir():
                    raise RuntimeError(f"checkpoint is missing or unsafe: {checkpoint}")
                if self._storage is not None:
                    rewards = rollout_data["rewards"]
                    self._storage.complete(
                        job_id,
                        rollout_id,
                        reward=math.fsum(rewards) / len(rewards),
                    )
                if scenario is not None:
                    self._require_ledger().record_checkpoint(scenario, rollout_id)
                if durable_metrics:
                    marker["metrics"] = dict(durable_metrics)
                transition_marker(
                    self._marker_path(),
                    marker,
                    "CHECKPOINT",
                    checkpoint_path=str(checkpoint),
                )
            except BaseException:
                self._phase = "training_failed" if self._phase == "training" else "checkpoint_failed"
                # The caller's error slot is overwritten by later retries;
                # keep a durable record of what actually failed.
                traceback.print_exc(file=sys.stderr)
                raise
        return marker, train_metrics, durable_metrics

    def serving_weight_version(self) -> str:
        """Return the last successfully published serving-weight version.

        Failed swaps can consume a backend counter before raising, so this
        caches only completed publications. Reef recovery uses the value to
        reconcile the serving engine with its recovered head.
        """
        if self._weight_version == "0":
            self._weight_version = str(self._get(self._group.async_get_rank0_weight_version()))
        return self._weight_version

    def _update_serving(self, *, force_full: bool = False, scenario: str | None = None) -> str:
        """Publish the group's weights; ``scenario`` names the adapter a LoRA publication belongs to.

        A per-scenario adapter publication loads a new versioned name into
        every engine, so the residency manager frees a slot first (evicting
        the publishing scenario's own current revision when nothing else
        fits: generation is paused, so no request observes the gap) and
        records the published revision afterwards.
        """
        residency = self._residency if scenario is not None else None
        try:
            self._phase = "publishing"
            if self._colocate:
                self._manager_call("onload_weights")
            if residency is not None and scenario is not None:
                residency.make_room(scenario, self._adapter_engine, supersede=True)
            self._group.update_weights(
                manage_generation=not self._generation_paused,
                force_full=force_full,
            )
            if self._colocate:
                self._manager_call("onload_kv")
            raw_version = str(self._get(self._group.async_get_rank0_weight_version()))
            observed = [str(value) for value in self._manager_call("get_weight_versions")]
            if not observed or set(observed) != {raw_version}:
                raise RuntimeError(f"serving engines disagree after update: {observed!r}")
            if residency is not None and scenario is not None:
                residency.register(scenario, raw_version)
        except BaseException:
            self._phase = "weight_sync_failed"
            with suppress(Exception):
                self._manager_call("terminate_updatable_engines")
            raise
        self._weight_version = raw_version
        return raw_version

    def _manager_call(self, method: str, *args: Any) -> Any:
        return self._get(getattr(self._rollout_manager, method).remote(*args))

    def _pause_generation(self) -> None:
        if self._generation_paused:
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
        return ray.get(value, timeout=_TRAIN_RPC_TIMEOUT_S)

    def _checkpoint_path(self, rollout_id: int) -> str:
        if self._save_hf_template is None:
            raise RuntimeError("slime args.save_hf is not set")
        return self._save_hf_template.format(rollout_id=rollout_id)

    def _marker_path(self) -> Path:
        if self._save_hf_template is None:
            raise RuntimeError("slime args.save_hf is not set")
        return marker_path(self._save_hf_template)

    def _recover_marker(self) -> dict[str, Any] | None:
        marker = read_marker(self._marker_path())
        if marker is None:
            return None
        if marker["status"] == "RUNNING":
            raise RuntimeError(f"ambiguous training job {marker['job_id']}")
        self._next_rollout_id = max(self._next_rollout_id, marker["rollout_id"] + 1)
        return marker


# A concurrent health call must remain responsive while a training RPC is
# waiting for all workers.  Keeping the implementation plain also makes the
# payload contract unit-testable without starting a Ray cluster.
# type ignore: ray's remote() overloads do not declare actor-only options
# such as max_concurrency, but the runtime accepts them for classes.
TrainBridgeActor = ray.remote(max_concurrency=64)(TrainBridgeActorImpl)  # type: ignore[call-overload]


def start_bridge(
    args,
    *,
    retention: RetentionConfig | None = None,
    loss_family: str | None = None,
    loss_family_config: object | None = None,
    actor_name: str = DEFAULT_BRIDGE_ACTOR_NAME,
    namespace: str = DEFAULT_NAMESPACE,
):
    """Boot the slime training stack and publish it as a named bridge actor.

    Run this in the slime driver process. ``args`` is slime's
    ``argparse.Namespace`` (see ``slime.utils.arguments.parse_args``);
    ``num_rollout`` is the number of externally supplied Reef training steps
    and ``save_hf`` is a checkpoint path template. The Reef service connects
    with the same ``actor_name`` and ``namespace``.
    """
    spec = resolve_loss_family(loss_family) if loss_family is not None else None
    validate_bridge_args(args, spec)
    configure_sglang_runtime(args)
    configure_megatron_runtime(args)
    configure_rollout_runtime(args)
    colocate = bool(getattr(args, "colocate", False))
    # Imported here, not at module scope: the LoRA module reaches the Megatron
    # stack, and importing the bridge actor must not drag that in (see
    # tests/reef_service/test_dependency_boundaries.py).
    from reef.train.slime_backend.reef_adapters.megatron.lora import lora_engine_slots, megatron_lora_enabled

    lora = megatron_lora_enabled(args)
    retention = retention or RetentionConfig()
    prepare_checkpoint_storage(args, retention)
    if not ray.is_initialized():
        ray.init(namespace=namespace)
    pgs = create_placement_groups(args)
    rollout_manager = create_rollout_manager(args, pgs["rollout"])
    # The sao loss family needs the critic actor group; the other loss
    # families discard it. ``args.use_critic`` comes from the explicit
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
        critic_group=critic_group,
        critic_steps_per_actor=getattr(args, "critic_steps_per_actor", None),
        critic_only_steps=getattr(args, "num_critic_only_steps", 0),
        loss_family=loss_family,
        loss_family_config=loss_family_config,
    )
