"""Slime training operations for Reef's backend-independent coordinator.

The adapter prepares Slime batches, executes optimizer work, persists paired
checkpoints and sends native tensors under a Reef-selected serving identity.
Reef owns scheduling, resource handoff, admission, publication and recovery.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reef.core.batches import StepScheduling, TrainingBatch
from reef.runtime.executor import resolve
from reef.runtime.executor.failure import ExecutorFailedError, ExecutorFailure, ExecutorFailureListener
from reef.runtime.interfaces import (
    PreparedTrainingJob,
    PreparedTrainingStep,
    ScenarioHistoryStore,
    TrainingBackend,
    TrainingCheckpoint,
    TrainingContext,
    TrainingCoordinationConfig,
    TrainingJobResult,
    TrainingMetrics,
)
from reef.runtime.recovery import ScenarioHistory, history_path, marker_rollouts
from reef.runtime.scheduler import _producing_runtime_load_ids
from reef.runtime.scheduler import max_staleness as _max_staleness
from reef.train.algos.registry import loss_family_refs
from reef.train.slime_backend.algorithm import SlimeAlgorithm
from reef.train.slime_backend.data_builder import to_slime_rollout_data
from reef.train.slime_backend.loss_families import resolve_loss_family
from reef.train.slime_backend.reef_adapters.arguments import SlimeArguments
from reef.train.slime_backend.reef_adapters.batches import TrainingBatchProcessor
from reef.train.slime_backend.reef_adapters.preflight import (
    configure_megatron_runtime,
    configure_rollout_runtime,
    prepare_checkpoint_storage,
    validate_bridge_args,
)
from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_step
from reef.train.slime_backend.reef_adapters.train_groups import SlimeTrainGroup
from reef.train.slime_backend.reef_adapters.training_job.storage import (
    CheckpointStorage,
    RetentionConfig,
    critic_checkpoint_due,
)

# One training step (train + checkpoint + publish) legitimately takes hours;
# this bounds a single Ray RPC from the bridge to its workers.
_TRAIN_RPC_TIMEOUT_S = 14_400


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


class SlimeTrainingBackend(TrainingBackend, ExecutorFailureListener):
    """Slime training, checkpointing and native sender operations.

    This adapter has no inference control object and never reads or advances
    Reef's publication marker. Its context contains only scheduling values.
    """

    def __init__(
        self,
        actor_group,
        *,
        batch_processor: TrainingBatchProcessor,
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
        critic_save_interval: int = 1,
        loss_family: str | None = None,
        loss_family_config: object | None = None,
        loss_runtime: SlimeAlgorithm | None = None,
    ) -> None:
        self._worker_failure: ExecutorFailure | None = None
        self._group = actor_group
        self._critic_group = critic_group
        self._critic_save_root = critic_save_root if critic_group is not None else None
        if (
            not isinstance(critic_save_interval, int)
            or isinstance(critic_save_interval, bool)
            or critic_save_interval < 1
        ):
            raise ValueError("critic_save_interval must be a positive integer")
        self.critic_save_interval = critic_save_interval
        self._batch_processor = batch_processor
        self._save_hf_template = save_hf_template
        self._config = TrainingCoordinationConfig(
            save_hf_template, colocate, lora, adapter_capacity, keep_lora_base_resident
        )
        self._context = TrainingContext(
            next_rollout_id=start_rollout_id,
            history=ScenarioHistory(history_path(save_hf_template)) if lora and save_hf_template is not None else None,
        )
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
        self._storage = (
            CheckpointStorage(
                storage_config,
                hf_template=save_hf_template,
                megatron_root=megatron_save_root,
                critic_root=self._critic_save_root,
                critic_save_interval=critic_save_interval,
                source_hf=source_hf,
                source_megatron=source_megatron,
                lora=lora,
            )
            if storage_config is not None and save_hf_template is not None and megatron_save_root is not None
            else None
        )

    @property
    def config(self) -> TrainingCoordinationConfig:
        return self._config

    @property
    def context(self) -> TrainingContext:
        return self._context

    def start(self) -> None:
        for group in (self._group, self._critic_group):
            if group is not None:
                group.register_failure_listener(self)

    def check_health(self) -> None:
        if self._worker_failure is not None:
            raise ExecutorFailedError(self._worker_failure)

    def on_executor_failure(self, failure: ExecutorFailure) -> None:
        self._worker_failure = failure

    @property
    def _runtime_load_id(self) -> str:
        return self.context.runtime_load_id

    @property
    def _next_rollout_id(self) -> int:
        return self.context.next_rollout_id

    @property
    def _history(self) -> ScenarioHistoryStore | None:
        return self.context.history

    @contextmanager
    def prepare(
        self,
        payload: Mapping[str, Any],
        *,
        job_id: str,
        scenario_step: int,
        prior_marker: Mapping[str, Any] | None,
    ) -> Iterator[PreparedTrainingJob | TrainingJobResult]:
        scenario = self._job_scenario(payload)
        # The checkpoint index is the bridge's own sequence, not the scenario step.
        rollout_id = self._next_rollout_id
        max_staleness = _max_staleness(payload)
        checkpoint = Path(self._checkpoint_path(rollout_id))
        if self._storage is None and (checkpoint.exists() or checkpoint.is_symlink()):
            raise RuntimeError(f"checkpoint target already exists: {checkpoint}")
        rollout_data = to_slime_rollout_data(dict(payload))
        rollout_versions = rollout_data.get("producing_runtime_load_ids")
        if (
            max_staleness > 0
            and rollout_versions is not None
            and list(rollout_versions) != list(_producing_runtime_load_ids(payload))
        ):
            raise ValueError("loss-family row producing versions do not match the shared training payload")
        self._algo.validate_payload(rollout_data)
        context: Any = nullcontext(None)
        if self._storage is not None:
            protected = marker_rollouts(prior_marker)
            if self._history is not None:
                # Every scenario's latest checkpoint is its restart source.
                protected |= self._history.protected_rollouts()
            context = self._storage.admit(
                rollout_id=rollout_id,
                active_rollouts=protected,
            )
        with context as storage_plan:
            if storage_plan is not None and storage_plan["blocked"]:
                yield TrainingJobResult(
                    outcome="storage_blocked",
                    storage=storage_plan,
                    runtime_load_id=self._runtime_load_id,
                )
                return
            # The teacher is scored before the RUNNING marker so a
            # scoring failure leaves no partial state: the job
            # stays retryable under the same identity.
            algorithm_metrics = self._algo.prepare_rollout(rollout_data)
            # Local batch processing preserves Slime's DP schedule and
            # object-store transport: one Box per training DP rank.
            packed = self._batch_processor.prepare_external_train_data(rollout_data)
            yield _SlimePreparedTrainingJob(
                self,
                checkpoint=TrainingCheckpoint(
                    rollout_id=rollout_id, path=checkpoint, scenario_step=scenario_step, scenario=scenario
                ),
                job_id=job_id,
                rollout_data=rollout_data,
                packed=packed,
                algorithm_metrics=algorithm_metrics,
            )

    def train_job(self, job: _SlimePreparedTrainingJob) -> TrainingMetrics:
        """Run one optimizer step for an admitted job and collect its metrics."""
        checkpoint = job.checkpoint
        if checkpoint.scenario is not None:
            self._group.activate_scenario(checkpoint.scenario)
        training = self._algo.train(
            checkpoint.rollout_id,
            job.packed,
            actor_group=self._group,
            critic_group=self._critic_group,
            resolve=self._get,
        )
        durable_metrics = {
            **training.durable_metrics,
            **self._algo.rollout_metrics(job.rollout_data, self.context.runtime_load_id),
        }
        train_metrics = next(
            (dict(result) for result in training.worker_results if isinstance(result, Mapping) and result),
            {},
        )
        train_metrics.update(self._get(self._group.async_pop_rank0_metrics()))
        train_metrics.update(job.algorithm_metrics)
        return TrainingMetrics(training=train_metrics, durable=durable_metrics)

    def save_job_checkpoint(self, job: _SlimePreparedTrainingJob) -> None:
        """Persist the paired model/optimizer checkpoints and record the step."""
        checkpoint = job.checkpoint
        rollout_id = checkpoint.rollout_id
        self._group.save_model(rollout_id, force_sync=True, scenario_step=checkpoint.scenario_step)
        if self._critic_save_root is not None and critic_checkpoint_due(rollout_id, self.critic_save_interval):
            # Persist the critic's weights and optimizer alongside the actor
            # pair: every commit by default, critic-only warmup included,
            # otherwise the value head cold-starts on every reboot (SAO's
            # stated cold-start concern). A larger interval skips the full
            # critic save on the commits in between. No HF export: the
            # critic never serves.
            self._critic_group.save_model(rollout_id, force_sync=True, scenario_step=checkpoint.scenario_step)
        if checkpoint.path.is_symlink() or not checkpoint.path.is_dir():
            raise RuntimeError(f"checkpoint is missing or unsafe: {checkpoint.path}")
        if self._storage is not None:
            rewards = job.rollout_data["rewards"]
            self._storage.complete(job.job_id, rollout_id, reward=math.fsum(rewards) / len(rewards))
        if checkpoint.scenario is not None:
            self._require_history().record_checkpoint(checkpoint.scenario, rollout_id)

    def prepare_weights(self, runtime_load_id: str, *, force_full: bool) -> None:
        self._group.prepare_weight_update(runtime_load_id, force_full=force_full)

    def send_weights(self, runtime_load_id: str, *, force_full: bool) -> str:
        self._group.send_prepared_weights(runtime_load_id, force_full=force_full)
        return self.current_runtime_load_id()

    def current_runtime_load_id(self) -> str:
        return str(self._get(self._group.async_get_rank0_runtime_load_id()))

    def initialize_version(self, runtime_load_id: str) -> None:
        self._group.initialize_runtime_load_id(runtime_load_id)

    def activate_scenario(self, scenario: str) -> None:
        self._group.activate_scenario(scenario)

    def send_adapter(self, scenario: str, name: str) -> None:
        self._group.publish_adapter(scenario, name)

    def close(self) -> None:
        errors = []
        for group in (self._critic_group, self._group):
            if group is not None:
                try:
                    group.release()
                except Exception as exc:
                    errors.append(exc)
        if errors:
            raise errors[0]

    def _require_history(self) -> ScenarioHistoryStore:
        """The per-scenario history; only LoRA runs with a checkpoint save path keep one."""
        if self._history is None:
            raise RuntimeError("scenario bookkeeping requires LoRA training with a checkpoint save path")
        return self._history

    def _job_scenario(self, payload: Mapping[str, Any]) -> str | None:
        scenario = payload.get("scenario")
        if self._history is None:
            return None
        if not isinstance(scenario, str) or not scenario:
            raise ValueError("per-scenario LoRA training jobs must name their scenario")
        return scenario

    def prepare_training_step(
        self,
        batch: TrainingBatch,
        objective: str,
        algorithm_state: Mapping[str, Any],
        scheduling: StepScheduling,
    ) -> PreparedTrainingStep:
        """Prepare a framework-neutral Reef batch with Slime-owned logic."""
        prepared = prepare_slime_step(batch, objective, algorithm_state, scheduling)
        if prepared.payload is not None:
            self._algo.validate_payload(prepared.payload)
        return prepared

    def _checkpoint_path(self, rollout_id: int) -> str:
        if self._save_hf_template is None:
            raise RuntimeError("slime args.save_hf is not set")
        return self._save_hf_template.format(rollout_id=rollout_id)

    @staticmethod
    def _get(value: Any) -> Any:
        return resolve(value, timeout=_TRAIN_RPC_TIMEOUT_S)


class _SlimePreparedTrainingJob(PreparedTrainingJob):
    """One admitted Slime step; the backend runs it when Reef says so."""

    def __init__(
        self,
        backend: SlimeTrainingBackend,
        *,
        checkpoint: TrainingCheckpoint,
        job_id: str,
        rollout_data: dict[str, Any],
        packed: Any,
        algorithm_metrics: Mapping[str, Any],
    ) -> None:
        self._backend = backend
        self._checkpoint = checkpoint
        self.job_id = job_id
        self.rollout_data = rollout_data
        self.packed = packed
        self.algorithm_metrics = algorithm_metrics

    @property
    def checkpoint(self) -> TrainingCheckpoint:
        return self._checkpoint

    def train(self) -> TrainingMetrics:
        return self._backend.train_job(self)

    def save_checkpoint(self) -> None:
        self._backend.save_job_checkpoint(self)


@dataclass(frozen=True)
class BridgePreparation:
    """Validated bridge choices, resolved before model resources are allocated."""

    retention: RetentionConfig
    loss_family: str | None
    lora: bool


def prepare_bridge(
    args: Any, *, retention: RetentionConfig | None = None, loss_family: str | None = None
) -> BridgePreparation:
    """Validate training configuration and checkpoint storage before allocation."""
    retention = retention or RetentionConfig()
    spec = resolve_loss_family(loss_family) if loss_family is not None else None
    validate_bridge_args(args, spec)
    if loss_family is not None and ":" not in loss_family:
        loss_family = loss_family_refs().get(loss_family) or loss_family
    configure_megatron_runtime(args)
    configure_rollout_runtime(args)
    from reef.train.slime_backend.reef_adapters.executors.config import slime_executor_class

    slime_executor_class(getattr(args, "reef_executor_backend", "auto"))
    # Imported here, not at module scope: the LoRA module reaches the Megatron
    # stack, and importing the bridge actor must not drag that in (see
    # tests/reef_service/test_dependency_boundaries.py).
    from reef.train.slime_backend.reef_adapters.megatron.lora import megatron_lora_enabled

    lora = megatron_lora_enabled(args)
    prepare_checkpoint_storage(args, retention)
    return BridgePreparation(retention=retention, loss_family=loss_family, lora=lora)


def create_training_backend(
    args: SlimeArguments,
    actor_group: SlimeTrainGroup,
    critic_group: SlimeTrainGroup | None,
    *,
    preparation: BridgePreparation,
    loss_family_config: object | None = None,
) -> SlimeTrainingBackend:
    """Build a training-only adapter around already-started Slime workers."""
    from reef.train.slime_backend.reef_adapters.megatron.lora import lora_engine_slots

    return SlimeTrainingBackend(
        actor_group,
        batch_processor=TrainingBatchProcessor(args, actor_group.train_parallel_config),
        save_hf_template=args.save_hf,
        start_rollout_id=args.start_rollout_id or 0,
        storage_config=preparation.retention,
        megatron_save_root=args.save,
        critic_save_root=args.critic_save,
        source_hf=args.hf_checkpoint,
        source_megatron=args.load,
        colocate=bool(args.colocate),
        lora=preparation.lora,
        adapter_capacity=lora_engine_slots(args) if preparation.lora else None,
        keep_lora_base_resident=bool(args.keep_lora_base_resident),
        critic_group=critic_group,
        critic_steps_per_actor=args.critic_steps_per_actor,
        critic_only_steps=args.num_critic_only_steps,
        critic_save_interval=args.critic_save_interval,
        loss_family=preparation.loss_family,
        loss_family_config=loss_family_config,
    )
