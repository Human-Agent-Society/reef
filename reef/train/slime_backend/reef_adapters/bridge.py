"""Slime training operations for Reef's backend-independent coordinator.

The adapter prepares Slime batches, executes optimizer work, persists paired
checkpoints and sends native tensors under a Reef-selected serving identity.
Reef owns scheduling, resource handoff, admission, publication and recovery.

Every job trains the loss family and learning-rate schedule its payload
names. The workers start with the recipe's startup family; a job of another
family is validated against the workers' startup arguments before it trains,
and the actor workers switch to it (and to the job's schedule) when they
receive the job's data. The schedule's progress is kept per scenario beside
the job marker, written with each checkpoint, so a restart continues it.
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
    LearningRateScheduleState,
    PreparedTrainingJob,
    PreparedTrainingStep,
    ScenarioHistoryStore,
    TrainingBackend,
    TrainingCheckpoint,
    TrainingContext,
    TrainingCoordinationConfig,
    TrainingJobResult,
    TrainingMethod,
    TrainingMetrics,
    learning_rate_metrics,
    resolve_learning_rate_schedule,
)
from reef.runtime.recovery import ScenarioHistory, history_path, marker_rollouts, read_json, write_json
from reef.runtime.scheduler import _producing_runtime_load_ids
from reef.runtime.scheduler import max_staleness as _max_staleness
from reef.train.algos.registry import loss_family_refs
from reef.train.slime_backend.algorithm import SlimeAlgorithm
from reef.train.slime_backend.data_builder import to_slime_rollout_data
from reef.train.slime_backend.distill import DistillAlgorithm
from reef.train.slime_backend.loss_families import resolve_loss_family
from reef.train.slime_backend.reef_adapters.arguments import SlimeArguments
from reef.train.slime_backend.reef_adapters.batches import (
    TRAINING_METHOD_KEY,
    TrainingBatchProcessor,
    optimizer_step_sizes,
)
from reef.train.slime_backend.reef_adapters.preflight import (
    configure_megatron_runtime,
    configure_rollout_runtime,
    prepare_checkpoint_storage,
    validate_bridge_args,
)
from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_step
from reef.train.slime_backend.reef_adapters.train_groups import SlimeTrainGroup
from reef.train.slime_backend.reef_adapters.training_job.storage import (
    LEARNING_RATE_SCHEDULES_FILENAME,
    CheckpointStorage,
    RetentionConfig,
    critic_checkpoint_due,
)
from reef.train.slime_backend.score_centering import ScoreCenteringSettings, settings_from_args

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
    ``args`` are the arguments the workers started with.
    """

    def __init__(
        self,
        actor_group,
        *,
        batch_processor: TrainingBatchProcessor,
        args: SlimeArguments,
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
        score_centering: ScoreCenteringSettings | None = None,
    ) -> None:
        self._worker_failure: ExecutorFailure | None = None
        self._score_centering = score_centering
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
        self.args = args
        self.critic_steps_per_actor = critic_steps_per_actor
        self.critic_only_steps = critic_only_steps
        # Bound algorithms and each family's worker arguments, by family name;
        # the startup family's arguments are the workers' own.
        self.loss_algorithms: dict[str, SlimeAlgorithm] = {self._algo.loss_family: self._algo}
        self.loss_family_args: dict[str, SlimeArguments] = {self._algo.loss_family: args}
        # Arguments some family changes from the startup ones: an activation sends all of them.
        self.switched_arg_names: set[str] = set()
        # The loss family and schedule state the actor workers train with now.
        self.worker_method: tuple[str, dict[str, Any] | None] = (self._algo.loss_family, None)
        self.learning_rate_schedules_path = (
            None
            if save_hf_template is None
            else Path(save_hf_template.format(rollout_id=0)).expanduser().parent / LEARNING_RATE_SCHEDULES_FILENAME
        )
        self.learning_rate_schedules = read_learning_rate_schedules(
            self.learning_rate_schedules_path, next_rollout_id=start_rollout_id
        )
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
        algorithm = self.loss_algorithm(payload["loss"])
        method = TrainingMethod.from_dict(payload["method"])
        schedule = resolve_learning_rate_schedule(self.learning_rate_schedule(scenario), method.learning_rate_schedule)
        if self._score_centering is not None:
            # Torch, like the tensorization that follows; loaded only when the term is on.
            from reef.train.slime_backend.score_centering.heads import attach_sampler_heads

            attach_sampler_heads(rollout_data, payload, self._score_centering)
        rollout_versions = rollout_data.get("producing_runtime_load_ids")
        if (
            max_staleness > 0
            and rollout_versions is not None
            and list(rollout_versions) != list(_producing_runtime_load_ids(payload))
        ):
            raise ValueError("loss-family row producing versions do not match the shared training payload")
        algorithm.validate_payload(rollout_data)
        optimizer_steps = 0
        if schedule is not None:
            if self.args.decoupled_lr is not None:
                raise RuntimeError(
                    "a recipe-selected learning-rate schedule cannot drive --decoupled-lr parameter groups"
                )
            optimizer_steps = len(optimizer_step_sizes(rollout_data, self.args.global_batch_size))
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
            algorithm_metrics = algorithm.prepare_rollout(rollout_data)
            worker_method = (algorithm.loss_family, None if schedule is None else schedule.to_dict())
            if worker_method != self.worker_method:
                rollout_data[TRAINING_METHOD_KEY] = self.worker_activation(algorithm.loss_family, schedule)
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
                algorithm=algorithm,
                algorithm_metrics=algorithm_metrics,
                worker_method=worker_method,
                learning_rate_schedule=schedule,
                optimizer_steps=optimizer_steps,
            )

    def train_job(self, job: _SlimePreparedTrainingJob) -> TrainingMetrics:
        """Run one job's optimizer steps and collect its metrics."""
        checkpoint = job.checkpoint
        if checkpoint.scenario is not None:
            self._group.activate_scenario(checkpoint.scenario)
        training = job.algorithm.train(
            checkpoint.rollout_id,
            job.packed,
            actor_group=self._group,
            critic_group=self._critic_group,
            resolve=self._get,
        )
        durable_metrics = {
            **training.durable_metrics,
            **job.algorithm.rollout_metrics(job.rollout_data, self.context.runtime_load_id),
        }
        train_metrics = next(
            (dict(result) for result in training.worker_results if isinstance(result, Mapping) and result),
            {},
        )
        train_metrics.update(self._get(self._group.async_pop_rank0_metrics()))
        train_metrics.update(job.algorithm_metrics)
        if training.actor_trained:
            # The actor fetched the job's data, and with it any activation.
            self.worker_method = job.worker_method
            schedule = job.learning_rate_schedule
            if schedule is not None:
                job.learning_rate_schedule_after = schedule.advanced(job.optimizer_steps)
                train_metrics.update(
                    learning_rate_metrics(
                        schedule.learning_rates(job.optimizer_steps), job.learning_rate_schedule_after
                    )
                )
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
        if job.learning_rate_schedule_after is not None:
            self.record_learning_rate_schedule(checkpoint.scenario, rollout_id, job.learning_rate_schedule_after)

    def loss_algorithm(self, loss_family: str) -> SlimeAlgorithm:
        """The bound algorithm of a job's loss family; a new family is validated against the workers first.

        A family other than the startup one runs with its default driver
        options and the workers' startup arguments (see
        ``loss_family_job_args``). The first use registers its wire keys with
        the batch processor; a family that declares one of them with another
        dtype is refused.
        """
        if not self._algo.loss_family:
            # A bridge started without a family trains whatever it is given.
            return self._algo
        spec = resolve_loss_family(loss_family)
        algorithm = self.loss_algorithms.get(spec.loss_family)
        if algorithm is not None:
            return algorithm
        if isinstance(spec, DistillAlgorithm) and any(
            isinstance(bound, DistillAlgorithm) for bound in self.loss_algorithms.values()
        ):
            raise RuntimeError(
                f"loss family {spec.loss_family!r} is a second distillation family in this run; "
                "a worker keeps one teacher, built from the first distillation family's settings"
            )
        # Imported here: the argument module reaches the Megatron LoRA stack,
        # which importing the bridge must not load (test_dependency_boundaries).
        from reef.train.slime_backend.reef_adapters.slime_arguments import loss_family_job_args

        job_args = loss_family_job_args(self.args, spec)
        dtypes = dict(self.args.reef_rollout_tensor_dtypes or {})
        for key, dtype in job_args.reef_rollout_tensor_dtypes.items():
            if dtypes.setdefault(key, dtype) != dtype:
                raise RuntimeError(
                    f"loss family {spec.loss_family!r} declares rollout key {key!r} as {dtype}, "
                    f"another family of this run as {dtypes[key]}"
                )
        startup = vars(self.args)
        self.switched_arg_names.update(
            name for name, value in vars(job_args).items() if name not in startup or startup[name] != value
        )
        # The batch processor partitions and tensorizes the keys a job's data carries.
        self.args.custom_rollout_data_keys = tuple(
            dict.fromkeys((*(self.args.custom_rollout_data_keys or ()), *(job_args.custom_rollout_data_keys or ())))
        )
        self.args.reef_rollout_tensor_dtypes = dtypes
        algorithm = spec.bind(
            None, critic_steps_per_actor=self.critic_steps_per_actor, critic_only_steps=self.critic_only_steps
        )
        self.loss_family_args[spec.loss_family] = job_args
        self.loss_algorithms[spec.loss_family] = algorithm
        return algorithm

    def worker_activation(self, loss_family: str, schedule: LearningRateScheduleState | None) -> dict[str, Any]:
        """What the actor workers set before a job: the family's switched arguments and the schedule state."""
        family_args = {}
        if self.switched_arg_names:
            values = vars(self.loss_family_args[loss_family])
            family_args = {name: values.get(name) for name in sorted(self.switched_arg_names)}
        return {
            "loss_family_args": family_args,
            "learning_rate_schedule": None if schedule is None else schedule.to_dict(),
        }

    def learning_rate_schedule(self, scenario: str | None) -> LearningRateScheduleState | None:
        """The active schedule of ``scenario``'s weights (``None``: the shared model), as its last checkpoint left it."""
        record = self.learning_rate_schedules.get(scenario or "")
        return None if record is None else LearningRateScheduleState.from_dict(record)

    def record_learning_rate_schedule(
        self, scenario: str | None, rollout_id: int, schedule: LearningRateScheduleState
    ) -> None:
        """Keep the schedule state a saved checkpoint trained to, beside the job marker."""
        if self.learning_rate_schedules_path is None:
            raise RuntimeError("the Slime bridge keeps learning-rate schedule progress beside --save-hf")
        self.learning_rate_schedules[scenario or ""] = {"rollout_id": rollout_id, **schedule.to_dict()}
        write_json(self.learning_rate_schedules_path, {"schedules": self.learning_rate_schedules})

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
        method: TrainingMethod,
        algorithm_state: Mapping[str, Any],
        scheduling: StepScheduling,
    ) -> PreparedTrainingStep:
        """Prepare a framework-neutral Reef batch with Slime-owned logic; refuse a method the workers cannot train."""
        prepared = prepare_slime_step(
            batch, method, algorithm_state, scheduling, sampler_topk=self._score_centering is not None
        )
        if prepared.payload is not None:
            self.loss_algorithm(prepared.payload["loss"]).validate_payload(prepared.payload)
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
        algorithm: SlimeAlgorithm,
        algorithm_metrics: Mapping[str, Any],
        worker_method: tuple[str, dict[str, Any] | None],
        learning_rate_schedule: LearningRateScheduleState | None,
        optimizer_steps: int,
    ) -> None:
        self._backend = backend
        self._checkpoint = checkpoint
        self.job_id = job_id
        self.rollout_data = rollout_data
        self.packed = packed
        self.algorithm = algorithm
        self.algorithm_metrics = algorithm_metrics
        self.worker_method = worker_method
        self.learning_rate_schedule = learning_rate_schedule
        self.optimizer_steps = optimizer_steps
        # The schedule state after training, once the actor stepped; recorded with the checkpoint.
        self.learning_rate_schedule_after: LearningRateScheduleState | None = None

    @property
    def checkpoint(self) -> TrainingCheckpoint:
        return self._checkpoint

    def train(self) -> TrainingMetrics:
        return self._backend.train_job(self)

    def save_checkpoint(self) -> None:
        self._backend.save_job_checkpoint(self)


def read_learning_rate_schedules(path: Path | None, *, next_rollout_id: int) -> dict[str, dict[str, Any]]:
    """Every scenario's schedule progress, refusing progress newer than the checkpoints the workers loaded."""
    value = None if path is None else read_json(path)
    if value is None:
        return {}
    records: dict[str, dict[str, Any]] = {}
    for scenario, record in value["schedules"].items():
        LearningRateScheduleState.from_dict(record)
        if record["rollout_id"] >= next_rollout_id:
            raise RuntimeError(
                f"learning-rate schedule progress in {path} was recorded at rollout {record['rollout_id']}, "
                f"after the checkpoint the workers loaded (next rollout {next_rollout_id}); restore the file "
                "that belongs to that checkpoint"
            )
        records[scenario] = dict(record)
    return records


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
        args=args,
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
        score_centering=settings_from_args(args),
    )
