"""Slime as a training component in the HTTP service's process.

``SlimeTrainingComponent`` owns one Ray session, the training GPUs it reserves
there, and Slime's actor (and critic) workers. A job trains as Slime
rollouts, one by default or one per ``save_interval`` optimizer steps: Reef's
job data becomes Slime's rollout data, the workers train its optimizer steps,
and the component saves the paired Hugging Face and Megatron checkpoints of
each rollout. No serving engine takes part; Reef serves the Hugging Face export
(or, with LoRA, the PEFT adapter) of a checkpoint that passes evaluation.

Training continues from the workers' own weights. A job therefore starts from
the component's latest checkpoint; after a restart, the workers load that
checkpoint again. A rollback to another release is refused, as Slime cannot
load a published release back into its workers.

Every job trains the loss family and learning-rate schedule its parameters
name. The workers start with the recipe's startup family; a job of another
family is validated against the workers' startup arguments before it trains,
and the actor workers switch to it (and to the job's schedule) when they
receive the job's data.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reef.runtime.executor import resolve
from reef.runtime.interfaces import (
    LearningRateScheduleState,
    learning_rate_metrics,
    resolve_learning_rate_schedule_state,
)
from reef.runtime.jobs import ReleaseFormat, SavedCheckpoint, TrainingJob
from reef.runtime.recovery import read_json, write_json
from reef.runtime.serving import ADAPTER_DIR
from reef.train.algos.registry import job_training, loss_family_refs
from reef.train.algos.signals import PreparedJob
from reef.train.local_jobs import LocalJobComponent
from reef.train.slime_backend.algorithm import SlimeAlgorithm
from reef.train.slime_backend.data_builder import to_slime_rollout_data
from reef.train.slime_backend.distill import DistillAlgorithm
from reef.train.slime_backend.loss_families import UnknownLossFamilyError, resolve_loss_family
from reef.train.slime_backend.reef_adapters.arguments import SlimeArguments
from reef.train.slime_backend.reef_adapters.batches import WORKER_SWITCH_KEY, TrainingBatchProcessor
from reef.train.slime_backend.reef_adapters.preflight import (
    configure_megatron_runtime,
    configure_rollout_runtime,
    prepare_checkpoint_storage,
    validate_slime_args,
)
from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_job, slime_payload_steps
from reef.train.slime_backend.reef_adapters.train_groups import (
    DEFAULT_TRAIN_RPC_TIMEOUT_S,
    SlimeTrainGroup,
    merge_rank_metrics,
)
from reef.train.slime_backend.reef_adapters.training_job.storage import (
    CheckpointStorage,
    RetentionConfig,
    critic_checkpoint_due,
)
from reef.train.slime_backend.resources import SlimeDeploymentResources
from reef.train.slime_backend.score_centering import ScoreCenteringSettings, settings_from_args

logger = logging.getLogger(__name__)

#: A start directory's description of the weights a later job continues from.
START_STATE = "start.json"
#: The rollout a job trains and the job's optimizer steps through it, written before the workers train it.
ROLLOUT_FILE = "rollout.json"


@dataclass(frozen=True)
class WorkerTrainingSettings:
    """The loss family and learning-rate schedule state that the actor workers train with.

    ``loss_family`` is the name of a bound loss family, non-empty.
    ``schedule_state`` is the job's schedule progress, or ``None`` for Slime's
    configured scheduler. ``SlimeTrainingComponent.train_and_save`` makes the
    value for each job; a value unequal to the workers' current one sends a
    worker switch with the job's data.
    """

    loss_family: str
    schedule_state: LearningRateScheduleState | None

    def __post_init__(self) -> None:
        if not isinstance(self.loss_family, str) or not self.loss_family:
            raise ValueError("WorkerTrainingSettings.loss_family must be a non-empty string")


@dataclass(frozen=True)
class SlimeStart:
    """What a job continues from: the weights' rollout, the objective's state, the schedule, and whether it is fresh.

    ``rollout_id`` is the non-negative rollout whose checkpoint holds the
    weights, or ``None`` for the weights the workers loaded at startup.
    ``objective`` names the objective whose state ``algorithm_state`` (a JSON
    object) is; it is ``None`` exactly when the state is empty. A job continues
    that state only when it trains the same objective; another objective
    starts with an empty state. ``schedule_state`` is the learning-rate
    schedule's progress, or ``None`` for Slime's configured scheduler.
    ``fresh`` is True for a start that ``fresh_start`` made: a job from it
    first clears the workers' optimizer state and restarts their scheduler.
    A fresh start has an empty state and no schedule progress. A start
    directory keeps the value in ``start.json``; ``record_rollout`` and
    ``fresh_start`` write it, and ``check_job`` and ``run`` read it.
    """

    rollout_id: int | None
    objective: str | None
    algorithm_state: Mapping[str, Any]
    schedule_state: LearningRateScheduleState | None
    fresh: bool = False

    def __post_init__(self) -> None:
        if self.rollout_id is not None and (
            isinstance(self.rollout_id, bool) or not isinstance(self.rollout_id, int) or self.rollout_id < 0
        ):
            raise ValueError("SlimeStart.rollout_id must be a non-negative int or None")
        if not isinstance(self.algorithm_state, Mapping):
            raise TypeError("SlimeStart.algorithm_state must be a mapping")
        if self.objective is not None and (not isinstance(self.objective, str) or not self.objective):
            raise ValueError("SlimeStart.objective must be a non-empty string or None")
        if (self.objective is None) != (not self.algorithm_state):
            raise ValueError("SlimeStart.objective names the objective of a non-empty state, and only of one")
        if self.fresh and (self.algorithm_state or self.schedule_state is not None):
            raise ValueError("a fresh SlimeStart has no objective state and no schedule progress")

    def write(self, directory: Path) -> None:
        write_json(
            directory / START_STATE,
            {
                "rollout_id": self.rollout_id,
                "objective": self.objective,
                "algorithm_state": dict(self.algorithm_state),
                "schedule_state": None if self.schedule_state is None else self.schedule_state.to_dict(),
                "fresh": self.fresh,
            },
        )

    @classmethod
    def read(cls, directory: Path) -> SlimeStart:
        """Read the start in ``directory``; raise ``ValueError`` for a directory that holds none."""
        value = read_json(directory / START_STATE)
        if value is None:
            raise ValueError(f"{directory} is not a Slime start: it has no {START_STATE}")
        schedule = value["schedule_state"]
        return cls(
            value["rollout_id"],
            value["objective"],
            value["algorithm_state"],
            None if schedule is None else LearningRateScheduleState.from_dict(schedule),
            value["fresh"],
        )

    def state_for(self, objective: str) -> Mapping[str, Any]:
        """The objective state a job of ``objective`` continues: this start's state for the same objective, else empty."""
        return self.algorithm_state if self.objective == objective else {}


class SlimeTrainingComponent(LocalJobComponent):
    """Train each job as Slime rollouts on the component's own workers, from its latest checkpoint.

    A job's parameters are ``method`` (a ``TrainingMethod`` as a dict),
    optionally ``scheduling`` (a ``StepScheduling`` as a dict, see
    ``job_training``), and optionally ``save_interval``, a positive int: the
    optimizer steps between two saved checkpoints. ``submit`` refuses any
    other parameter. Each run of ``save_interval`` steps trains as one Slime
    rollout and saves one checkpoint; without it, the job is one rollout with
    one checkpoint. A checkpoint's step is the job's optimizer steps up to it.

    The workers hold one set of weights, so the component accepts only a job
    that continues from them: a job whose start names the rollout the workers
    hold (the latest saved rollout's start, or a fresh start of it), or a job
    whose start is ``None``, which continues from the weights the workers
    hold. It refuses every release as a start, and it trains the jobs of one
    scenario. ``fresh_start`` keeps the weights and drops the training state:
    the optimizer state, the scheduler's progress and the objective's state.

    ``args`` are the arguments the workers started with; ``storage`` bounds
    the paired checkpoints on disk. Beside the checkpoint roots,
    ``reef-jobs/`` holds the jobs (see ``LocalJobComponent``) and
    ``reef-starts/rollout_<id>/`` the start of each saved rollout: its
    ``start.json`` and, with LoRA, the published release in
    ``release/adapter/``. Without LoRA, the release is the rollout's Hugging
    Face export.
    """

    def __init__(
        self,
        args: SlimeArguments,
        actor_group: SlimeTrainGroup,
        critic_group: SlimeTrainGroup | None,
        *,
        batch_processor: TrainingBatchProcessor,
        storage: CheckpointStorage | None,
        lora: bool,
        loss_family: str,
        loss_family_config: object | None = None,
        score_centering: ScoreCenteringSettings | None = None,
        train_rpc_timeout_s: float = DEFAULT_TRAIN_RPC_TIMEOUT_S,
        resources: SlimeDeploymentResources | None = None,
    ) -> None:
        self.args = args
        self.train_rpc_timeout_s = train_rpc_timeout_s
        self._group = actor_group
        self._critic_group = critic_group
        self._critic_save_root = args.critic_save if critic_group is not None else None
        if (
            not isinstance(args.critic_save_interval, int)
            or isinstance(args.critic_save_interval, bool)
            or args.critic_save_interval < 1
        ):
            raise ValueError("critic_save_interval must be a positive integer")
        self._batch_processor = batch_processor
        self._storage = storage
        self._lora = lora
        self._score_centering = score_centering
        self._resources = resources
        self.critic_steps_per_actor = args.critic_steps_per_actor
        self.critic_only_steps = args.num_critic_only_steps
        self._algo = resolve_loss_family(loss_family).bind(
            loss_family_config,
            critic_steps_per_actor=self.critic_steps_per_actor,
            critic_only_steps=self.critic_only_steps,
        )
        # Bound algorithms and each family's worker arguments, by family name;
        # the startup family's arguments are the workers' own.
        self.algorithms_by_loss_family: dict[str, SlimeAlgorithm] = {self._algo.loss_family: self._algo}
        self.args_by_loss_family: dict[str, SlimeArguments] = {self._algo.loss_family: args}
        # Arguments some family changes from the startup ones: a worker switch sends all of them.
        self.switched_arg_names: set[str] = set()
        self.worker_settings = WorkerTrainingSettings(self._algo.loss_family, None)
        root = Path(args.save_hf.format(rollout_id=0)).expanduser().resolve().parent.parent
        self.starts_root = root / "reef-starts"
        self.next_rollout_id = int(args.start_rollout_id or 0)
        # The workers loaded the checkpoint of the rollout before the next one, or the startup weights. A job
        # that a restart stopped after Megatron saved that rollout writes that rollout's start when it resumes.
        self.loaded_rollout_id: int | None = self.next_rollout_id - 1 if self.next_rollout_id > 0 else None
        super().__init__(root / "reef-jobs")
        self.start_jobs()

    @property
    def name(self) -> str:
        return "slime"

    @property
    def release_format(self) -> ReleaseFormat:
        """Full weights (the rollout's Hugging Face export), or with LoRA the PEFT adapter of the workers' rank."""
        if not self._lora:
            return ReleaseFormat("full")
        from reef.train.slime_backend.reef_adapters.megatron.lora import sglang_lora_target_modules

        return ReleaseFormat(
            "adapter", int(self.args.megatron_lora_rank), tuple(sglang_lora_target_modules(self.args))
        )

    def import_start(self, release: Path) -> str | None:
        raise ValueError(
            "Slime continues training from its own latest checkpoint; it cannot load a published release "
            "into its workers"
        )

    def fresh_start(self, start: str | None) -> str | None:
        """A start of the weights ``start`` names, with no training state; ``check_job`` checks the workers hold them."""
        rollout_id = None if start is None else SlimeStart.read(Path(start)).rollout_id
        directory = self.starts_root / ("fresh_startup" if rollout_id is None else f"fresh_rollout_{rollout_id}")
        SlimeStart(rollout_id, None, {}, None, fresh=True).write(directory)
        return str(directory)

    def check_job(self, job: TrainingJob) -> None:
        unknown = set(job.parameters) - {"method", "scheduling", "save_interval"}
        if unknown:
            raise ValueError(f"Slime jobs accept no parameters {sorted(unknown)}")
        interval = job.parameters.get("save_interval")
        if interval is not None and (isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0):
            raise ValueError("save_interval must be a positive int")
        job_training(job.parameters)
        # A start without a rollout names the weights the workers loaded at startup, whichever they are.
        rollout_id = None if job.start is None else SlimeStart.read(Path(job.start)).rollout_id
        if rollout_id is not None and rollout_id != self.loaded_rollout_id:
            raise ValueError(
                f"Slime's workers hold the weights of rollout {self.loaded_rollout_id!r}; "
                f"a job cannot start from rollout {rollout_id!r} ({job.start})"
            )

    def close_resources(self) -> None:
        errors = []
        for group in (self._critic_group, self._group):
            if group is not None:
                try:
                    group.release()
                except Exception as exc:
                    errors.append(exc)
        if self._resources is not None:
            try:
                self._resources.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]

    # -- One job

    def run(self, job: TrainingJob, directory: Path) -> Mapping[str, Any]:
        start = None if job.start is None else SlimeStart.read(Path(job.start))
        method, scheduling = job_training(job.parameters)
        prepared = prepare_slime_job(
            job.data,
            method,
            {} if start is None else start.state_for(method.objective),
            scheduling,
            step_size=self.args.global_batch_size,
            sampler_topk=self._score_centering is not None,
        )
        if prepared.payload is None:
            # The objective skipped the job's data: the job trains nothing and saves no checkpoint.
            return prepared.metrics
        payload = prepared.payload
        algorithm = self.algorithm_for_loss_family(payload["loss"])
        algorithm.validate_payload(payload)
        step_count = len(payload["external_step_sizes"])
        interval = job.parameters.get("save_interval") or step_count
        saved = self.saved(directory)
        # A resumed job continues after its last saved checkpoint, with that checkpoint's schedule progress.
        done = saved[-1].step if saved else 0
        if saved:
            schedule_state = SlimeStart.read(Path(saved[-1].path)).schedule_state
        else:
            schedule_state = resolve_learning_rate_schedule_state(
                None if start is None else start.schedule_state, method.learning_rate_schedule
            )
        if schedule_state is not None and self.args.decoupled_lr is not None:
            raise RuntimeError("a recipe-selected learning-rate schedule cannot drive --decoupled-lr parameter groups")
        # Until the job saves its first rollout, the workers hold the start's training state: a fresh job resets it.
        fresh = start is not None and start.fresh and not saved
        metrics: dict[str, Any] = dict(prepared.metrics)
        for first in range(done, step_count, interval):
            if self.stopping(directory):
                break
            last = min(first + interval, step_count)
            rollout_payload = slime_payload_steps(payload, first, last)
            # Optimizer steps advance only a recipe-selected schedule.
            next_schedule = None if schedule_state is None else schedule_state.advanced(last - first)
            started = read_json(directory / ROLLOUT_FILE)
            if started is not None and started["step"] == last and started["rollout_id"] < self.next_rollout_id:
                # A restart stopped the job after Megatron saved this rollout, which the workers loaded again.
                # The job records that checkpoint instead of training the same data a second time.
                self.record_rollout(
                    directory, started["rollout_id"], method.objective, prepared, next_schedule, {}, step=last
                )
                metrics.update(rollout_id=started["rollout_id"], resumed=1)
                schedule_state = next_schedule
            else:
                rollout_id = self.next_rollout_id
                write_json(directory / ROLLOUT_FILE, {"rollout_id": rollout_id, "step": last})
                schedule_state = self.train_rollout(
                    job,
                    directory,
                    rollout_id,
                    rollout_payload,
                    algorithm,
                    schedule_state,
                    next_schedule,
                    prepared,
                    objective=method.objective,
                    step=last,
                    fresh=fresh,
                )
                metrics["rollout_id"] = rollout_id
            fresh = False
        return metrics

    def train_rollout(
        self,
        job: TrainingJob,
        directory: Path,
        rollout_id: int,
        payload: Mapping[str, Any],
        algorithm: SlimeAlgorithm,
        schedule_state: LearningRateScheduleState | None,
        next_schedule: LearningRateScheduleState | None,
        prepared: PreparedJob,
        *,
        objective: str,
        step: int,
        fresh: bool,
    ) -> LearningRateScheduleState | None:
        """Train ``payload`` as rollout ``rollout_id``, save the checkpoint pair and record it as the job's ``step``.

        ``schedule_state`` is the schedule's progress before the rollout and
        ``next_schedule`` after it. ``objective`` is the job's objective, whose
        state the saved start keeps. A ``fresh`` rollout first clears the
        workers' training state. Storage admission bounds the checkpoints on
        disk, and retention never deletes the rollout that the workers hold.
        Return the schedule progress that the checkpoint recorded.
        """
        rollout_data = to_slime_rollout_data(dict(payload))
        if self._score_centering is not None:
            # Torch, like the tensorization that follows; loaded only when the term is on.
            from reef.train.slime_backend.score_centering.heads import attach_sampler_heads

            attach_sampler_heads(rollout_data, payload, self._score_centering)
        algorithm.validate_payload(rollout_data)
        if self._storage is None:
            return self.train_and_save(
                job,
                directory,
                rollout_id,
                rollout_data,
                algorithm,
                schedule_state,
                next_schedule,
                prepared,
                objective=objective,
                step=step,
                fresh=fresh,
            )
        protected = set() if self.loaded_rollout_id is None else {self.loaded_rollout_id}
        with self._storage.admit(rollout_id=rollout_id, active_rollouts=protected) as storage_plan:
            if storage_plan["blocked"]:
                # The job fails; its data was consumed when Reef submitted it.
                raise RuntimeError(f"checkpoint storage is full: {'; '.join(storage_plan['reasons'])}")
            return self.train_and_save(
                job,
                directory,
                rollout_id,
                rollout_data,
                algorithm,
                schedule_state,
                next_schedule,
                prepared,
                objective=objective,
                step=step,
                fresh=fresh,
            )

    def train_and_save(
        self,
        job: TrainingJob,
        directory: Path,
        rollout_id: int,
        rollout_data: dict[str, Any],
        algorithm: SlimeAlgorithm,
        schedule_state: LearningRateScheduleState | None,
        next_schedule: LearningRateScheduleState | None,
        prepared: PreparedJob,
        *,
        objective: str,
        step: int,
        fresh: bool,
    ) -> LearningRateScheduleState | None:
        """Train ``rollout_data`` as rollout ``rollout_id``, save the checkpoint pair and record it.

        Return the schedule progress that the checkpoint recorded: ``next_schedule`` when the actor
        trained, else ``schedule_state``.
        """
        if fresh:
            # The weights stay; the optimizer state and the scheduler's progress start again.
            for group in (self._group, self._critic_group):
                if group is not None:
                    group.reset_training_state()
        algorithm_metrics = algorithm.prepare_rollout(rollout_data)
        worker_settings = WorkerTrainingSettings(algorithm.loss_family, schedule_state)
        if worker_settings != self.worker_settings:
            rollout_data[WORKER_SWITCH_KEY] = self.worker_switch(worker_settings)
        # Local batch processing preserves Slime's DP schedule and object-store transport: one Box per DP rank.
        packed = self._batch_processor.prepare_external_train_data(rollout_data)
        training = algorithm.train(
            rollout_id, packed, actor_group=self._group, critic_group=self._critic_group, resolve=self.resolve
        )
        metrics = next(
            (dict(result) for result in training.worker_results if isinstance(result, Mapping) and result), {}
        )
        metrics.update(merge_rank_metrics(self.resolve(self._group.async_pop_metrics())))
        metrics.update(algorithm_metrics)
        metrics.update(training.durable_metrics)
        metrics.update(algorithm.rollout_metrics(rollout_data))
        saved_schedule = schedule_state
        if training.actor_trained:
            # The actor fetched the rollout's data, and with it any worker switch.
            self.worker_settings = worker_settings
            if schedule_state is not None and next_schedule is not None:
                saved_schedule = next_schedule
                optimizer_steps = len(rollout_data["external_step_sizes"])
                metrics.update(learning_rate_metrics(schedule_state.learning_rates(optimizer_steps), next_schedule))
        self._group.save_model(rollout_id, force_sync=True)
        if (
            self._critic_group is not None
            and self._critic_save_root is not None
            and critic_checkpoint_due(rollout_id, self.args.critic_save_interval)
        ):
            # The critic's weights and optimizer persist beside the actor's, so the value head survives a restart.
            # No Hugging Face export: the critic never serves.
            self._critic_group.save_model(rollout_id, force_sync=True)
        if self._storage is not None:
            rewards = rollout_data["rewards"]
            self._storage.complete(job.job_id, rollout_id, reward=math.fsum(rewards) / len(rewards))
        self.record_rollout(directory, rollout_id, objective, prepared, saved_schedule, metrics, step=step)
        return saved_schedule

    def record_rollout(
        self,
        directory: Path,
        rollout_id: int,
        objective: str,
        prepared: PreparedJob,
        schedule_state: LearningRateScheduleState | None,
        metrics: Mapping[str, Any],
        *,
        step: int,
    ) -> None:
        """Record the saved rollout as the job's checkpoint after ``step`` optimizer steps, and as the next start."""
        hf = Path(self.args.save_hf.format(rollout_id=rollout_id))
        if hf.is_symlink() or not hf.is_dir():
            raise RuntimeError(f"checkpoint is missing or unsafe: {hf}")
        self.next_rollout_id = max(self.next_rollout_id, rollout_id + 1)
        start_dir = self.starts_root / f"rollout_{rollout_id}"
        release = hf
        if self._lora:
            # Serving loads an adapter from a release's adapter/ directory.
            release = start_dir / "release"
            shutil.rmtree(release, ignore_errors=True)
            shutil.copytree(hf, release / ADAPTER_DIR)
        state = dict(prepared.next_algorithm_state)
        SlimeStart(rollout_id, objective if state else None, state, schedule_state).write(start_dir)
        self.loaded_rollout_id = rollout_id
        self.save_checkpoint(directory, SavedCheckpoint(step, str(start_dir), str(release), metrics))

    # -- Loss families and worker switches

    def algorithm_for_loss_family(self, loss_family: str) -> SlimeAlgorithm:
        """The bound algorithm of a job's loss family; a new family is validated against the workers first.

        A family other than the startup one runs with its default driver
        options and the workers' startup arguments (see
        ``loss_family_job_args``). The first use registers its wire keys with
        the batch processor; a family that declares one of them with another
        dtype is refused.
        """
        spec = resolve_loss_family(loss_family)
        algorithm = self.algorithms_by_loss_family.get(spec.loss_family)
        if algorithm is not None:
            return algorithm
        if isinstance(spec, DistillAlgorithm) and any(
            isinstance(bound, DistillAlgorithm) for bound in self.algorithms_by_loss_family.values()
        ):
            raise RuntimeError(
                f"loss family {spec.loss_family!r} is a second distillation family in this run; "
                "a worker keeps one teacher, built from the first distillation family's settings"
            )
        # Imported here: the argument module reaches the Megatron LoRA stack (test_dependency_boundaries).
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
        self.args_by_loss_family[spec.loss_family] = job_args
        self.algorithms_by_loss_family[spec.loss_family] = algorithm
        return algorithm

    def worker_switch(self, settings: WorkerTrainingSettings) -> dict[str, Any]:
        """What the actor workers set before a job: the family's switched arguments and the schedule state."""
        loss_family_args = {}
        if self.switched_arg_names:
            values = vars(self.args_by_loss_family[settings.loss_family])
            loss_family_args = {name: values.get(name) for name in sorted(self.switched_arg_names)}
        schedule_state = settings.schedule_state
        return {
            "loss_family_args": loss_family_args,
            "learning_rate_schedule_state": None if schedule_state is None else schedule_state.to_dict(),
        }

    def resolve(self, value: Any) -> Any:
        return resolve(value, timeout=self.train_rpc_timeout_s)


# -- Building the component -----------------------------------------------------------------------


def create_train_groups(
    args, placement_groups, rollout_manager, *, train_rpc_timeout_s: float = DEFAULT_TRAIN_RPC_TIMEOUT_S
):
    from reef.train.slime_backend.reef_adapters.megatron.train_actor import ReefMegatronTrainRayActor
    from reef.train.slime_backend.reef_adapters.train_groups import create_train_groups as implementation

    return implementation(
        args,
        placement_groups,
        rollout_manager,
        actor_cls=ReefMegatronTrainRayActor,
        train_rpc_timeout_s=train_rpc_timeout_s,
    )


@dataclass(frozen=True)
class SlimeSettings:
    """What the HTTP service's runtime factory builds the Slime component from.

    ``arguments`` are the native Slime arguments, with Reef's retention
    options. ``loss_family`` is the recipe's startup loss family, by name or
    dotted reference. ``recipe`` names the recipe in error messages.
    ``execution`` holds the deployment's ``execution`` and ``executors``
    sections, which select the training workers' executor and the serving
    engines' (``rollout``) executor. ``ray_address`` and ``namespace`` name the
    Ray cluster the workers run on. ``train_rpc_timeout_s`` bounds one worker
    call, in seconds. The strings are non-empty and the timeout is positive.
    ``SlimeRuntimeFactory`` makes the value; ``create_slime_component`` reads it.
    """

    arguments: tuple[str, ...]
    loss_family: str
    recipe: str
    execution: Mapping[str, Any]
    ray_address: str
    namespace: str
    train_rpc_timeout_s: float = DEFAULT_TRAIN_RPC_TIMEOUT_S

    def __post_init__(self) -> None:
        for name, value in (
            ("loss_family", self.loss_family),
            ("recipe", self.recipe),
            ("ray_address", self.ray_address),
            ("namespace", self.namespace),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"SlimeSettings.{name} must be a non-empty string")
        if not isinstance(self.execution, Mapping) or set(self.execution) - {"execution", "executors"}:
            raise ValueError("SlimeSettings.execution holds only the execution and executors sections")
        if not self.train_rpc_timeout_s > 0:
            raise ValueError("SlimeSettings.train_rpc_timeout_s must be positive")


def create_slime_component(settings: SlimeSettings) -> SlimeTrainingComponent:
    """Parse the arguments, reserve the training GPUs, start the workers and return the component."""
    try:
        spec = resolve_loss_family(settings.loss_family)
    except UnknownLossFamilyError as exc:
        raise RuntimeError(
            f"recipe {settings.recipe!r} declares unsupported loss family {settings.loss_family!r}: {exc}"
        ) from exc
    retention, remaining = retention_options(settings.arguments)
    loss_family_config, slime_arguments = spec.parse_driver_options(remaining)
    args = parse_slime_args(slime_arguments)
    configure_executors(args, settings.execution, slime_arguments)
    validate_tracking_args(args)
    apply_bridge_resume_fallback(args)
    apply_unsaved_optimizer_resume(args)
    spec.apply_driver_options(args, loss_family_config)
    stamp_loss_family_reference(args, settings.loss_family)
    from reef.train.slime_backend.reef_adapters.megatron.lora import megatron_lora_enabled
    from reef.train.slime_backend.reef_adapters.slime_arguments import configure_reef_loss_args

    configure_reef_loss_args(args)
    spec.validate_backend_args(args, recipe=settings.recipe)
    validate_slime_args(args, spec)
    configure_megatron_runtime(args)
    configure_rollout_runtime(args)
    from reef.train.slime_backend.reef_adapters.executors.config import slime_executor_class

    slime_executor_class(args.reef_executor_backend)
    storage = prepare_checkpoint_storage(args, retention)
    resources = SlimeDeploymentResources(
        args,
        ray_address=settings.ray_address,
        namespace=settings.namespace,
        runtime_env=job_runtime_env(),
    )
    resources.start()
    try:
        actor_group, critic_group = create_train_groups(
            args, resources.placement_groups, rollout_manager=None, train_rpc_timeout_s=settings.train_rpc_timeout_s
        )
        for group in (actor_group, critic_group):
            if group is not None:
                # No engine attaches: the workers only report their parallel layout, which the batch split reads.
                group.set_rollout_manager(None)
        return SlimeTrainingComponent(
            args,
            actor_group,
            critic_group,
            batch_processor=TrainingBatchProcessor(args, actor_group.train_parallel_config),
            storage=storage,
            lora=megatron_lora_enabled(args),
            loss_family=settings.loss_family,
            loss_family_config=loss_family_config,
            score_centering=settings_from_args(args),
            train_rpc_timeout_s=settings.train_rpc_timeout_s,
            resources=resources,
        )
    except BaseException:
        resources.close()
        raise


def parse_slime_args(arguments: Sequence[str]) -> SlimeArguments:
    """Parse prepared arguments through Slime."""
    if not arguments:
        raise RuntimeError("no Slime arguments: configure training options in the deployment config")

    from reef.train.slime_backend.reef_adapters.megatron.hf_arguments import add_hf_architecture_arguments

    prepared_arguments = add_hf_architecture_arguments(arguments)
    original_argv = sys.argv
    sys.argv = [original_argv[0], *prepared_arguments]
    try:
        from slime.utils.arguments import parse_args

        from reef.train.slime_backend.reef_adapters.slime_arguments import (
            add_reef_slime_arguments,
            finalize_reef_slime_args,
        )

        args = SlimeArguments(**vars(parse_args(add_custom_arguments=add_reef_slime_arguments)))
        finalize_reef_slime_args(args, prepared_arguments)
        return args
    finally:
        sys.argv = original_argv


def configure_executors(args: SlimeArguments, execution: Mapping[str, Any], arguments: Sequence[str]) -> None:
    """Select the training executor: an explicit Slime flag, else the deployment's role setting."""
    from reef.runtime.executor.config import (
        ExecutorSettings,
        executor_settings,
        role_executor_settings,
        select_executor,
    )

    def flag_given(flag: str) -> bool:
        return any(value == flag or value.startswith(flag + "=") for value in arguments)

    def selected(settings: ExecutorSettings, role: str) -> ExecutorSettings:
        selection = select_executor(settings, role=role)
        logger.info("%s: executor=%s (%s)", role, selection.settings.backend, selection.reason)
        return selection.settings

    if flag_given("--reef-executor-backend"):
        training = selected(executor_settings(execution, args.reef_executor_backend), "training")
    else:
        training = selected(role_executor_settings(execution, "training"), "training")
    args.reef_executor_backend = training.backend
    args.reef_train_executor_options = dict(training.options)


def stamp_loss_family_reference(args: SlimeArguments, reference: str | None) -> None:
    """Carry a dotted family reference to the workers, whose registry starts empty."""
    if not reference:
        return
    dotted = reference if ":" in reference else loss_family_refs().get(reference)
    if dotted is not None:
        args.loss_family_ref = dotted


def apply_bridge_resume_fallback(args: SlimeArguments) -> None:
    """Load the Hugging Face checkpoint on a Megatron Bridge first start, and the save directory after it.

    ``--load`` always names the persistent save directory. On a first start it
    has no checkpoint tracker, and Slime's parse normalizes such a ``--load`` to
    ``ref_load``, ``None`` here; the HF checkpoint is then the source.
    """
    if args.megatron_to_hf_mode != "bridge":
        return
    load = args.load
    if isinstance(load, str) and load.strip() and (Path(load) / "latest_checkpointed_iteration.txt").is_file():
        return
    fallback = args.ref_load or args.hf_checkpoint
    if not isinstance(fallback, str) or not fallback.strip():
        raise RuntimeError(
            "Megatron Bridge first start requires --ref-load or --hf-checkpoint when --load has no checkpoint"
        )
    args.load = fallback
    args.start_rollout_id = 0


def apply_unsaved_optimizer_resume(args: SlimeArguments) -> None:
    """Resume a run that never saved its optimizer (``--no-save-optim``) from its weights alone."""
    if not args.no_save_optim:
        return
    load, save = args.load, args.save
    if not (isinstance(load, str) and load.strip() and isinstance(save, str) and save.strip()):
        return
    if Path(load).resolve() != Path(save).resolve():
        return
    if (Path(load) / "latest_checkpointed_iteration.txt").is_file():
        args.no_load_optim = True


def retention_options(arguments: Sequence[str]) -> tuple[RetentionConfig, list[str]]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, argument_default=argparse.SUPPRESS)
    parser.add_argument("--reef-checkpoint-policy", dest="policy")
    parser.add_argument("--reef-checkpoint-max-storage-fraction", dest="max_storage_fraction", type=float)
    parser.add_argument("--reef-checkpoint-min-free-space-fraction", dest="min_free_space_fraction", type=float)
    parser.add_argument("--reef-checkpoint-max-storage", dest="max_storage_bytes", type=size_bytes)
    parser.add_argument("--reef-checkpoint-min-free-space", dest="min_free_space_bytes", type=size_bytes)
    options, remaining = parser.parse_known_args(list(arguments))
    return RetentionConfig(**vars(options)), remaining


def validate_tracking_args(args: SlimeArguments) -> None:
    if args.wandb_key:
        raise RuntimeError("--wandb-key is not supported; use WANDB_API_KEY or the W&B credential store")
    if args.use_wandb:
        raise RuntimeError(
            "--use-wandb in training.slime_flags is not supported; use Reef's observability.wandb config"
        )


def size_bytes(value: str) -> int:
    match = re.fullmatch(r"(\d+)\s*(B|[KMGT]i?B)", value.strip(), re.IGNORECASE)
    if match is None:
        raise argparse.ArgumentTypeError("size must include a unit such as GB or GiB")
    unit = match[2].upper()
    size = int(match[1]) * (1024 if "I" in unit else 1000) ** "BKMGT".index(unit[0])
    if size <= 0:
        raise argparse.ArgumentTypeError("size must be positive")
    return size


def job_runtime_env(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Ray job ``runtime_env`` for the training workers: the CUDA/NCCL defaults and this process's ``PYTHONPATH``.

    Ray actors are forked from the raylet, whose environment predates the
    service. The workers get ``CUDA_DEVICE_MAX_CONNECTIONS=1`` and
    ``NCCL_NVLS_ENABLE=0`` unless this process sets them, and the service's
    ``PYTHONPATH``: cookbook loss families resolve inside the workers, so
    without it the first actor fails to import ``recipes``.
    """
    source = os.environ if environ is None else environ
    env_vars = {
        "CUDA_DEVICE_MAX_CONNECTIONS": source.get("CUDA_DEVICE_MAX_CONNECTIONS", "1"),
        "NCCL_NVLS_ENABLE": source.get("NCCL_NVLS_ENABLE", "0"),
    }
    pythonpath = source.get("PYTHONPATH", "").strip()
    if pythonpath:
        env_vars["PYTHONPATH"] = pythonpath
    return {"env_vars": env_vars}


__all__ = ["SlimeSettings", "SlimeStart", "SlimeTrainingComponent", "create_slime_component"]
