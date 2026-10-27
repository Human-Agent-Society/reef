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

import json
import math
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from reef.core.job_data import TaskSample
from reef.runtime.durable_files import read_json, write_json
from reef.runtime.executor import resolve
from reef.runtime.interfaces import (
    LearningRateScheduleState,
    learning_rate_metrics,
    resolve_learning_rate_schedule_state,
)
from reef.runtime.jobs import ReleaseFormat, SavedCheckpoint, TrainingJob
from reef.runtime.serving import ADAPTER_DIR
from reef.train.algos.registry import job_training
from reef.train.local_jobs import LocalJobComponent
from reef.train.slime_backend.algorithm import SlimeAlgorithm
from reef.train.slime_backend.data_builder import to_slime_rollout_data
from reef.train.slime_backend.distill import DistillAlgorithm
from reef.train.slime_backend.loss_families import resolve_loss_family
from reef.train.slime_backend.reef_adapters.arguments import SlimeArguments
from reef.train.slime_backend.reef_adapters.batches import WORKER_SWITCH_KEY, TrainingBatchProcessor
from reef.train.slime_backend.reef_adapters.preflight import prepare_checkpoint_storage
from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_job, slime_payload_steps
from reef.train.slime_backend.reef_adapters.startup import job_runtime_env, prepare_slime_arguments
from reef.train.slime_backend.reef_adapters.train_groups import (
    DEFAULT_TRAIN_TIMEOUT_S,
    SlimeTrainGroup,
    create_train_groups,
    merge_rank_metrics,
)
from reef.train.slime_backend.reef_adapters.training_job.storage import CheckpointStorage, critic_checkpoint_due
from reef.train.slime_backend.resources import SlimeDeploymentResources
from reef.train.slime_backend.score_centering import ScoreCenteringSettings, settings_from_args

#: A start directory's description of the weights a later job continues from.
START_STATE = "start.json"
#: The rollout a job trains and the job's optimizer steps through it, written before the workers train it.
ROLLOUT_FILE = "rollout.json"


def check_objective_state(objective: str | None, algorithm_state: Mapping[str, Any], owner: str) -> None:
    """Raise unless ``algorithm_state`` is a JSON object and ``objective`` names its objective exactly when it is non-empty."""
    if not isinstance(algorithm_state, Mapping):
        raise TypeError(f"{owner}.algorithm_state must be a mapping")
    try:
        json.dumps(dict(algorithm_state), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{owner}.algorithm_state must encode as JSON: {exc}") from exc
    if objective is not None and (not isinstance(objective, str) or not objective):
        raise ValueError(f"{owner}.objective must be a non-empty string or None")
    if (objective is None) != (not algorithm_state):
        raise ValueError(f"{owner}.objective names the objective of a non-empty state, and only of one")


def check_schedule_state(value: object, name: str) -> None:
    """Raise ``TypeError`` unless ``value`` is a ``LearningRateScheduleState`` or ``None``."""
    if value is not None and not isinstance(value, LearningRateScheduleState):
        raise TypeError(f"{name} must be a LearningRateScheduleState or None")


@dataclass(frozen=True)
class WorkerTrainingSettings:
    """The loss family and learning-rate schedule state that the actor workers train with.

    ``loss_family`` is the name of a bound loss family, non-empty.
    ``schedule_state`` is the job's schedule progress, or ``None`` for Slime's
    configured scheduler. ``SlimeTrainingComponent.train_and_save`` makes the
    value for each rollout; a value unequal to the workers' current one sends
    a worker switch with the rollout's data.
    """

    loss_family: str
    schedule_state: LearningRateScheduleState | None

    def __post_init__(self) -> None:
        if not isinstance(self.loss_family, str) or not self.loss_family:
            raise ValueError("WorkerTrainingSettings.loss_family must be a non-empty string")
        check_schedule_state(self.schedule_state, "WorkerTrainingSettings.schedule_state")


@dataclass(frozen=True)
class SlimeStart:
    """What a job continues from: the weights' rollout, the objective's state, the schedule, and whether it is fresh.

    ``rollout_id`` is the non-negative rollout whose checkpoint holds the
    weights, or ``None`` for the weights the workers loaded when they first
    started, before they saved any rollout. ``objective`` names the objective
    whose state ``algorithm_state`` (a mapping that encodes as a JSON object)
    is; it is ``None`` exactly when the state is empty. A job continues that
    state only when it trains the same objective; another objective starts
    with an empty state. ``schedule_state`` is the learning-rate schedule's
    progress, or ``None`` for Slime's configured scheduler. ``fresh`` is True
    for a start whose weights still carry training state that a job from it
    first clears: the workers' optimizer state and their scheduler's
    progress. A fresh start has an empty objective state and no schedule
    progress; ``fresh_start`` makes one. A start
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
        check_objective_state(self.objective, self.algorithm_state, "SlimeStart")
        check_schedule_state(self.schedule_state, "SlimeStart.schedule_state")
        if not isinstance(self.fresh, bool):
            raise TypeError("SlimeStart.fresh must be a bool")
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


@dataclass(frozen=True)
class SlimeRollout:
    """One rollout of a job: the run of optimizer steps that the workers train and save as one checkpoint.

    ``rollout_id`` is the non-negative id that names the rollout's Megatron
    iteration and Hugging Face export. ``step`` is the job's optimizer steps
    up to the end of the rollout, a positive int. ``schedule_state`` is the learning-rate
    schedule's progress before the rollout and ``next_schedule`` after it:
    both are ``None`` (Slime's configured scheduler), or both are set.
    ``fresh`` is True while the workers still hold the training state that
    the job's fresh start drops; the rollout clears it before its first step.
    ``objective`` and ``algorithm_state`` are the
    objective state that the rollout's start records, with the same
    condition as in ``SlimeStart``: the job's next state on its last rollout,
    and the state of the job's start on an earlier one. ``run`` makes the
    value; ``train_and_save`` and ``record_rollout`` read it.
    """

    rollout_id: int
    step: int
    schedule_state: LearningRateScheduleState | None
    next_schedule: LearningRateScheduleState | None
    fresh: bool
    objective: str | None
    algorithm_state: Mapping[str, Any]

    def __post_init__(self) -> None:
        if isinstance(self.rollout_id, bool) or not isinstance(self.rollout_id, int) or self.rollout_id < 0:
            raise ValueError("SlimeRollout.rollout_id must be a non-negative int")
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step <= 0:
            raise ValueError("SlimeRollout.step must be a positive int")
        check_schedule_state(self.schedule_state, "SlimeRollout.schedule_state")
        check_schedule_state(self.next_schedule, "SlimeRollout.next_schedule")
        if (self.schedule_state is None) != (self.next_schedule is None):
            raise ValueError("SlimeRollout has a schedule before and after the rollout, or neither")
        if not isinstance(self.fresh, bool):
            raise TypeError("SlimeRollout.fresh must be a bool")
        check_objective_state(self.objective, self.algorithm_state, "SlimeRollout")


class SlimeTrainingComponent(LocalJobComponent):
    """Train each job as Slime rollouts on the component's own workers, from its latest checkpoint.

    A job's parameters are ``method`` (a ``TrainingMethod`` as a dict),
    optionally ``scheduling`` (a ``StepScheduling`` as a dict, see
    ``job_training``), and optionally ``save_interval``, a positive int: the
    optimizer steps between two saved checkpoints. ``submit`` refuses any
    other parameter. Each run of ``save_interval`` steps trains as one Slime
    rollout and saves one checkpoint; without it, the job is one rollout with
    one checkpoint. A checkpoint's step is the job's optimizer steps up to it.
    The job's last checkpoint records the objective's state after the job's
    data; an earlier checkpoint records the state of the job's start, since
    the objective computes its state for the whole data. A job whose objective
    skips its data saves no checkpoint, so the next job continues the start's
    weights and objective state, as on every training component.

    The workers hold one set of weights, so the component accepts only a job
    whose start names the weights the workers hold: the latest saved
    rollout's start or a fresh start of it, or, until the workers save their
    first rollout, a ``None`` start or a fresh start of it, which name the
    weights the workers loaded at startup. ``submit`` checks the start, and
    ``run`` checks it again, since a job queued behind another job runs after
    that job moved the workers on. The component refuses every release as a
    start, and it trains the jobs of one scenario. ``fresh_start`` keeps the
    weights and drops the training state: the optimizer state, the
    scheduler's progress and the objective's state.

    ``args`` are the arguments the workers started with; ``storage`` bounds
    the paired checkpoints on disk, and has a critic root exactly when a
    critic group trains. Beside the checkpoint roots,
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
        storage: CheckpointStorage,
        lora: bool,
        loss_family: str,
        loss_family_config: object | None = None,
        score_centering: ScoreCenteringSettings | None = None,
        train_timeout_s: float = DEFAULT_TRAIN_TIMEOUT_S,
        resources: SlimeDeploymentResources | None = None,
    ) -> None:
        self.args = args
        self.train_timeout_s = train_timeout_s
        self.group = actor_group
        if (critic_group is None) != (storage.critic_root is None):
            raise ValueError("the critic group saves into the storage's critic root: give both or neither")
        self.critic_group = critic_group
        self.batch_processor = batch_processor
        self.storage = storage
        self.lora = lora
        self.score_centering = score_centering
        self.resources = resources
        self.critic_steps_per_actor = args.critic_steps_per_actor
        self.critic_only_steps = args.num_critic_only_steps
        algorithm = resolve_loss_family(loss_family).bind(
            loss_family_config,
            critic_steps_per_actor=self.critic_steps_per_actor,
            critic_only_steps=self.critic_only_steps,
        )
        # Bound algorithms and each family's worker arguments, by family name;
        # the startup family's arguments are the workers' own.
        self.algorithms_by_loss_family: dict[str, SlimeAlgorithm] = {algorithm.loss_family: algorithm}
        self.args_by_loss_family: dict[str, SlimeArguments] = {algorithm.loss_family: args}
        # Arguments some family changes from the startup ones: a worker switch sends all of them.
        self.switched_arg_names: set[str] = set()
        self.worker_settings = WorkerTrainingSettings(algorithm.loss_family, None)
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
        if not self.lora:
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
        if any(isinstance(item, TaskSample) for item in job.data.samples):
            raise ValueError("Slime trains trajectories; it does not roll tasks out (task jobs) yet")
        unknown = set(job.parameters) - {"method", "scheduling", "save_interval"}
        if unknown:
            raise ValueError(f"Slime jobs accept no parameters {sorted(unknown)}")
        interval = job.parameters.get("save_interval")
        if interval is not None and (isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0):
            raise ValueError("save_interval must be a positive int")
        job_training(job.parameters)
        self.check_held_weights(None if job.start is None else SlimeStart.read(Path(job.start)).rollout_id, job.start)

    def check_held_weights(self, rollout_id: int | None, start: str | None) -> None:
        """Raise ``ValueError`` unless the workers hold the weights of ``rollout_id``, which ``start`` names.

        ``None`` names the weights the workers loaded at startup, which they
        hold only until they save their first rollout.
        """
        if rollout_id != self.loaded_rollout_id:
            held = (
                "the startup weights"
                if self.loaded_rollout_id is None
                else f"the weights of rollout {self.loaded_rollout_id}"
            )
            wanted = "the startup weights" if rollout_id is None else f"rollout {rollout_id}"
            raise ValueError(f"Slime's workers hold {held}; a job cannot start from {wanted} ({start})")

    def close_resources(self) -> None:
        errors = []
        for group in (self.critic_group, self.group):
            if group is not None:
                try:
                    group.release()
                except Exception as exc:
                    errors.append(exc)
        if self.resources is not None:
            try:
                self.resources.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]

    # -- One job

    def run(self, job: TrainingJob, directory: Path) -> Mapping[str, Any]:
        start = None if job.start is None else SlimeStart.read(Path(job.start))
        method, scheduling = job_training(job.parameters)
        saved = self.saved(directory)
        started = read_json(directory / ROLLOUT_FILE)
        # The workers hold the job's last saved rollout, or its start before it saved one. After a restart that
        # stopped the job once Megatron saved a rollout, they hold that rollout, which the loop below records.
        if started is None or started["rollout_id"] != self.loaded_rollout_id:
            held = SlimeStart.read(Path(saved[-1].path)) if saved else start
            self.check_held_weights(None if held is None else held.rollout_id, job.start)
        start_state = {} if start is None else start.state_for(method.objective)
        prepared = prepare_slime_job(
            job.data,
            method,
            start_state,
            scheduling,
            step_size=self.args.global_batch_size,
            sampler_topk=self.score_centering is not None,
        )
        next_state = dict(prepared.next_algorithm_state)
        payload = prepared.payload
        if payload is None:
            # The objective skipped the job's data: the job saves no checkpoint.
            return prepared.metrics
        algorithm = self.algorithm_for_loss_family(payload["loss"])
        algorithm.validate_payload(payload)
        if saved:
            # A resumed job continues after its last saved checkpoint, with that checkpoint's schedule progress.
            schedule_state = SlimeStart.read(Path(saved[-1].path)).schedule_state
        else:
            schedule_state = resolve_learning_rate_schedule_state(
                None if start is None else start.schedule_state, method.learning_rate_schedule
            )
        if schedule_state is not None and self.args.decoupled_lr is not None:
            raise RuntimeError("a recipe-selected learning-rate schedule cannot drive --decoupled-lr parameter groups")
        step_count = len(payload["external_step_sizes"])
        interval = job.parameters.get("save_interval") or step_count
        done = saved[-1].step if saved else 0
        runs = [(first, min(first + interval, step_count)) for first in range(done, step_count, interval)]
        # Until the job saves its first rollout, the workers hold the start's training state: a fresh job resets it.
        fresh = start is not None and start.fresh and not saved
        metrics: dict[str, Any] = dict(prepared.metrics)
        for first, last in runs:
            if self.stopping(directory):
                break
            # Only the job's last rollout has trained all the data that the objective's next state is for.
            if last == step_count:
                objective, state = (method.objective if next_state else None), next_state
            elif start is not None:
                objective, state = start.objective, dict(start.algorithm_state)
            else:
                objective, state = None, {}
            # Optimizer steps advance only a recipe-selected schedule.
            next_schedule = None if schedule_state is None else schedule_state.advanced(last - first)
            training = (algorithm, slime_payload_steps(payload, first, last))
            rollout = SlimeRollout(self.next_rollout_id, last, schedule_state, next_schedule, fresh, objective, state)
            started = read_json(directory / ROLLOUT_FILE)
            if started is not None and started["step"] == last and started["rollout_id"] < self.next_rollout_id:
                # A restart stopped the job after Megatron saved this rollout, which the workers loaded again.
                # The job records that checkpoint instead of training the same data a second time.
                rollout = replace(rollout, rollout_id=started["rollout_id"])
                rewards = to_slime_rollout_data(dict(training[1]))["rewards"]
                self.storage.complete(job.job_id, rollout.rollout_id, reward=math.fsum(rewards) / len(rewards))
                self.record_rollout(directory, rollout, next_schedule, {})
                metrics.update(rollout_id=rollout.rollout_id, resumed=1)
                schedule_state = next_schedule
            else:
                write_json(directory / ROLLOUT_FILE, {"rollout_id": rollout.rollout_id, "step": last})
                schedule_state = self.train_and_save(job, directory, rollout, training)
                metrics["rollout_id"] = rollout.rollout_id
            fresh = False
        return metrics

    def train_and_save(
        self,
        job: TrainingJob,
        directory: Path,
        rollout: SlimeRollout,
        training: tuple[SlimeAlgorithm, Mapping[str, Any]],
    ) -> LearningRateScheduleState | None:
        """Train ``rollout``, save its checkpoint pair and record it as the job's checkpoint and the next start.

        ``training`` is the job's algorithm and the payload of the rollout's
        optimizer steps. Storage
        admission bounds the checkpoints on disk, and retention never deletes
        the rollout that the workers hold. Return the schedule progress that
        the checkpoint records: ``rollout.next_schedule`` when the actor
        trained, else ``rollout.schedule_state``.
        """
        algorithm, payload = training
        rollout_data = to_slime_rollout_data(dict(payload))
        if self.score_centering is not None:
            # Torch, like the tensorization that follows; loaded only when the term is on.
            from reef.train.slime_backend.score_centering.heads import attach_sampler_heads

            attach_sampler_heads(rollout_data, payload, self.score_centering)
        algorithm.validate_payload(rollout_data)
        protected = set() if self.loaded_rollout_id is None else {self.loaded_rollout_id}
        with self.storage.admit(rollout_id=rollout.rollout_id, active_rollouts=protected) as storage_plan:
            if storage_plan["blocked"]:
                # The job fails; its data was consumed when Reef submitted it.
                raise RuntimeError(f"checkpoint storage is full: {'; '.join(storage_plan['reasons'])}")
            saved_schedule = rollout.schedule_state
            if rollout.fresh:
                # The weights stay; the optimizer state and the scheduler's progress start again.
                for group in (self.group, self.critic_group):
                    if group is not None:
                        group.reset_training_state()
            algorithm_metrics = algorithm.prepare_rollout(rollout_data)
            worker_settings = WorkerTrainingSettings(algorithm.loss_family, rollout.schedule_state)
            if worker_settings != self.worker_settings:
                rollout_data[WORKER_SWITCH_KEY] = self.worker_switch(worker_settings)
            # Local batch processing preserves Slime's DP schedule and object-store transport: one Box per DP rank.
            packed = self.batch_processor.prepare_external_train_data(rollout_data)
            result = algorithm.train(
                rollout.rollout_id,
                packed,
                actor_group=self.group,
                critic_group=self.critic_group,
                resolve=self.resolve,
            )
            metrics = next(
                (dict(value) for value in result.worker_results if isinstance(value, Mapping) and value), {}
            )
            metrics.update(merge_rank_metrics(self.resolve(self.group.async_pop_metrics())))
            metrics.update(algorithm_metrics)
            metrics.update(result.durable_metrics)
            metrics.update(algorithm.rollout_metrics(rollout_data))
            if result.actor_trained:
                # The actor fetched the rollout's data, and with it any worker switch.
                self.worker_settings = worker_settings
                if rollout.schedule_state is not None and rollout.next_schedule is not None:
                    saved_schedule = rollout.next_schedule
                    optimizer_steps = len(rollout_data["external_step_sizes"])
                    metrics.update(
                        learning_rate_metrics(
                            rollout.schedule_state.learning_rates(optimizer_steps), rollout.next_schedule
                        )
                    )
            rewards = rollout_data["rewards"]
            reward = math.fsum(rewards) / len(rewards)
            self.group.save_model(rollout.rollout_id, force_sync=True)
            if self.critic_group is not None and critic_checkpoint_due(
                rollout.rollout_id, self.storage.critic_save_interval
            ):
                # The critic's weights and optimizer persist beside the actor's, so the value head survives a restart.
                # No Hugging Face export: the critic never serves.
                self.critic_group.save_model(rollout.rollout_id, force_sync=True)
            self.storage.complete(job.job_id, rollout.rollout_id, reward=reward)
            self.record_rollout(directory, rollout, saved_schedule, metrics)
        return saved_schedule

    def record_rollout(
        self,
        directory: Path,
        rollout: SlimeRollout,
        schedule_state: LearningRateScheduleState | None,
        metrics: Mapping[str, Any],
    ) -> None:
        """Record the saved ``rollout`` as the job's checkpoint at ``rollout.step`` and as the next start.

        The start records ``schedule_state`` and the rollout's objective state. It is not fresh: the rollout
        trained, which cleared the workers' training state when the rollout was fresh.
        """
        hf = Path(self.args.save_hf.format(rollout_id=rollout.rollout_id))
        if hf.is_symlink() or not hf.is_dir():
            raise RuntimeError(f"checkpoint is missing or unsafe: {hf}")
        self.next_rollout_id = max(self.next_rollout_id, rollout.rollout_id + 1)
        start_dir = self.starts_root / f"rollout_{rollout.rollout_id}"
        release = hf
        if self.lora:
            # Serving loads an adapter from a release's adapter/ directory.
            release = start_dir / "release"
            shutil.rmtree(release, ignore_errors=True)
            shutil.copytree(hf, release / ADAPTER_DIR)
        SlimeStart(
            rollout.rollout_id,
            rollout.objective,
            rollout.algorithm_state,
            schedule_state,
        ).write(start_dir)
        self.loaded_rollout_id = rollout.rollout_id
        self.save_checkpoint(directory, SavedCheckpoint(rollout.step, str(start_dir), str(release), metrics))

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
        try:
            # The batch processor partitions and tensorizes the keys a job's data carries.
            self.batch_processor.register_keys(
                tuple(job_args.custom_rollout_data_keys or ()), job_args.reef_rollout_tensor_dtypes
            )
        except ValueError as exc:
            raise RuntimeError(
                f"loss family {spec.loss_family!r} conflicts with another family of this run: {exc}"
            ) from exc
        startup = vars(self.args)
        self.switched_arg_names.update(
            name for name, value in vars(job_args).items() if name not in startup or startup[name] != value
        )
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
        return resolve(value, timeout=self.train_timeout_s)


# -- Building the component -----------------------------------------------------------------------


@dataclass(frozen=True)
class SlimeSettings:
    """What the HTTP service's runtime factory builds the Slime component from.

    ``arguments`` are the native Slime arguments, with Reef's retention
    options. ``loss_family`` is the recipe's startup loss family, by name or
    dotted reference. ``recipe`` names the recipe in error messages.
    ``execution`` holds the deployment's ``execution`` and ``executors``
    sections, which select the training workers' executor. ``ray_address``
    and ``namespace`` name the Ray cluster the workers run on.
    ``train_timeout_s`` bounds one worker call, in seconds. The strings are
    non-empty and the timeout is positive. ``SlimeComponentFactory`` makes the
    value; ``create_slime_component`` reads it.
    """

    arguments: tuple[str, ...]
    loss_family: str
    recipe: str
    execution: Mapping[str, Any]
    ray_address: str
    namespace: str
    train_timeout_s: float = DEFAULT_TRAIN_TIMEOUT_S

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
        if isinstance(self.train_timeout_s, bool) or not isinstance(self.train_timeout_s, int | float):
            raise TypeError("SlimeSettings.train_timeout_s must be a number")
        if not self.train_timeout_s > 0:
            raise ValueError("SlimeSettings.train_timeout_s must be positive")


def create_slime_component(settings: SlimeSettings) -> SlimeTrainingComponent:
    """Prepare the arguments, reserve the training GPUs, start the workers and return the component."""
    args, loss_family_config, retention = prepare_slime_arguments(
        settings.arguments, loss_family=settings.loss_family, recipe=settings.recipe, execution=settings.execution
    )
    storage = prepare_checkpoint_storage(args, retention)
    resources = SlimeDeploymentResources(
        args,
        ray_address=settings.ray_address,
        namespace=settings.namespace,
        runtime_env=job_runtime_env(),
    )
    resources.start()
    try:
        # Imported here: the worker class loads torch, Megatron and Slime's Ray actor.
        from reef.train.slime_backend.reef_adapters.megatron.lora import megatron_lora_enabled
        from reef.train.slime_backend.reef_adapters.megatron.train_actor import ReefMegatronTrainRayActor

        actor_group, critic_group = create_train_groups(
            args,
            resources.placement_groups,
            actor_cls=ReefMegatronTrainRayActor,
            train_timeout_s=settings.train_timeout_s,
        )
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
            train_timeout_s=settings.train_timeout_s,
            resources=resources,
        )
    except BaseException:
        resources.close()
        raise


__all__ = ["SlimeSettings", "SlimeStart", "SlimeTrainingComponent", "create_slime_component"]
