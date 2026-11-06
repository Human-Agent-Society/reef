"""Slime as a training component: each job runs in its own driver process, on Slime's own placement.

``SlimeTrainingComponent`` lives in the HTTP service's process and keeps the
job contract: it checks and queues jobs, and for each job starts a driver
(``reef.train.slime_backend.driver``) on the Ray cluster, which reports the
job's checkpoints in the job's directory. It holds no GPU itself.

The driver runs ``SlimeJobRunner``: it starts Slime's actor (and critic)
workers and, for a dataset job or a task job, attaches them to Slime's own
rollout engines (``reef.train.slime_backend.rollout_group``). The engines are
those of Slime's serving session, which the first such job starts and which
holds the engines and their GPUs until the component closes; while it runs,
every job trains on its GPUs. Without a session, a job places itself with
Slime's own placement logic and holds its GPUs only while it runs. A job of
trajectories trains as Slime rollouts, one by default or one per
``save_interval`` optimizer steps: Reef's job data becomes Slime's rollout data,
the workers train its optimizer steps, and the runner saves the paired Hugging
Face and Megatron checkpoints of each rollout. A dataset job and a task job run
Slime's own loop: the engines take the workers' weights, Slime generates a
rollout, and the workers train it (``run_slime_loop``). With ``--colocate`` the
engines share the training GPUs, and Slime moves the GPU memory between them
and the workers. Reef's serving engines take no part; Reef serves the Hugging
Face export (or, with LoRA, the PEFT adapter) of a checkpoint that passes
evaluation.

Training continues from Slime's latest checkpoint: every driver loads the
latest Megatron checkpoint of one save root. A job therefore starts from the
component's latest checkpoint. A new training start from another snapshot
(``train-from``) is refused, as Slime cannot load a published snapshot back
into its workers.

A job's parameters are ``SlimeJobParameters``. Every job trains the loss family
of its objective and the learning-rate schedule it selects. The workers start
with the startup family, that of the recipe's default objective; a job of another
family is validated against the workers' startup arguments before it trains,
and the actor workers switch to it (and to the job's schedule) when they
receive the job's data.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
import time
import uuid
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from reef.artifact.repository import DefaultStart, ModelDirectory
from reef.core.chat_rendering import ChatTemplate, TokenizerChatTemplate, needs_rendering, rendered_job_data
from reef.core.job_data import TaskSample
from reef.core.weight_layout import ADAPTER_DIR, ArtifactFormat
from reef.execution.executor import resolve
from reef.storage.durable_files import read_json, write_json
from reef.storage.job_data import JobDataKind, read_job_data
from reef.train.algos.schedule import StepScheduling
from reef.train.component import SavedCheckpoint, ScenarioRoot, TrainingJob
from reef.train.learning_rate import (
    LearningRateScheduleState,
    learning_rate_metrics,
    resolve_learning_rate_schedule_state,
)
from reef.train.local_jobs import LocalJobComponent
from reef.train.slime_backend.algorithm import SlimeAlgorithm
from reef.train.slime_backend.checkpoints import (
    START_STATE,
    SlimeRound,
    SlimeStart,
    check_schedule_state,
    write_rollout_start,
)
from reef.train.slime_backend.data_builder import to_slime_rollout_data
from reef.train.slime_backend.dataset import DATASET_FILE, DatasetSettings, save_root
from reef.train.slime_backend.distill import DistillAlgorithm
from reef.train.slime_backend.harbor_rollouts import (
    TaskJobSource,
    check_rollout_can_train,
    check_task_job_options,
    has_group_baseline,
    read_task_rollout_results,
    task_job_metrics,
)
from reef.train.slime_backend.loss_families import UnknownLossFamilyError, resolve_loss_family
from reef.train.slime_backend.objectives import resolve_slime_objective
from reef.train.slime_backend.parameters import SlimeJobParameters, slime_parameters_schema
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
from reef.train.slime_backend.resources import SlimeDeploymentResources, retire_lease
from reef.train.slime_backend.rollout_group import (
    SERVING_FILE,
    SGLANG_PLUGIN_ENV,
    SlimeRolloutGroup,
    dispose_serving,
    serving_exists,
    serving_session,
)
from reef.train.slime_backend.score_centering import ScoreCenteringSettings, settings_from_args
from reef.train.task_rollouts import HarborSettings, task_name

#: The rollout a job trains and the job's optimizer steps through it, written before the workers train it.
ROLLOUT_FILE = "rollout.json"
#: The settings a job's driver starts with, in the job's directory.
DRIVER_FILE = "driver.json"
#: The metrics of a job whose driver finished, in the job's directory.
DRIVER_RESULT_FILE = "driver-result.json"
#: The output of a job's drivers, in the job's directory.
DRIVER_LOG = "driver.log"
#: Seconds a driver has to stop after SIGTERM before it is killed.
DRIVER_STOP_TIMEOUT_S = 60
#: The output of the disposal of the component's Slime serving session, in the save root.
SERVING_LOG = "reef-serving.log"
#: The loss family that leaves the training signal to Slime (``reef.train.slime_backend.native``).
SLIME_NATIVE_FAMILY = "slime"
#: The Slime option that starts no rollout engine, for a job that rolls nothing out.
NO_ROLLOUT_ENGINES = "--rollout-num-gpus=0"


@dataclass(frozen=True)
class WorkerTrainingSettings:
    """The loss family and learning-rate schedule state that the actor workers train with.

    ``loss_family`` is the name of a bound loss family, non-empty.
    ``schedule_state`` is the job's schedule progress, or ``None`` for Slime's
    configured scheduler. ``SlimeJobRunner.train_and_save`` makes the
    value for each rollout; a value unequal to the workers' current one sends
    a worker switch with the rollout's data.
    """

    loss_family: str
    schedule_state: LearningRateScheduleState | None

    def __post_init__(self) -> None:
        if not isinstance(self.loss_family, str) or not self.loss_family:
            raise ValueError("WorkerTrainingSettings.loss_family must be a non-empty string")
        check_schedule_state(self.schedule_state, "WorkerTrainingSettings.schedule_state")


class SlimeJobs(LocalJobComponent):
    """The jobs of one Slime save root: their directories, the starts they continue from, and the snapshots.

    ``args`` are Slime's parsed arguments of the deployment, ``lora`` whether
    the actor trains a LoRA adapter. Beside the checkpoint roots, ``reef-jobs/``
    holds the jobs (see ``LocalJobComponent``) and ``reef-starts/rollout_<id>/``
    the start of each saved rollout: its ``start.json`` and, with LoRA, the
    published snapshot in ``release/adapter/``; ``reef-starts/base/`` is the
    start of the untrained weights. ``SlimeTrainingComponent`` (the component
    Reef calls) and ``SlimeJobRunner`` (the driver that trains a job) are the
    two subclasses, over the same directories. A subclass names the rollout
    whose weights the next job continues, ``held_rollout_id``.
    """

    def __init__(self, args: SlimeArguments, *, lora: bool) -> None:
        self.args = args
        self.lora = lora
        root = self.root_of(args)
        self.root = root
        self.starts_root = root / "reef-starts"
        # The start of the startup weights, the untrained checkpoint that the default start holds.
        self.base_start_dir = self.starts_root / "base"
        if not (self.base_start_dir / START_STATE).exists():
            SlimeStart(None, None, {}, None).write(self.base_start_dir)
        super().__init__(root / "reef-jobs")

    @staticmethod
    def root_of(args: SlimeArguments) -> Path:
        """The save root of ``args``: the directory of the checkpoint roots, the jobs and the starts."""
        return save_root(args.save_hf)

    @property
    @abstractmethod
    def held_rollout_id(self) -> int | None:
        """The rollout whose weights the next job continues, or ``None`` for the untrained weights."""

    @property
    def name(self) -> str:
        return "slime"

    @property
    def artifact_format(self) -> ArtifactFormat:
        """Full weights (the rollout's Hugging Face export), or with LoRA the PEFT adapter of the workers' rank."""
        if not self.lora:
            return ArtifactFormat("full")
        from reef.train.slime_backend.reef_adapters.megatron.lora import sglang_lora_target_modules

        return ArtifactFormat(
            "adapter", int(self.args.megatron_lora_rank), tuple(sglang_lora_target_modules(self.args))
        )

    @property
    def default_start_inputs(self) -> dict[str, str]:
        """The configuration that decides the startup weights: the model, and with LoRA the adapter's settings."""
        model = Path(self.args.hf_checkpoint).expanduser().resolve()
        if not self.lora:
            return {"model": str(model)}
        from reef.train.slime_backend.reef_adapters.megatron.lora import sglang_lora_target_modules

        return {
            "model": str(model),
            "lora_rank": str(self.args.megatron_lora_rank),
            "lora_alpha": str(self.args.megatron_lora_alpha),
            "lora_target_modules": ",".join(sorted(sglang_lora_target_modules(self.args))),
        }

    def default_start(self) -> DefaultStart:
        """The startup weights as a snapshot: the model directory, or with LoRA an adapter whose weights are zero.

        Full-weight artifacts are Hugging Face checkpoints, so the default start
        is ``hf_checkpoint`` itself. LoRA artifacts hold the adapter in
        ``adapter/``, so the default start holds the untrained adapter there:
        every LoRA weight is zero, so it serves the startup weights unchanged.
        """
        inputs = self.default_start_inputs
        if not self.lora:
            return DefaultStart(model_files=ModelDirectory(Path(inputs["model"])), inputs=inputs)
        from reef.train.slime_backend.reef_adapters.megatron.lora_checkpoint import save_untrained_lora_adapter

        artifact_dir = self.base_start_dir / "release"
        if not (artifact_dir / ADAPTER_DIR).is_dir():
            save_untrained_lora_adapter(self.args, artifact_dir / ADAPTER_DIR)
        return DefaultStart(model_files=ModelDirectory(artifact_dir), inputs=inputs)

    @property
    def base_start(self) -> str:
        """The start of the startup weights, which the workers load when they first start."""
        return str(self.base_start_dir)

    def root_start(self, root: ScenarioRoot) -> str:
        """The startup weights, when ``root`` holds the default start of this configuration; else ``ValueError``.

        Slime's workers load ``hf_checkpoint`` when they start and cannot load a snapshot, so they train a scenario
        only when its root came from the same configured inputs.
        """
        configured = self.default_start_inputs
        if root.source.kind != "default" or dict(root.source.inputs) != configured:
            started = (
                f"the item {root.source.item!r}"
                if root.source.kind == "item"
                else f"the inputs {dict(root.source.inputs)!r}"
            )
            raise ValueError(
                f"Slime trains the weights that it starts with, from the inputs {configured!r}, but the scenario's root "
                f"release came from {started}; create the scenario again with this configuration"
            )
        return self.base_start

    def import_start(self, artifact_dir: Path) -> str:
        raise ValueError(
            "Slime continues training from its own latest checkpoint; it cannot load a published snapshot "
            "into its workers"
        )

    def parameters_schema(self, kind: JobDataKind) -> dict[str, Any]:
        return slime_parameters_schema(kind)

    def check_job(self, job: TrainingJob) -> None:
        parameters = SlimeJobParameters.from_parameters(job.parameters)
        dataset_family = self.args.loss_family == SLIME_NATIVE_FAMILY
        if parameters.dataset is not None:
            if not dataset_family:
                raise ValueError(f"a dataset job needs the {SLIME_NATIVE_FAMILY!r} loss family, which Slime trains")
            if job.sample_count:
                raise ValueError("a dataset job has no samples: Slime rolls out the dataset's prompts")
        elif dataset_family and job.kind != "task":
            raise ValueError(f"the {SLIME_NATIVE_FAMILY!r} loss family trains only dataset jobs and task jobs")
        if job.kind == "task":
            tasks = [item for item in read_job_data(job.data_dir, job.data_id).samples if isinstance(item, TaskSample)]
            self.check_task_jobs(job.parameters)
            names = [task_name(task) for task in tasks]
            if len(set(names)) != len(names):
                raise ValueError("the tasks of one job need distinct names: each name groups its task's samples")
        elif parameters.harbor is not None:
            raise ValueError("the harbor parameter applies only to a job of tasks")
        loss_family = resolve_slime_objective(parameters.objective).loss_family
        try:
            resolve_loss_family(loss_family)
        except UnknownLossFamilyError as exc:
            raise ValueError(f"Slime has no loss family {loss_family!r} for the job's objective: {exc}") from exc
        self.check_held_weights(SlimeStart.read(Path(job.start)).rollout_id, job.start)

    def check_held_weights(self, rollout_id: int | None, start: str) -> None:
        """Raise ``ValueError`` unless the next job continues the weights of ``rollout_id``, which ``start`` names.

        ``None`` names the untrained weights, which a job continues only until the first rollout is saved.
        """
        held_rollout_id = self.held_rollout_id
        if rollout_id != held_rollout_id:
            held = "the startup weights" if held_rollout_id is None else f"the weights of rollout {held_rollout_id}"
            wanted = "the startup weights" if rollout_id is None else f"rollout {rollout_id}"
            raise ValueError(f"Slime continues {held}; a job cannot start from {wanted} ({start})")

    def check_task_jobs(self, parameters: Mapping[str, Any]) -> None:
        """A task job runs Slime's own loop, with Harbor rollouts as its generate step, as ``harbor`` says.

        Slime computes its advantages and trains its loss, so the job's objective has the ``slime`` loss family, and
        Slime's options must leave that loop's generate step and data source to the job (``check_task_job_options``).
        ``ValueError``: the parameters or the options do not allow it.
        """
        job_parameters = SlimeJobParameters.from_parameters(parameters)
        if job_parameters.harbor is None:
            raise ValueError("a task job needs the harbor parameter as an object")
        family = resolve_slime_objective(job_parameters.objective).loss_family
        if family != SLIME_NATIVE_FAMILY:
            raise ValueError(
                f"a task job's objective needs the {SLIME_NATIVE_FAMILY!r} loss family, not {family!r}; "
                "name slime_dataset, or a method on that family"
            )
        if job_parameters.learning_rate_schedule is not None or job_parameters.scheduling != StepScheduling():
            raise ValueError(
                "a task job trains on Slime's own loop: Slime's options set its learning rate and optimizer steps, "
                "so remove learning_rate_schedule and scheduling"
            )
        check_task_job_options(self.args)


class SlimeJobRunner(SlimeJobs):
    """Train a job as Slime rollouts on the runner's workers, from Slime's latest checkpoint.

    A job's driver makes one runner and calls ``run`` for its job; the CPU
    tests run it as an in-process component. A job's parameters are
    ``SlimeJobParameters``. Each
    run of ``save_interval`` steps trains as one Slime rollout and saves one
    checkpoint; without it, the job is one rollout with one checkpoint. A
    checkpoint's step is the job's optimizer steps up to it. The job's last
    checkpoint records the objective's state after the job's data; an
    earlier checkpoint records the state of the job's start, since the
    objective computes its state for the whole data. A job whose objective
    skips its data saves no checkpoint, so the next job continues the start's
    weights and objective state, as on every training component.

    The workers load Slime's latest checkpoint, so a job's start must name
    it: the latest saved rollout's start, or, before the first rollout, the
    base start. ``run`` checks it again, since a job queued behind another
    job runs after that job moved the checkpoint on.

    A dataset job and a task job need ``rollout_group`` and run Slime's own
    loop (``run_slime_loop``). A task job reports each task's rewards of the
    rollouts that ran (``rollout_rewards``, by task name).

    ``args`` are the arguments the workers started with; ``storage`` bounds
    the paired checkpoints on disk, and has a critic root exactly when a
    critic group trains. ``rollout_group`` is Slime's rollout engines of a
    dataset job or a task job, or None. ``resources`` is the job's Ray session
    and GPUs, which ``close`` releases.
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
        rollout_group: SlimeRolloutGroup | None = None,
    ) -> None:
        self.rollout_group = rollout_group
        self.train_timeout_s = train_timeout_s
        #: The chat template of the trained model, made at the first job that renders a trajectory.
        self.template: ChatTemplate | None = None
        self.group = actor_group
        if (critic_group is None) != (storage.critic_root is None):
            raise ValueError("the critic group saves into the storage's critic root: give both or neither")
        self.critic_group = critic_group
        self.batch_processor = batch_processor
        self.storage = storage
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
        self.next_rollout_id = int(args.start_rollout_id or 0)
        # The workers loaded the checkpoint of the rollout before the next one, or the startup weights. A job
        # that a restart stopped after Megatron saved that rollout writes that rollout's start when it resumes.
        self.loaded_rollout_id: int | None = self.next_rollout_id - 1 if self.next_rollout_id > 0 else None
        super().__init__(args, lora=lora)

    def check_task_jobs(self, parameters: Mapping[str, Any]) -> None:
        """A task job rolls its tasks out on this runner's rollout engines, as the ``harbor`` parameter says."""
        if self.rollout_group is None:
            raise ValueError(
                "Slime started no rollout engines for this task job; set training.options.colocate or rollout-num-gpus"
            )
        super().check_task_jobs(parameters)

    @property
    def held_rollout_id(self) -> int | None:
        return self.loaded_rollout_id

    def chat_template(self) -> ChatTemplate:
        """The chat template of ``--hf-checkpoint``'s tokenizer, with ``--apply-chat-template-kwargs``, which must be
        the template arguments that the serving engine renders with."""
        if self.template is None:
            # transformers is part of the training environment.
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
            self.template = TokenizerChatTemplate(tokenizer, self.args.apply_chat_template_kwargs or {})
        return self.template

    def close_resources(self) -> None:
        errors = []
        for group in (self.critic_group, self.group):
            if group is not None:
                try:
                    group.release()
                except Exception as exc:
                    errors.append(exc)
        # After the workers: the serving session releases whatever trainers are still attached to it.
        if self.rollout_group is not None:
            try:
                self.rollout_group.close()
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
        parameters = SlimeJobParameters.from_parameters(job.parameters)
        if parameters.dataset is not None:
            return self.run_dataset(job, directory, parameters.dataset, parameters.save_interval)
        if parameters.harbor is not None:
            return self.run_task_job(job, directory, parameters.harbor, parameters.save_interval)
        start = SlimeStart.read(Path(job.start))
        saved = self.saved(directory)
        started = read_json(directory / ROLLOUT_FILE)
        # The workers hold the job's last saved rollout, or its start before it saved one. After a restart that
        # stopped the job once Megatron saved a rollout, they hold that rollout, which the loop below records.
        if started is None or started["rollout_id"] != self.loaded_rollout_id:
            held = SlimeStart.read(Path(saved[-1].path)) if saved else start
            self.check_held_weights(held.rollout_id, job.start)
        data = read_job_data(job.data_dir, job.data_id)
        rollout_metrics: dict[str, Any] = {}
        # A trajectory that serving recorded carries no tokens: the component renders it with the trained model's
        # chat template, and the actor computes its log-probabilities with the weights of this job's start.
        if needs_rendering(data):
            data, left_out = rendered_job_data(data, self.chat_template())
            if left_out:
                rollout_metrics["unrenderable_samples"] = len(left_out)
                if not data.samples:
                    return rollout_metrics
        start_state = start.state_for(parameters.objective)
        prepared = prepare_slime_job(
            data,
            parameters.objective,
            start_state,
            parameters.scheduling,
            step_size=self.args.global_batch_size,
            sampler_topk=self.score_centering is not None,
        )
        next_state = dict(prepared.next_algorithm_state)
        payload = prepared.payload
        if payload is None:
            # The objective skipped the job's data: the job saves no checkpoint.
            return {**prepared.metrics, **rollout_metrics}
        algorithm = self.algorithm_for_loss_family(payload["loss"])
        algorithm.validate_payload(payload)
        if saved:
            # A resumed job continues after its last saved checkpoint, with that checkpoint's schedule progress.
            schedule_state = SlimeStart.read(Path(saved[-1].path)).schedule_state
        else:
            schedule_state = resolve_learning_rate_schedule_state(
                start.schedule_state, parameters.learning_rate_schedule
            )
        if schedule_state is not None and self.args.decoupled_lr is not None:
            raise RuntimeError("a recipe-selected learning-rate schedule cannot drive --decoupled-lr parameter groups")
        step_count = len(payload["external_step_sizes"])
        interval = parameters.save_interval or step_count
        done = saved[-1].step if saved else 0
        runs = [(first, min(first + interval, step_count)) for first in range(done, step_count, interval)]
        metrics: dict[str, Any] = {**prepared.metrics, **rollout_metrics}
        for first, last in runs:
            if self.stopping(directory):
                break
            # Only the job's last rollout has trained all the data that the objective's next state is for.
            if last == step_count:
                objective, state = (parameters.objective if next_state else None), next_state
            else:
                objective, state = start.objective, dict(start.algorithm_state)
            # Optimizer steps advance only a recipe-selected schedule.
            next_schedule = None if schedule_state is None else schedule_state.advanced(last - first)
            training = (algorithm, slime_payload_steps(payload, first, last))
            rollout = SlimeRound(self.next_rollout_id, last, schedule_state, next_schedule, objective, state)
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
        return metrics

    def train_and_save(
        self,
        job: TrainingJob,
        directory: Path,
        rollout: SlimeRound,
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
            self.save_rollout(job, directory, rollout, saved_schedule, metrics, math.fsum(rewards) / len(rewards))
        return saved_schedule

    def save_rollout(
        self,
        job: TrainingJob,
        directory: Path,
        rollout: SlimeRound,
        schedule_state: LearningRateScheduleState | None,
        metrics: Mapping[str, Any],
        reward: float,
    ) -> None:
        """Save the trained ``rollout``'s checkpoint pair, with ``reward`` for retention, and record it."""
        self.group.save_model(rollout.rollout_id, force_sync=True)
        if self.critic_group is not None and critic_checkpoint_due(
            rollout.rollout_id, self.storage.critic_save_interval
        ):
            # The critic's weights and optimizer persist beside the actor's, so the value head survives a restart.
            # No Hugging Face export: the critic never serves.
            self.critic_group.save_model(rollout.rollout_id, force_sync=True)
        self.storage.complete(job.job_id, rollout.rollout_id, reward=reward)
        self.record_rollout(directory, rollout, schedule_state, metrics)

    def run_dataset(
        self, job: TrainingJob, directory: Path, settings: DatasetSettings, save_interval: int | None
    ) -> Mapping[str, Any]:
        """Train a dataset job with Slime's own loop: ``epochs`` passes over the dataset, in rollouts that Slime generates."""
        if self.rollout_group is None:
            raise RuntimeError("a dataset job rolls out on Slime's serving session, which this runner lacks")
        done = self.resume_slime_loop(job, directory)
        write_json(
            self.root / DATASET_FILE,
            {"job_id": job.job_id, "path": str(settings.path), "skip_rollouts": done, "token": uuid.uuid4().hex},
        )
        total = self.rollout_group.rollouts_per_epoch() * settings.epochs
        if total == 0:
            raise ValueError(f"the dataset {settings.path} holds fewer prompts than one rollout batch")
        return {"dataset_rollouts": total, **self.run_slime_loop(job, directory, done, total, save_interval)}

    def run_task_job(
        self, job: TrainingJob, directory: Path, settings: HarborSettings, save_interval: int | None
    ) -> Mapping[str, Any]:
        """Train a task job with Slime's own loop, whose generate step runs one Harbor rollout of each sample's task.

        The job trains every task at least once: the last rollout's batch wraps to the first tasks when the tasks do
        not fill it, as Slime's data source wraps to its next epoch.
        """
        done = self.resume_slime_loop(job, directory)
        tasks = tuple(
            (task_name(item), str(item.task_path.resolve()))
            for item in read_job_data(job.data_dir, job.data_id).samples
            if isinstance(item, TaskSample)
        )
        source = TaskJobSource(
            job.job_id, uuid.uuid4().hex, tasks, settings, str(directory.resolve()), self.train_timeout_s, done
        )
        write_json(self.root / DATASET_FILE, source.to_record())
        total = math.ceil(len(tasks) / self.args.rollout_batch_size)
        metrics = self.run_slime_loop(job, directory, done, total, save_interval, source)
        names = [name for name, _ in tasks]
        return {"task_rollouts": total, **metrics, **task_job_metrics(names, read_task_rollout_results(directory))}

    def resume_slime_loop(self, job: TrainingJob, directory: Path) -> int:
        """The job's rollouts of Slime's own loop that trained before this driver.

        ``ValueError``: the workers do not hold the job's start or its last checkpoint. A rollout that Megatron saved
        before a restart stopped the job is recorded here.
        """
        saved = self.saved(directory)
        started = read_json(directory / ROLLOUT_FILE)
        recorded = saved[-1].step if saved else 0
        if started is not None and started["rollout_id"] == self.loaded_rollout_id and started["step"] > recorded:
            # A restart stopped the job after Megatron saved this rollout, which the workers loaded again.
            rollout = SlimeRound(started["rollout_id"], started["step"], None, None, None, {})
            self.storage.complete(job.job_id, rollout.rollout_id, reward=0.0)
            self.record_rollout(directory, rollout, None, {"resumed": 1})
            return started["step"]
        held = SlimeStart.read(Path(saved[-1].path if saved else job.start))
        self.check_held_weights(held.rollout_id, job.start)
        return recorded

    def run_slime_loop(
        self,
        job: TrainingJob,
        directory: Path,
        done: int,
        total: int,
        save_interval: int | None,
        tasks: TaskJobSource | None = None,
    ) -> dict[str, Any]:
        """Train the job's rollouts ``done`` to ``total`` of Slime's own loop: rollouts that Slime generates and trains.

        Each rollout runs Slime's sequence: the engines take the workers' weights, generate the rollout and release
        their memory, and the workers train it. Every ``save_interval`` rollouts (None: only the last) and the last
        save a checkpoint, whose step counts the job's rollouts. With ``tasks``, a rollout whose prompt groups cannot
        train fails the job (``check_rollout_can_train``).
        """
        if self.rollout_group is None:
            raise RuntimeError("Slime's own loop rolls out on the serving session, which this runner lacks")
        interval = save_interval or total
        metrics: dict[str, Any] = {}
        results_read = 0
        for index in range(done, total):
            if self.stopping(directory):
                break
            rollout_id = self.next_rollout_id
            self.rollout_group.sync(self.group)
            data = self.rollout_group.generate(rollout_id)
            self.rollout_group.release()
            if tasks is not None:
                results = [r for r in read_task_rollout_results(directory) if r.token == tasks.token]
                check_rollout_can_train(results[results_read:], group_size=2 if has_group_baseline(self.args) else 1)
                results_read = len(results)
            self.resolve(self.group.async_train(rollout_id, data))
            rollout_metrics = merge_rank_metrics(self.resolve(self.group.async_pop_metrics()))
            metrics.update(rollout_metrics)
            step = index + 1
            if step % interval and step != total:
                self.next_rollout_id = rollout_id + 1
                continue
            rollout = SlimeRound(rollout_id, step, None, None, None, {})
            reward = rollout_metrics.get("rollout/raw_reward", rollout_metrics.get("rollout/rewards", 0.0))
            protected = set() if self.loaded_rollout_id is None else {self.loaded_rollout_id}
            with self.storage.admit(rollout_id=rollout_id, active_rollouts=protected) as storage_plan:
                if storage_plan["blocked"]:
                    raise RuntimeError(f"checkpoint storage is full: {'; '.join(storage_plan['reasons'])}")
                write_json(directory / ROLLOUT_FILE, {"rollout_id": rollout_id, "step": step})
                self.save_rollout(job, directory, rollout, None, dict(rollout_metrics), float(reward))
            metrics["rollout_id"] = rollout_id
        return metrics

    def record_rollout(
        self,
        directory: Path,
        rollout: SlimeRound,
        schedule_state: LearningRateScheduleState | None,
        metrics: Mapping[str, Any],
    ) -> None:
        """Record the saved ``rollout`` as the job's checkpoint at ``rollout.step`` and as the next start.

        The start records ``schedule_state`` and the rollout's objective state.
        """
        hf = Path(self.args.save_hf.format(rollout_id=rollout.rollout_id))
        if hf.is_symlink() or not hf.is_dir():
            raise RuntimeError(f"checkpoint is missing or unsafe: {hf}")
        self.next_rollout_id = max(self.next_rollout_id, rollout.rollout_id + 1)
        start_dir, artifact_dir = write_rollout_start(self.starts_root, hf, rollout, schedule_state, lora=self.lora)
        self.loaded_rollout_id = rollout.rollout_id
        self.save_checkpoint(directory, SavedCheckpoint(rollout.step, str(start_dir), str(artifact_dir), metrics))

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
    options. ``loss_family`` is the startup loss family, by name or
    dotted reference. ``recipe`` names the recipe in error messages.
    ``execution`` holds the deployment's ``execution`` and ``executors``
    sections, which select the training workers' executor. ``ray_address``
    and ``namespace`` name the Ray cluster the workers run on.
    ``train_timeout_s`` bounds one worker call, in seconds. The strings are
    non-empty and the timeout is positive. ``SlimeComponentFactory`` makes the
    value; ``SlimeTrainingComponent`` reads it and hands it to each job's driver, whose
    ``start_slime_runner`` reads it.
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


class SlimeTrainingComponent(SlimeJobs):
    """The Slime component that Reef calls: it runs each job in a driver process of its own.

    ``settings`` are the deployment's Slime settings. Building the component
    parses the Slime arguments once, without starting anything, so a
    malformed option fails at deployment, and checks the checkpoint storage.
    ``run`` writes the settings to the job's ``driver.json`` and starts
    ``python -m reef.train.slime_backend.driver`` on the job's directory,
    with its output in ``driver.log``; the driver records the job's
    checkpoints there and its metrics in ``driver-result.json``. A cancel or
    ``close`` stops the driver (SIGTERM, then SIGKILL after
    ``DRIVER_STOP_TIMEOUT_S``); a closed job resumes in a new driver after a
    restart. The next job continues the latest rollout that a driver saved.
    """

    def __init__(self, settings: SlimeSettings) -> None:
        self.settings = settings
        args, _, retention = prepare_slime_arguments(
            (*settings.arguments, NO_ROLLOUT_ENGINES),
            loss_family=settings.loss_family,
            recipe=settings.recipe,
            execution=settings.execution,
        )
        prepare_checkpoint_storage(args, retention)
        # Imported here: the LoRA module loads Megatron's LoRA stack.
        from reef.train.slime_backend.reef_adapters.megatron.lora import megatron_lora_enabled

        super().__init__(args, lora=megatron_lora_enabled(args))

    @property
    def held_rollout_id(self) -> int | None:
        """The latest rollout whose start a driver recorded, or ``None`` before the first."""
        rollouts = [
            int(path.name.removeprefix("rollout_"))
            for path in self.starts_root.glob("rollout_*")
            if (path / START_STATE).is_file()
        ]
        return max(rollouts, default=None)

    def driver_command(self, directory: Path) -> list[str]:
        """The command that runs the driver of the job in ``directory``."""
        python = os.environ.get("REEF_PYTHON", sys.executable)
        return [python, "-m", "reef.train.slime_backend.driver", str(directory)]

    def dispose_command(self) -> list[str]:
        """The command that disposes the component's Slime serving session, in a Ray session of its own."""
        python = os.environ.get("REEF_PYTHON", sys.executable)
        settings = self.settings
        return [
            python,
            "-m",
            "reef.train.slime_backend.driver",
            "--dispose",
            str(self.root),
            settings.ray_address,
            settings.namespace,
        ]

    def close_resources(self) -> None:
        """Dispose the serving session that a task job's driver left: its engines, router and GPUs."""
        if not (self.root / SERVING_FILE).exists():
            return
        with open(self.root / SERVING_LOG, "ab") as log:
            subprocess.run(self.dispose_command(), cwd=self.root, stdout=log, stderr=subprocess.STDOUT, check=True)

    def run(self, job: TrainingJob, directory: Path) -> Mapping[str, Any]:
        write_json(directory / DRIVER_FILE, settings_to_json(self.settings))
        (directory / DRIVER_RESULT_FILE).unlink(missing_ok=True)
        with open(directory / DRIVER_LOG, "ab") as log:
            # The job's directory as the working directory: ``python -m`` puts it first on the import path, where
            # no other ``reef`` package can shadow the installed one.
            driver = subprocess.Popen(
                self.driver_command(directory),
                cwd=directory,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        try:
            while driver.poll() is None:
                if self.stopping(directory):
                    # A cancelled job already has its result; a closed one resumes after a restart.
                    return {}
                time.sleep(1)
        finally:
            stop_driver(driver)
        if driver.returncode != 0:
            tail = (directory / DRIVER_LOG).read_text(errors="replace")[-2000:]
            raise RuntimeError(f"the Slime driver exited with {driver.returncode}; the end of {DRIVER_LOG}:\n{tail}")
        result = read_json(directory / DRIVER_RESULT_FILE)
        if result is None:
            raise RuntimeError(f"the Slime driver exited without {DRIVER_RESULT_FILE}")
        return result["metrics"]


def stop_driver(driver: subprocess.Popen[bytes]) -> None:
    """Stop a driver that still runs: SIGTERM to its process group, then SIGKILL after ``DRIVER_STOP_TIMEOUT_S``."""
    if driver.poll() is not None:
        return
    os.killpg(driver.pid, signal.SIGTERM)
    try:
        driver.wait(timeout=DRIVER_STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        os.killpg(driver.pid, signal.SIGKILL)
        driver.wait()


def settings_to_json(settings: SlimeSettings) -> dict[str, Any]:
    """``settings`` as a JSON object, which ``settings_from_json`` reads back."""
    value = asdict(settings)
    return {**value, "arguments": list(settings.arguments), "execution": dict(settings.execution)}


def settings_from_json(value: Mapping[str, Any]) -> SlimeSettings:
    return SlimeSettings(**{**value, "arguments": tuple(value["arguments"])})


def start_slime_runner(settings: SlimeSettings, *, serving: bool = False) -> SlimeJobRunner:
    """Prepare the arguments, place the job, start its workers (and engines) and return the runner.

    With ``serving``, and arguments that give the rollout engines GPUs, the driver creates or attaches to the
    component's Slime serving session (``SlimeRolloutGroup``), which holds the GPUs and the engines across jobs,
    and trains on its placements; otherwise it places the job with Slime's own placement and starts no engines.
    The engines start before the workers, as in Slime's own training loop: colocated engines take their GPU
    memory and release it before the workers load the model on the same GPUs.
    """
    args, loss_family_config, retention = prepare_slime_arguments(
        settings.arguments, loss_family=settings.loss_family, recipe=settings.recipe, execution=settings.execution
    )
    storage = prepare_checkpoint_storage(args, retention)
    with_engines = serving and bool(args.rollout_num_gpus)
    runtime_env = job_runtime_env()
    if with_engines:
        runtime_env["env_vars"][SGLANG_PLUGIN_ENV] = "1"
    resources = SlimeDeploymentResources(
        args,
        ray_address=settings.ray_address,
        namespace=settings.namespace,
        runtime_env=runtime_env,
        place=not with_engines,
    )
    resources.start()
    rollout_group: SlimeRolloutGroup | None = None
    try:
        # Imported here: the worker class loads torch, Megatron and Slime's Ray actor.
        from reef.train.slime_backend.reef_adapters.megatron.lora import megatron_lora_enabled
        from reef.train.slime_backend.reef_adapters.megatron.train_actor import ReefMegatronTrainRayActor

        if with_engines:
            rollout_group = SlimeRolloutGroup.start(args, serving_record(args, SlimeJobs.root_of(args)))
            resources.adopt(rollout_group.placements)
        actor_group, critic_group = create_train_groups(
            args,
            resources.placement_groups,
            actor_cls=ReefMegatronTrainRayActor,
            train_timeout_s=settings.train_timeout_s,
        )
        if rollout_group is not None:
            actor_group.set_rollout_manager(rollout_group.manager)
        return SlimeJobRunner(
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
            rollout_group=rollout_group,
        )
    except BaseException:
        if rollout_group is not None:
            rollout_group.close()
        resources.close()
        raise


def serving_record(args: SlimeArguments, root: Path) -> str:
    """The process lease of the serving session of ``args``: the recorded one while the session runs, else a new one.

    A new lease is recorded in ``root``'s ``SERVING_FILE`` before the session starts, so the component disposes a
    session whose driver failed while starting it.
    """
    session = serving_session(args)
    record = read_json(root / SERVING_FILE)
    if record is not None and record["session"] == session and serving_exists(session):
        return record["lease"]
    lease = uuid.uuid4().hex
    write_json(root / SERVING_FILE, {"session": session, "lease": lease})
    return lease


def dispose_component_serving(root: Path, *, ray_address: str, namespace: str, timeout_s: float) -> None:
    """Dispose the serving session that ``root``'s ``SERVING_FILE`` records, and retire its processes."""
    import ray

    record = read_json(root / SERVING_FILE)
    if record is None:
        return
    ray.init(address=ray_address, namespace=namespace)
    try:
        dispose_serving(record["session"], timeout_s)
        retire_lease(record["lease"], [])
    finally:
        ray.shutdown()
    (root / SERVING_FILE).unlink()


__all__ = ["SlimeJobRunner", "SlimeSettings", "SlimeStart", "SlimeTrainingComponent", "start_slime_runner"]
