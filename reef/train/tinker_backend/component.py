"""Tinker as a training component: each job trains its data remotely, from the job's start checkpoint.

A job starts from ``job.start``, a directory that holds a Tinker checkpoint
manifest, or from the base checkpoint when ``start`` is ``None``. The manifest
carries the remote training state, the learning-rate schedule's progress and
the objective's state, so a job that starts from it continues all three. Each
attempt branches from those weights and their optimizer state, so repeating an
attempt can consume API resources but cannot apply its gradient twice.

Serving reads each saved checkpoint from its artifact: Tinker's sampler
(``reef.inference.tinker``) reads the manifest, and Reef's serving engines
(``reef.runtime.serving``) load the PEFT adapter the component exports beside
it. Nothing is shared in process.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from reef.core.evaluation import RolloutScores
from reef.core.job_data import JobData, TaskSample, trajectories
from reef.runtime.durable_files import read_json, write_json
from reef.runtime.interfaces import TrainingMethod, learning_rate_metrics, resolve_learning_rate_schedule_state
from reef.runtime.jobs import ReleaseFormat, SavedCheckpoint, TrainingJob
from reef.runtime.serving import ADAPTER_DIR, holds_weights
from reef.tinker.checkpoint import MANIFEST, TinkerCheckpoint
from reef.tinker.sampling import TinkerSampler
from reef.train.algos.registry import job_training
from reef.train.algos.schedule import StepScheduling
from reef.train.local_jobs import LocalJobComponent
from reef.train.tinker_backend.client import TINKER_ADAPTER_DIR, TinkerClient
from reef.train.tinker_backend.config import TinkerConfig
from reef.train.tinker_backend.preparation import prepare_tinker_job
from reef.train.tinker_backend.rollout import RolloutSettings, TaskJobRollouts, roll_out, task_name

#: The rollouts of a task job, in its job directory.
ROLLOUTS_FILE = "rollouts.json"
#: The parameters of a Tinker job besides ``method`` and ``scheduling``.
JOB_PARAMETER_NAMES = frozenset({"batch_size", "kl_coef", "learning_rate", "save_interval", "rollout"})


@dataclass(frozen=True)
class TinkerJobParameters:
    """The parameters of one Tinker job, with the component's defaults for those the job leaves out.

    Values: ``method`` and ``scheduling`` as ``job_training`` reads them; ``batch_size`` a positive int, the
    rollouts of one optimizer step; ``learning_rate`` finite and positive; ``kl_coef`` finite and non-negative;
    ``save_interval`` a positive int, the optimizer steps between two saved checkpoints, or None for one
    checkpoint when the job ends; ``rollout`` the rollouts of a task job, or None for a job of trajectories.
    Purpose: ``TinkerTrainingComponent.job_parameters`` makes it when it checks and when it runs a job; the run
    reads it.
    """

    method: TrainingMethod
    scheduling: StepScheduling
    batch_size: int
    learning_rate: float
    kl_coef: float
    save_interval: int | None
    rollout: RolloutSettings | None

    def __post_init__(self) -> None:
        counts = {
            "batch_size": self.batch_size,
            "save_interval": 1 if self.save_interval is None else self.save_interval,
        }
        for name, count in counts.items():
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError(f"{name} must be a positive int")
        for name, rate in (("learning_rate", self.learning_rate), ("kl_coef", self.kl_coef)):
            if isinstance(rate, bool) or not isinstance(rate, int | float) or not math.isfinite(rate):
                raise ValueError(f"{name} must be a finite number")
        if self.learning_rate <= 0 or self.kl_coef < 0:
            raise ValueError("learning_rate must be positive and kl_coef non-negative")


class TinkerTrainingComponent(LocalJobComponent):
    """Train each job as remote Tinker sessions from the job's start checkpoint, in a background thread.

    A job's parameters are ``TinkerJobParameters``: ``method``, optionally
    ``scheduling``, and ``batch_size``, ``kl_coef``, ``learning_rate``,
    ``save_interval`` and ``rollout``, where ``batch_size``, ``kl_coef`` and
    ``learning_rate`` default to the component's ``TinkerConfig``. ``submit``
    refuses any other parameter or a malformed value. Jobs of several
    scenarios can share the component, since each job trains from its own
    start.

    A job's data is trajectories, which it trains as they are, or Harbor
    tasks (``TaskSample``), a task job. A task job needs the ``rollout``
    parameter (``RolloutSettings``) and a ``sampler``: before it trains, the
    component runs the tasks' rollouts on the sampler of the job's start
    (``reef.train.tinker_backend.rollout``) and trains the samples of their
    groups. The tasks of a job have distinct names (``task_name``), each of
    which names its group. The component keeps the rollouts in
    ``rollouts.json`` in the job's directory, so a resumed job trains the same
    samples, and reports in the job's metrics each task's rewards of the
    rollouts that ran (``rollout_rewards``, by task name).

    A job's last checkpoint carries the objective's state after the job's
    data; an intermediate checkpoint (``save_interval``) carries the state of
    the job's start, since the objective computes its state for the whole data.

    ``config.state_dir`` holds ``base/`` (the base checkpoint),
    ``imported/<digest>/`` (starts taken from releases) and ``jobs/`` (see
    ``LocalJobComponent``); a saved checkpoint is ``jobs/<job_id>/<step>/``,
    with its manifest and, with ``config.export_adapter``, its PEFT adapter in
    ``adapter/`` and Tinker's adapter as Tinker saved it in ``tinker-adapter/``
    (``TinkerClient.download``). A checkpoint directory is both the start of a later job and
    the release Reef publishes.
    """

    def __init__(
        self, base_model: str, config: TinkerConfig, client: TinkerClient, *, sampler: TinkerSampler | None = None
    ) -> None:
        self._model = base_model
        self._config = config
        self._client = client
        self._sampler = sampler
        self._root = Path(config.state_dir).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._state_lock = (self._root / ".lock").open("a")
        try:
            fcntl.flock(self._state_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._state_lock.close()
            raise ValueError("Tinker state_dir is already owned by another runtime") from None
        try:
            super().__init__(self._root / "jobs")
            base = self._root / "base"
            if (base / MANIFEST).exists():
                self._base = TinkerCheckpoint.read(base)
            else:
                self._base = client.initialize()
                self._base.write(base)
            self._base.check_compatible(base_model, service=config.base_url, lora_rank=config.lora_rank)
        except BaseException:
            self._state_lock.close()
            raise
        self.start_jobs()

    @property
    def name(self) -> str:
        return "tinker"

    @property
    def release_format(self) -> ReleaseFormat | None:
        """The exported PEFT adapter of the configured rank, or ``None`` when releases hold only the manifest."""
        if not self._config.export_adapter:
            return None
        return ReleaseFormat("adapter", self._config.lora_rank, ("all",))

    @property
    def state_dir(self) -> Path:
        return self._root

    def _validated(self, checkpoint: TinkerCheckpoint) -> TinkerCheckpoint:
        checkpoint.check_compatible(self._model, service=self._config.base_url, lora_rank=self._config.lora_rank)
        return checkpoint

    @property
    def shares_scenarios(self) -> bool:
        return True

    def import_start(self, release: Path) -> str | None:
        if not holds_weights(release):
            return None
        if not (release / MANIFEST).exists():
            raise ValueError(f"artifact is missing {MANIFEST}")
        checkpoint = self._validated(TinkerCheckpoint.read(release))
        digest = hashlib.sha256(json.dumps(checkpoint.to_dict(), sort_keys=True).encode()).hexdigest()
        imported = self._root / "imported" / digest
        if not imported.exists():
            # The manifest names the remote state; the exported adapter is not needed to train from it.
            checkpoint.write(imported)
        return str(imported)

    def job_parameters(self, parameters: Mapping[str, Any]) -> TinkerJobParameters:
        """``parameters`` over the component's defaults; ``ValueError`` for an unknown or a malformed parameter."""
        unknown = set(parameters) - {"method", "scheduling", *JOB_PARAMETER_NAMES}
        if unknown:
            raise ValueError(f"Tinker jobs accept no parameters {sorted(unknown)}")
        method, scheduling = job_training(parameters)
        return TinkerJobParameters(
            method,
            scheduling,
            parameters.get("batch_size", self._config.batch_size),
            parameters.get("learning_rate", self._config.learning_rate),
            parameters.get("kl_coef", self._config.kl_coef),
            parameters.get("save_interval"),
            None if parameters.get("rollout") is None else RolloutSettings.from_parameters(parameters),
        )

    def check_job(self, job: TrainingJob) -> None:
        parameters = self.job_parameters(job.parameters)
        if job.start is not None:
            if not (Path(job.start) / MANIFEST).is_file():
                raise ValueError(f"the job's start {job.start} holds no Tinker checkpoint manifest")
            self._validated(TinkerCheckpoint.read(Path(job.start)))
        tasks = [item for item in job.data.samples if isinstance(item, TaskSample)]
        if tasks and len(tasks) != len(job.data.samples):
            raise ValueError("a Tinker job trains either trajectories or tasks, not both")
        if tasks:
            self.check_task_jobs(job.parameters)
            names = [task_name(task) for task in tasks]
            if len(set(names)) != len(names):
                raise ValueError("the tasks of one job need distinct names: each name groups its task's samples")
        elif parameters.rollout is not None:
            raise ValueError("the rollout parameter applies only to a job of tasks")

    def check_task_jobs(self, parameters: Mapping[str, Any]) -> None:
        """A task job rolls its tasks out on this component's sampler, as the ``rollout`` parameter says."""
        if self._sampler is None:
            raise ValueError("this Tinker component has no sampler, so it cannot roll tasks out")
        if parameters.get("rollout") is None:
            raise ValueError("a task job needs the rollout parameter as an object")
        RolloutSettings.from_parameters(parameters)

    def close_resources(self) -> None:
        try:
            self._client.close()
            if self._sampler is not None:
                self._sampler.close()
        finally:
            self._state_lock.close()

    def task_rollouts(
        self, job: TrainingJob, directory: Path, start: TinkerCheckpoint, settings: RolloutSettings
    ) -> TaskJobRollouts:
        """The rollouts of a task job: those of an earlier attempt, else new ones on the start's sampler."""
        record = directory / ROLLOUTS_FILE
        kept = read_json(record)
        if kept is not None:
            samples = JobData.from_dict(kept["samples"]).samples
            return TaskJobRollouts(
                trajectories(JobData(job.data.data_id, samples)), RolloutScores.from_json(kept["scores"])
            )
        if self._sampler is None:
            raise RuntimeError("this Tinker component has no sampler, so it cannot roll tasks out")
        tasks = tuple(item for item in job.data.samples if isinstance(item, TaskSample))
        rollouts = roll_out(
            tasks,
            settings,
            self._sampler,
            start.sampler_path,
            model=self._model,
            runtime_load_id=f"tinker:{start.sampler_path}",
            work_dir=directory / "rollouts",
        )
        write_json(
            record,
            {
                "samples": JobData(job.data.data_id, rollouts.samples).to_dict(),
                "scores": rollouts.scores.to_json(),
            },
        )
        return rollouts

    def run(self, job: TrainingJob, directory: Path) -> Mapping[str, Any]:
        parameters = self.job_parameters(job.parameters)
        start = self._base if job.start is None else self._validated(TinkerCheckpoint.read(Path(job.start)))
        method = parameters.method
        data = job.data
        rollout_metrics: dict[str, Any] = {}
        if parameters.rollout is not None:
            rollouts = self.task_rollouts(job, directory, start, parameters.rollout)
            rollout_metrics = {
                "rollout_rewards": {name: list(values) for name, values in rollouts.rewards().items()},
                "rollout_samples": len(rollouts.samples),
            }
            if not rollouts.samples:
                # No task has a group of two samples to compare, so the job trains nothing.
                return rollout_metrics
            data = JobData(data.data_id, rollouts.samples)
        prepared = prepare_tinker_job(
            data,
            method,
            start.state_for(method.objective),
            parameters.scheduling,
            batch_size=parameters.batch_size,
            kl_coef=parameters.kl_coef,
        )
        if prepared.loss is None:
            # The objective skipped the job's data: the job trains nothing and saves no checkpoint.
            return {**prepared.metrics, **rollout_metrics}
        steps = prepared.steps
        schedule = resolve_learning_rate_schedule_state(
            start.learning_rate_schedule_state, method.learning_rate_schedule
        )
        learning_rates = (
            (parameters.learning_rate,) * len(steps) if schedule is None else schedule.learning_rates(len(steps))
        )
        interval = parameters.save_interval or len(steps)
        saved = self.saved(directory)
        # A resumed job continues after its last saved checkpoint, from that checkpoint.
        done = saved[-1].step if saved else 0
        current = TinkerCheckpoint.read(Path(saved[-1].path)) if saved else start
        for first in range(done, len(steps), interval):
            if self.stopping(directory):
                break
            last = min(first + interval, len(steps))
            checkpoint, metrics = self._client.train(
                current,
                steps[first:last],
                prepared.loss,
                learning_rates[first:last],
                kl_coef=parameters.kl_coef,
                export_adapter=self._config.export_adapter,
            )
            schedule_state = None if schedule is None else schedule.advanced(last)
            # Only the job's last checkpoint has trained all the data that the objective's next state is for.
            final = last == len(steps)
            current = replace(
                self._validated(checkpoint),
                learning_rate_schedule_state=schedule_state,
                algorithm_state=dict(prepared.next_algorithm_state) if final else dict(start.algorithm_state),
                objective=(method.objective if prepared.next_algorithm_state else None) if final else start.objective,
            )
            target = directory / str(last)
            if self._config.export_adapter:
                # An earlier attempt may have left part of the adapter.
                for adapter in (ADAPTER_DIR, TINKER_ADAPTER_DIR):
                    shutil.rmtree(target / adapter, ignore_errors=True)
                self._client.download(current, target)
            current.write(target)
            self.save_checkpoint(
                directory,
                SavedCheckpoint(
                    last,
                    str(target),
                    str(target),
                    {
                        **prepared.metrics,
                        **metrics,
                        "checkpoint_optimizer_steps": last - first,
                        **learning_rate_metrics(learning_rates[first:last], schedule_state),
                    },
                ),
            )
        return {**prepared.metrics, **rollout_metrics}
