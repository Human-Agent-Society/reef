"""Tinker as a training component: each job trains its data remotely, from the job's start checkpoint.

``TinkerJobComponent`` is the job loop that every component on a Tinker-compatible service shares;
``TinkerTrainingComponent`` trains a LoRA on it, and ``reef.train.tinker_backend.full`` all the parameters on
Spindle. A job starts from ``job.start``, a directory that holds a Tinker checkpoint
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
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from reef.artifact.files import ArtifactFiles
from reef.artifact.repository import BaseArtifactSource
from reef.core.evaluation import RolloutScores
from reef.core.job_data import JobData, TaskSample, trajectories
from reef.runtime.durable_files import read_json, write_json
from reef.runtime.interfaces import TrainingMethod, learning_rate_metrics, resolve_learning_rate_schedule_state
from reef.runtime.jobs import ReleaseFormat, SavedCheckpoint, TrainingJob
from reef.tinker.checkpoint import MANIFEST, FreshStart, TinkerCheckpoint
from reef.tinker.sampling import TinkerSampler
from reef.train.algos.registry import job_training
from reef.train.algos.schedule import StepScheduling
from reef.train.local_jobs import LocalJobComponent
from reef.train.tinker_backend.client import TinkerClient
from reef.train.tinker_backend.config import TinkerConfig, TinkerServiceConfig
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


class TinkerJobComponent(LocalJobComponent, ABC):
    """Train each job as remote sessions on a Tinker-compatible service from the job's start, in a background thread.

    A job's parameters are ``TinkerJobParameters``: ``method``, optionally
    ``scheduling``, and ``batch_size``, ``kl_coef``, ``learning_rate``,
    ``save_interval`` and ``rollout``, where ``batch_size``, ``kl_coef`` and
    ``learning_rate`` default to the component's ``TinkerServiceConfig``.
    ``submit`` refuses any other parameter or a malformed value. Jobs of
    several scenarios can share the component, since each job trains from its
    own start.

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

    ``config.state_dir`` holds ``base/`` (the base start, which
    ``prepare_base`` writes), ``imported/<digest>/`` (starts taken from
    releases) and ``jobs/`` (see ``LocalJobComponent``); a saved checkpoint is
    ``jobs/<job_id>/<step>/``, which holds its manifest alone. A checkpoint
    directory is both the start of a later job and the release Reef publishes.
    The weights stay on the service. A subclass decides the parameterization:
    which checkpoints it trains (``validated``), what its base is
    (``prepare_base``, ``start_at``), and which releases it imports.
    """

    def __init__(
        self,
        base_model: str,
        config: TinkerServiceConfig,
        client: TinkerClient,
        *,
        sampler: TinkerSampler | None = None,
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
            self._base = self._root / "base"
            self.prepare_base()
            self.start_at(self._base)
        except BaseException:
            self._state_lock.close()
            raise
        self.start_jobs()

    @abstractmethod
    def prepare_base(self) -> None:
        """Write the base start into ``base/`` when it is not there; the constructor then checks it with ``start_at``."""

    @abstractmethod
    def validated(self, checkpoint: TinkerCheckpoint) -> TinkerCheckpoint:
        """``checkpoint`` when this component trains it: of its model, service and parameterization; else ``ValueError``."""

    def start_at(self, directory: Path) -> TinkerCheckpoint | FreshStart:
        """The start that ``directory`` holds: a checkpoint manifest that ``validated`` takes; else ``ValueError``."""
        if not (directory / MANIFEST).is_file():
            raise ValueError(f"the start {directory} holds no Tinker checkpoint manifest")
        return self.validated(TinkerCheckpoint.read(directory))

    def complete_release(self, checkpoint: TinkerCheckpoint, directory: Path) -> None:
        """Add to ``directory``, which holds ``checkpoint``'s manifest, what else its release holds; by default nothing.

        The job loop calls it for each saved checkpoint before the checkpoint is recorded, so a job that a restart
        interrupts here saves the checkpoint again.
        """
        return

    def check_manifest(self, files: ArtifactFiles) -> None:
        """Raise ``ValueError`` unless ``files`` hold a manifest that ``validated`` takes and whose state resolves."""
        if MANIFEST not in files.paths():
            raise ValueError(f"a Tinker release holds {MANIFEST}")
        try:
            checkpoint = TinkerCheckpoint.from_dict(json.loads(files.read_bytes(MANIFEST)))
        except (TypeError, ValueError) as exc:
            # A manifest from outside Reef may have any shape; ``from_dict`` refuses unknown fields with TypeError.
            raise ValueError(f"{MANIFEST} is no Tinker checkpoint manifest: {exc}") from exc
        self._client.check_remote(self.validated(checkpoint))

    def import_manifest(self, release: Path) -> str:
        """Take the manifest in ``release`` as a start: copy it into ``imported/<digest>/``; return that directory."""
        if not (release / MANIFEST).exists():
            raise ValueError(f"artifact is missing {MANIFEST}")
        checkpoint = self.validated(TinkerCheckpoint.read(release))
        digest = hashlib.sha256(json.dumps(checkpoint.to_dict(), sort_keys=True).encode()).hexdigest()
        imported = self._root / "imported" / digest
        if not imported.exists():
            # The manifest names the remote state; the exported adapter is not needed to train from it.
            checkpoint.write(imported)
        return str(imported)

    @property
    def release_format(self) -> ReleaseFormat | None:
        """``None`` by default: a release holds only its manifest, and the weights stay on the service."""
        return None

    @property
    def state_dir(self) -> Path:
        return self._root

    @property
    def shares_scenarios(self) -> bool:
        return True

    @property
    def base_start(self) -> str:
        return str(self._base)

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
            self.start_at(Path(job.start))
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
        self, job: TrainingJob, directory: Path, start: TinkerCheckpoint | FreshStart, settings: RolloutSettings
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
            # A fresh start names no sampler: its rollouts sample the base model.
            runtime_load_id=f"tinker:{start.sampler_path or f'base:{self._model}'}",
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
        start = self.start_at(Path(job.start))
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
        current: TinkerCheckpoint | FreshStart = TinkerCheckpoint.read(Path(saved[-1].path)) if saved else start
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
            )
            schedule_state = None if schedule is None else schedule.advanced(last)
            # Only the job's last checkpoint has trained all the data that the objective's next state is for.
            final = last == len(steps)
            current = replace(
                self.validated(checkpoint),
                learning_rate_schedule_state=schedule_state,
                algorithm_state=dict(prepared.next_algorithm_state) if final else dict(start.algorithm_state),
                objective=(method.objective if prepared.next_algorithm_state else None) if final else start.objective,
            )
            target = directory / str(last)
            current.write(target)
            self.complete_release(current, target)
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


class TinkerTrainingComponent(TinkerJobComponent):
    """Train a LoRA of rank ``config.lora_rank`` on the base model, on Tinker or a Tinker-compatible service.

    The base start, ``base/``, is the untrained adapter's checkpoint, which the
    client saves the first time (``TinkerClient.initialize``). Serving reads
    each saved checkpoint from its artifact: Tinker's sampler
    (``reef.inference.tinker``) reads the manifest, and a serving runtime that
    loads the weights into its own engines downloads the adapter from the
    manifest's ``sampler_path`` (``TinkerSamplerWeights``). An import is a
    manifest of a LoRA of the model, rank and service.
    """

    def __init__(
        self, base_model: str, config: TinkerConfig, client: TinkerClient, *, sampler: TinkerSampler | None = None
    ) -> None:
        self._lora_rank = config.lora_rank
        super().__init__(base_model, config, client, sampler=sampler)

    @property
    def name(self) -> str:
        return "tinker"

    def prepare_base(self) -> None:
        if not (self._base / MANIFEST).exists():
            # The untrained adapter is the base checkpoint: the base artifact holds it as a release.
            self._client.initialize().write(self._base)

    def validated(self, checkpoint: TinkerCheckpoint) -> TinkerCheckpoint:
        checkpoint.check_compatible(self._model, service=self._config.base_url)
        if checkpoint.lora_rank != self._lora_rank:
            raise ValueError(
                f"the Tinker checkpoint is no rank {self._lora_rank} LoRA of {self._model}: "
                f"its rank is {checkpoint.lora_rank}"
            )
        return checkpoint

    def base_artifact_source(self) -> BaseArtifactSource:
        """The untrained adapter's release: its manifest."""
        inputs = {"model": self._model, "lora_rank": str(self._lora_rank)}
        if self._config.base_url is not None:
            inputs["service"] = self._config.base_url
        return BaseArtifactSource(model_directory=self._base, inputs=inputs)

    def check_release(self, files: ArtifactFiles) -> None:
        """A manifest of a checkpoint of this component's model, rank and service, whose state path resolves there."""
        self.check_manifest(files)

    def import_start(self, release: Path) -> str:
        return self.import_manifest(release)
