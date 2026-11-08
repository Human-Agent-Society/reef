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
(``reef.inference.serving``) load the PEFT adapter the component exports beside
it. Nothing is shared in process.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TextIO

from reef.artifact.files import ArtifactFiles
from reef.artifact.repository import BaseModelFiles, DefaultStart
from reef.core.chat_rendering import needs_rendering, rendered_job_data
from reef.core.evaluation import RolloutScores
from reef.core.job_data import JobData, TaskSample, trajectories
from reef.core.weight_layout import ArtifactFormat
from reef.storage.durable_files import read_json, write_json
from reef.storage.job_data import JobDataKind, read_job_data
from reef.tinker.checkpoint import MANIFEST
from reef.tinker.sampling import TinkerSampler
from reef.train.component import SavedCheckpoint, TrainingJob
from reef.train.learning_rate import learning_rate_metrics, resolve_learning_rate_schedule_state
from reef.train.local_jobs import LocalJobComponent
from reef.train.task_rollouts import RolloutSettings, TaskJobRollouts, run_train_tasks, task_name
from reef.train.tinker_backend.checkpoint import FreshStart, TinkerCheckpoint
from reef.train.tinker_backend.client import TinkerClient
from reef.train.tinker_backend.config import ScoreCentering, TinkerConfig, TinkerServiceConfig
from reef.train.tinker_backend.parameters import TinkerJobParameters, rollout_settings, tinker_parameters_schema
from reef.train.tinker_backend.preparation import prepare_tinker_job
from reef.train.tinker_backend.rollout import SamplerChat

#: Seconds that the start's sampler may take to answer its first sample before a task job's rollouts start.
SAMPLER_WARM_TIMEOUT_S = 3600.0
#: The rollouts of a task job, in its job directory.
ROLLOUTS_FILE = "rollouts.json"


class TinkerJobComponent(LocalJobComponent, ABC):
    """Train each job as remote sessions on a Tinker-compatible service from the job's start, in a background thread.

    A job's parameters are ``TinkerJobParameters``: ``objective``, optionally
    ``learning_rate_schedule`` and ``scheduling``, and ``batch_size``, ``kl_coef``,
    ``learning_rate``, ``save_interval`` and ``rollout``, where ``batch_size``,
    ``kl_coef`` and ``learning_rate`` default to the component's ``TinkerServiceConfig``.
    ``submit`` refuses any other parameter or a malformed value. Jobs of
    several scenarios can share the component, since each job trains from its
    own start.

    A job's data is trajectories, which it trains as they are, or Harbor
    tasks (``TaskSample``), a task job. A task job needs the ``rollout``
    parameter (``RolloutSettings``) and a ``sampler``: before it trains, the
    component runs the tasks' rollouts on the sampler of the job's start
    (``reef.train.task_rollouts``) and trains the samples of their
    groups. The tasks of a job have distinct names (``task_name``), each of
    which names its group. The component keeps the rollouts in
    ``rollouts.json`` in the job's directory, so a resumed job trains the same
    samples, and reports in the job's metrics each task's rewards of the
    rollouts that ran (``rollout_rewards``, by task name).

    With a ``score_centering``, a policy-gradient job trains as score-centered
    ``cispo`` (``prepare_tinker_job``), and a task job's rollouts record the
    sampler's top-k at each sampled token, so they sample at temperature 1
    without truncation (``sample_chat``).

    A job's last checkpoint carries the objective's state after the job's
    data; an intermediate checkpoint (``save_interval``) carries the state of
    the job's start, since the objective computes its state for the whole data.

    Every process that uses the component reads and writes ``config.state_dir``; only the process that runs the
    jobs holds its lock (``start_jobs``). A split deployment gives control and the training service one directory.
    ``config.state_dir`` holds ``base/`` (the base start), ``imported/<digest>/`` (starts taken from
    releases) and ``jobs/`` (see ``LocalJobComponent``). The constructor makes no base start. ``prepared_base``
    makes ``base/`` on its first call in a process: before the process runs jobs (``start_jobs``), when it reads
    ``base_start`` or checks a job from it, or when a new scenario's root snapshot holds the default start
    (``TinkerBaseFiles``). A process that does none of these, such as a service that only opens scenarios, makes
    nothing. A saved checkpoint is
    ``jobs/<job_id>/<step>/``, which holds its manifest alone. A checkpoint
    directory is both the start of a later job and the snapshot Reef publishes.
    The weights stay on the service. A subclass decides the parameterization:
    which checkpoints it trains (``validated``), what its base is
    (``prepare_base``, ``start_at``), and which snapshots it imports.
    """

    def __init__(
        self,
        base_model: str,
        config: TinkerServiceConfig,
        client: TinkerClient,
        *,
        sampler: TinkerSampler | None = None,
        score_centering: ScoreCentering | None = None,
    ) -> None:
        self._model = base_model
        #: The score centering of this component's policy-gradient jobs, or None for none.
        self.score_centering = score_centering
        self._config = config
        self._client = client
        self._sampler = sampler
        self._root = Path(config.state_dir).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        #: The lock of ``state_dir``, which the process that runs the jobs holds (``start_jobs``); None until then.
        self._state_lock: TextIO | None = None
        super().__init__(self._root / "jobs")
        self._base = self._root / "base"
        #: Whether this process prepared ``base/`` (``prepared_base``); ``_base_lock`` guards it.
        self._base_prepared = False
        self._base_lock = threading.Lock()

    def start_jobs(self) -> None:
        """Take ``state_dir`` for this process, prepare the base start, then run the jobs: one process at a time runs a
        state directory's jobs. Raise ``ValueError`` when another process holds it; a second call changes nothing.
        When the base start fails, the process releases ``state_dir`` and raises the failure."""
        if self._state_lock is not None:
            return
        lock = (self._root / ".lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            lock.close()
            raise ValueError("Tinker state_dir is already owned by another runtime") from None
        try:
            self.prepared_base()
        except BaseException:
            lock.close()
            raise
        self._state_lock = lock
        super().start_jobs()

    def prepared_base(self) -> Path:
        """``base/``, after this process prepared it.

        The first call in a process writes the base start (``prepare_base``) and checks it (``start_at``). A later
        call returns at once. A failed call prepares nothing, so the next call tries again.
        """
        with self._base_lock:
            if not self._base_prepared:
                self.prepare_base()
                self.start_at(self._base)
                self._base_prepared = True
        return self._base

    @abstractmethod
    def prepare_base(self) -> None:
        """Write the base start into ``base/`` when it is not there, or when it is a function of the configuration and
        ``base/`` holds the start of another configuration; ``prepared_base`` then checks it with ``start_at``."""

    @abstractmethod
    def validated(self, checkpoint: TinkerCheckpoint) -> TinkerCheckpoint:
        """``checkpoint`` when this component trains it: of its model, service and parameterization; else ``ValueError``."""

    def start_at(self, directory: Path) -> TinkerCheckpoint | FreshStart:
        """The start that ``directory`` holds: a checkpoint manifest that ``validated`` takes; else ``ValueError``."""
        if not (directory / MANIFEST).is_file():
            raise ValueError(f"the start {directory} holds no Tinker checkpoint manifest")
        return self.validated(TinkerCheckpoint.read(directory))

    def complete_artifact(self, checkpoint: TinkerCheckpoint, directory: Path) -> None:
        """Add to ``directory``, which holds ``checkpoint``'s manifest, what else its snapshot holds; by default nothing.

        The job loop calls it for each saved checkpoint before the checkpoint is recorded, so a job that a restart
        interrupts here saves the checkpoint again.
        """
        return

    def check_manifest(self, files: ArtifactFiles) -> None:
        """Raise ``ValueError`` unless ``files`` hold a manifest that ``validated`` takes and whose state resolves."""
        if MANIFEST not in files.paths():
            raise ValueError(f"a Tinker artifact holds {MANIFEST}")
        try:
            checkpoint = TinkerCheckpoint.from_dict(json.loads(files.read_bytes(MANIFEST)))
        except (TypeError, ValueError) as exc:
            # A manifest from outside Reef may have any shape; ``from_dict`` refuses unknown fields with TypeError.
            raise ValueError(f"{MANIFEST} is no Tinker checkpoint manifest: {exc}") from exc
        self._client.check_remote(self.validated(checkpoint))

    def import_manifest(self, artifact_dir: Path) -> str:
        """Take the manifest in ``release`` as a start: copy it into ``imported/<digest>/``; return that directory."""
        if not (artifact_dir / MANIFEST).exists():
            raise ValueError(f"artifact is missing {MANIFEST}")
        checkpoint = self.validated(TinkerCheckpoint.read(artifact_dir))
        digest = hashlib.sha256(json.dumps(checkpoint.to_dict(), sort_keys=True).encode()).hexdigest()
        imported = self._root / "imported" / digest
        if not imported.exists():
            # The manifest names the remote state; the exported adapter is not needed to train from it.
            checkpoint.write(imported)
        return str(imported)

    @property
    def artifact_format(self) -> ArtifactFormat | None:
        """``None`` by default: a snapshot holds only its manifest, and the weights stay on the service."""
        return None

    @property
    def state_dir(self) -> Path:
        return self._root

    @property
    def shares_scenarios(self) -> bool:
        return True

    @property
    def base_start(self) -> str:
        """``base/``, which this property prepares on its first use in a process (``prepared_base``)."""
        return str(self.prepared_base())

    def job_parameters(self, parameters: Mapping[str, Any]) -> TinkerJobParameters:
        """``parameters`` over the component's defaults; ``ValueError`` for an unknown or a malformed parameter."""
        return TinkerJobParameters.from_parameters(parameters, self._config)

    def parameters_schema(self, kind: JobDataKind) -> dict[str, Any]:
        return tinker_parameters_schema(kind)

    def check_job(self, job: TrainingJob) -> None:
        parameters = self.job_parameters(job.parameters)
        if job.start is not None:
            start = Path(job.start)
            # A job from the base start reads ``base/`` as this process prepares it.
            self.start_at(self.prepared_base() if start == self._base else start)
        if job.kind == "task":
            tasks = [item for item in read_job_data(job.data_dir, job.data_id).samples if isinstance(item, TaskSample)]
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
        self.job_parameters(parameters)
        rollout_settings(parameters)

    def close_resources(self) -> None:
        try:
            self._client.close()
            if self._sampler is not None:
                self._sampler.close()
        finally:
            if self._state_lock is not None:
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
                trajectories(JobData(job.data_id, samples)), RolloutScores.from_json(kept["scores"])
            )
        if self._sampler is None:
            raise RuntimeError("this Tinker component has no sampler, so it cannot roll tasks out")
        tasks = tuple(
            item for item in read_job_data(job.data_dir, job.data_id).samples if isinstance(item, TaskSample)
        )
        self.warm_sampler(start.sampler_path)
        chat = SamplerChat(
            self._sampler,
            start.sampler_path,
            model=self._model,
            # A fresh start names no sampler: its rollouts sample the base model.
            runtime_load_id=f"tinker:{start.sampler_path or f'base:{self._model}'}",
            topk_logprobs=0 if self.score_centering is None else self.score_centering.top_k,
        )
        rollouts = run_train_tasks(
            tasks,
            settings,
            chat,
            model=self._model,
            work_dir=directory / "rollouts",
            start=job.start,
            model_adapter=start.sampler_path,
        )
        write_json(
            record,
            {
                "samples": JobData(job.data_id, rollouts.samples).to_dict(),
                "scores": rollouts.scores.to_json(),
            },
        )
        return rollouts

    def warm_sampler(self, sampler_path: str | None) -> None:
        """Sample one token from ``sampler_path`` (None: the base model) until the sampler answers.

        A service such as Spindle starts a sampler at its first request, which can take minutes. The agents of
        a task job's rollouts would spend that time against their own timeouts and fail for it, so the job
        waits here first, retrying each sample that times out, for at most ``SAMPLER_WARM_TIMEOUT_S`` seconds.
        """
        if self._sampler is None:
            raise RuntimeError("this Tinker component has no sampler, so it cannot roll tasks out")
        prompt = self._sampler.render([{"role": "user", "content": "Hi"}], template_kwargs={})
        deadline = time.monotonic() + SAMPLER_WARM_TIMEOUT_S
        while True:
            try:
                self._sampler.sample(sampler_path, prompt, {"max_tokens": 1}, topk_logprobs=0)
                return
            except TimeoutError:
                if time.monotonic() >= deadline:
                    raise

    def run(self, job: TrainingJob, directory: Path) -> Mapping[str, Any]:
        parameters = self.job_parameters(job.parameters)
        start = self.start_at(Path(job.start))
        data = read_job_data(job.data_dir, job.data_id)
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
        # A trajectory that serving recorded carries no tokens: the component renders it with the base model's chat
        # template, and the client computes its log probabilities with the weights of the job's start.
        if needs_rendering(data):
            data, left_out = rendered_job_data(
                data, self._client.chat_template(self._config.chat_template_kwargs or {})
            )
            if left_out:
                rollout_metrics["unrenderable_samples"] = len(left_out)
                if not data.samples:
                    return rollout_metrics
        prepared = prepare_tinker_job(
            data,
            parameters.objective,
            start.state_for(parameters.objective),
            parameters.scheduling,
            batch_size=parameters.batch_size,
            kl_coef=parameters.kl_coef,
            score_centering=self.score_centering,
        )
        if prepared.loss is None:
            # The objective skipped the job's data: the job trains nothing and saves no checkpoint.
            return {**prepared.metrics, **rollout_metrics}
        steps = prepared.steps
        schedule = resolve_learning_rate_schedule_state(
            start.learning_rate_schedule_state, parameters.learning_rate_schedule
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
                objective=(
                    (parameters.objective if prepared.next_algorithm_state else None) if final else start.objective
                ),
            )
            target = directory / str(last)
            current.write(target)
            self.complete_artifact(current, target)
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


@dataclass(frozen=True)
class TinkerBaseFiles(BaseModelFiles):
    """The base start of a Tinker component as the model files of its default start.

    Values: ``component`` is the ``TinkerJobComponent`` whose ``base/`` the files are. ``directory`` returns
    ``component.prepared_base()``, so a new scenario's root snapshot makes the base start on its first call in a
    process. Purpose: ``TinkerJobComponent`` subclasses give it in their ``default_start``, which itself makes
    nothing.
    """

    component: TinkerJobComponent

    def directory(self) -> Path:
        return self.component.prepared_base()


class TinkerTrainingComponent(TinkerJobComponent):
    """Train a LoRA of rank ``config.lora_rank`` on the base model, on Tinker or a Tinker-compatible service.

    The base start, ``base/``, is the untrained adapter's checkpoint, which the
    client saves on first use (``TinkerClient.initialize``): before the training
    process runs jobs, or when a new scenario's root snapshot holds the
    default start (``prepared_base``). Serving reads
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
            # The untrained adapter is the base checkpoint: the default start holds it as a snapshot.
            self._client.initialize().write(self._base)

    def validated(self, checkpoint: TinkerCheckpoint) -> TinkerCheckpoint:
        checkpoint.check_compatible(self._model, service=self._config.base_url)
        if checkpoint.lora_rank != self._lora_rank:
            raise ValueError(
                f"the Tinker checkpoint is no rank {self._lora_rank} LoRA of {self._model}: "
                f"its rank is {checkpoint.lora_rank}"
            )
        return checkpoint

    def default_start(self) -> DefaultStart:
        """The untrained adapter's release: its manifest."""
        inputs = {"model": self._model, "lora_rank": str(self._lora_rank)}
        if self._config.base_url is not None:
            inputs["service"] = self._config.base_url
        return DefaultStart(model_files=TinkerBaseFiles(self), inputs=inputs)

    def check_artifact(self, files: ArtifactFiles) -> None:
        """A manifest of a checkpoint of this component's model, rank and service, whose state path resolves there."""
        self.check_manifest(files)

    def import_start(self, artifact_dir: Path) -> str:
        return self.import_manifest(artifact_dir)
