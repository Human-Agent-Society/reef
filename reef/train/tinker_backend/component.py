"""Tinker as a training component: each job trains its data remotely, from the job's start checkpoint.

``TinkerJobComponent`` is the job loop that every component on a Tinker-compatible service shares;
``TinkerTrainingComponent`` trains a LoRA on it, and ``reef.train.tinker_backend.full`` all the parameters on
Spindle. A job starts from ``job.start``, a directory that holds a Tinker checkpoint
manifest, or the backend's default start, which starts a new model of the base model. The manifest
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
import os
import threading
import time
import uuid
from abc import ABC
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, TextIO

from reef.artifact.files import ArtifactFiles
from reef.core.chat_rendering import needs_rendering, rendered_job_data
from reef.core.evaluation import RolloutScores
from reef.core.job_data import JobData, TaskSample, trajectories
from reef.core.weight_layout import ArtifactFormat
from reef.storage.durable_files import read_json, write_json
from reef.storage.job_data import JobDataKind
from reef.tinker.checkpoint import MANIFEST
from reef.tinker.sampling import TinkerSampler
from reef.train.component import SavedCheckpoint, TrainingJob
from reef.train.learning_rate import learning_rate_metrics, resolve_learning_rate_schedule_state
from reef.train.local_jobs import LocalJobComponent
from reef.train.task_rollouts import RolloutSettings, RolloutTokens, TaskJobRollouts, run_train_tasks, task_name
from reef.train.tinker_backend.checkpoint import FreshStart, TinkerCheckpoint
from reef.train.tinker_backend.client import TinkerClient
from reef.train.tinker_backend.config import ScoreCentering, TinkerServiceSettings, TinkerSettings
from reef.train.tinker_backend.facts import SpindleFullFacts, TinkerFacts
from reef.train.tinker_backend.parameters import TinkerJobParameters, rollout_settings
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
    ``kl_coef`` and ``learning_rate`` default to the component's ``TinkerServiceSettings``.
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
    samples, and reports the rollouts' metrics (``TaskJobRollouts.metrics``)
    in the job's metrics.

    A sample that cannot render, or whose tokens after the first are more than
    the model's context window (``TinkerClient.context_window``), does not
    train; the job metric ``unrenderable_samples`` counts these samples.

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
    releases) and ``jobs/`` (see ``LocalJobComponent``). ``base/`` holds the files of the backend's default start
    (``facts.default_start()``), which ``base_start`` writes without a call to the service: a job from it starts a
    new model of the base model (``FreshStart``). The constructor writes nothing. A saved checkpoint is
    ``jobs/<job_id>/<step>/``, which holds its manifest alone. A checkpoint
    directory is both the start of a later job and the snapshot Reef publishes.
    The weights stay on the service. A subclass decides the parameterization:
    which checkpoints it trains (``facts.validated``), which directories start a new model (``start_at``), what a
    job from such a start trains (``training_start``), and which snapshots it imports.

    ``facts`` are the backend's facts of the same model and settings (``TinkerFacts`` or ``SpindleFullFacts``):
    the name, the artifact format, the sharing of scenarios and jobs, the parameters schema and ``validated`` are
    its answers.
    """

    def __init__(
        self,
        base_model: str,
        config: TinkerServiceSettings,
        client: TinkerClient,
        *,
        facts: TinkerFacts | SpindleFullFacts,
        sampler: TinkerSampler | None = None,
        score_centering: ScoreCentering | None = None,
    ) -> None:
        self.facts = facts
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
        #: Guards the writes of this process into ``base/``.
        self._base_lock = threading.Lock()

    def start_jobs(self) -> None:
        """Take ``state_dir`` for this process, then run the jobs: one process at a time runs a state directory's
        jobs. Raise ``ValueError`` when another process holds it; a second call changes nothing."""
        if self._state_lock is not None:
            return
        lock = (self._root / ".lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            lock.close()
            raise ValueError("Tinker state_dir is already owned by another runtime") from None
        self._state_lock = lock
        super().start_jobs()

    def validated(self, checkpoint: TinkerCheckpoint) -> TinkerCheckpoint:
        """``checkpoint`` when this component trains it: of its model, service and parameterization; else ``ValueError``."""
        return self.facts.validated(checkpoint)

    def start_at(self, directory: Path) -> TinkerCheckpoint | FreshStart:
        """The start that ``directory`` holds: a checkpoint manifest that ``validated`` takes; else ``ValueError``.

        A subclass also reads its default start, which gives ``FreshStart``. The method calls no service.
        """
        if not (directory / MANIFEST).is_file():
            raise ValueError(f"the start {directory} holds no Tinker checkpoint manifest")
        return self.validated(TinkerCheckpoint.read(directory))

    def job_start(self, job: TrainingJob) -> TinkerCheckpoint | FreshStart:
        """The start that ``job.start`` holds (``start_at``). A job from ``base/`` reads the directory as this
        process writes it (``base_start``), so it follows this process's configuration."""
        start = Path(job.start)
        return self.start_at(Path(self.base_start) if start == self._base else start)

    def training_start(self, start: TinkerCheckpoint | FreshStart) -> TinkerCheckpoint | FreshStart:
        """What a job whose start is ``start`` trains from: ``start`` itself by default.

        Only the process that runs the job calls it, so a subclass may call the service here.
        """
        return start

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
        checkpoint = self.facts.checkpoint_in(artifact_dir)
        digest = hashlib.sha256(json.dumps(checkpoint.to_dict(), sort_keys=True).encode()).hexdigest()
        imported = self._root / "imported" / digest
        if not imported.exists():
            # The manifest names the remote state; the exported adapter is not needed to train from it.
            checkpoint.write(imported)
        return str(imported)

    @property
    def name(self) -> str:
        return self.facts.name

    @property
    def default_start_inputs(self) -> Mapping[str, str]:
        return self.facts.default_start_inputs

    @property
    def artifact_format(self) -> ArtifactFormat | None:
        return self.facts.artifact_format

    @property
    def state_dir(self) -> Path:
        return self._root

    @property
    def shares_scenarios(self) -> bool:
        return self.facts.shares_scenarios

    @property
    def concurrent_jobs(self) -> bool:
        return self.facts.concurrent_jobs

    @property
    def base_start(self) -> str:
        """``base/``, after it holds the files of the default start of this configuration (``facts.default_start()``).

        The property writes each file that is absent or that another configuration wrote, and calls no service.
        ``start_at`` of the directory is ``FreshStart``. It keeps the other files of ``base/``.
        """
        with self._base_lock:
            for relative, text in self.facts.default_start().files.items():
                path = self._base / relative
                if path.is_file() and path.read_text(encoding="utf-8") == text:
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                # Another process of the deployment can read the directory: it sees the old file or the new one.
                temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
                temporary.write_text(text, encoding="utf-8")
                os.replace(temporary, path)
        return str(self._base)

    def job_parameters(self, parameters: Mapping[str, Any]) -> TinkerJobParameters:
        """``parameters`` over the component's defaults; ``ValueError`` for an unknown or a malformed parameter."""
        return TinkerJobParameters.from_parameters(parameters, self._config)

    def parameters_schema(self, kind: JobDataKind) -> dict[str, Any]:
        return self.facts.parameters_schema(kind)

    def check_job(self, job: TrainingJob) -> None:
        parameters = self.job_parameters(job.parameters)
        self.job_start(job)
        if job.kind == "task":
            tasks = [item for item in job.read_data().samples if isinstance(item, TaskSample)]
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
                trajectories(JobData(job.data_id, samples)),
                RolloutScores.from_json(kept["scores"]),
                tuple(RolloutTokens(**item) for item in kept["tokens"]),
                kept["left_out"],
            )
        if self._sampler is None:
            raise RuntimeError("this Tinker component has no sampler, so it cannot roll tasks out")
        tasks = tuple(item for item in job.read_data().samples if isinstance(item, TaskSample))
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
            max_tokens=self._client.context_window(),
        )
        write_json(
            record,
            {
                "samples": JobData(job.data_id, rollouts.samples).to_dict(),
                "scores": rollouts.scores.to_json(),
                "tokens": [asdict(item) for item in rollouts.tokens],
                "left_out": rollouts.left_out,
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
        start = self.training_start(self.job_start(job))
        data = job.read_data()
        rollout_metrics: dict[str, Any] = {}
        if parameters.rollout is not None:
            rollouts = self.task_rollouts(job, directory, start, parameters.rollout)
            rollout_metrics = rollouts.metrics()
            if not rollouts.samples:
                # No task has a group of two samples to compare, so the job trains nothing.
                return rollout_metrics
            data = JobData(data.data_id, rollouts.samples)
        # A trajectory that serving recorded carries no tokens: the component renders it with the base model's chat
        # template, and the client computes its log probabilities with the weights of the job's start.
        left_out = 0
        if needs_rendering(data):
            data, unrenderable = rendered_job_data(
                data, self._client.chat_template(self._config.chat_template_kwargs or {})
            )
            left_out += len(unrenderable)
        # Tinker refuses a whole optimizer step for one row whose targets, every token but the first, are longer than
        # the model's context window, so such a sample is left out as one that cannot render.
        window = self._client.context_window()
        if window is not None:
            fitting = tuple(sample for sample in trajectories(data) if len(sample.training["tokens"]) - 1 <= window)
            left_out += len(data.samples) - len(fitting)
            data = JobData(data.data_id, fitting, request=data.request)
        if left_out:
            rollout_metrics["unrenderable_samples"] = left_out
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


class TinkerTrainingComponent(TinkerJobComponent):
    """Train a LoRA of rank ``config.lora_rank`` on the base model, on Tinker or a Tinker-compatible service.

    The default start is a base root of the model, the rank and the service (``TinkerFacts.default_start``), which
    ``base/`` holds. A job from it trains a new LoRA of the base model:

    - With ``config.seed``, the service makes the LoRA from the seed for each such job (``FreshStart``). Every job
      from the default start thus begins with equal weights.
    - With ``config.seed`` null, the service chooses the initial adapter. The component then saves one untrained
      adapter on the service (``TinkerClient.initialize``) as ``base/``'s checkpoint manifest, when the first job
      from the default start runs, and each such job restores it. Every job from the default start of one
      ``state_dir`` thus begins with equal weights here too.

    Serving reads each saved checkpoint from its artifact: Tinker's sampler
    (``reef.inference.tinker``) reads the manifest, and a serving runtime that
    loads the weights into its own engines downloads the adapter from the
    manifest's ``sampler_path`` (``TinkerSamplerWeights``). An import is a base root of this configuration, or a
    manifest of a LoRA of the model, rank and service.
    """

    facts: TinkerFacts

    def __init__(
        self, base_model: str, config: TinkerSettings, client: TinkerClient, *, sampler: TinkerSampler | None = None
    ) -> None:
        super().__init__(base_model, config, client, facts=TinkerFacts.of(base_model, config), sampler=sampler)
        #: The seed of a new LoRA, or None when the service chooses the initial adapter.
        self.seed = config.seed

    def start_at(self, directory: Path) -> TinkerCheckpoint | FreshStart:
        """A base root of this configuration, which starts a new LoRA; else a manifest that ``validated`` takes;
        else ``ValueError``."""
        if self.facts.base_root_in(directory) is not None:
            return FreshStart(self._model)
        return super().start_at(directory)

    def training_start(self, start: TinkerCheckpoint | FreshStart) -> TinkerCheckpoint | FreshStart:
        """``start`` for a checkpoint, and for a new LoRA with a seed. Without a seed, a new LoRA is the untrained
        adapter that ``base/``'s manifest names, which the first call saves on the service."""
        if isinstance(start, TinkerCheckpoint) or self.seed is not None:
            return start
        with self._base_lock:
            if not (self._base / MANIFEST).is_file():
                self._client.initialize().write(self._base)
            return self.validated(TinkerCheckpoint.read(self._base))

    def check_artifact(self, files: ArtifactFiles) -> None:
        """A base root of this component's model, rank and service, or a manifest of a checkpoint of them whose
        state path resolves on the service."""
        if not self.is_own_base_root(files):
            self.check_manifest(files)

    def import_start(self, artifact_dir: Path) -> str:
        """``base_start`` for a base root of this configuration, else the manifest's start (``import_manifest``)."""
        if self.facts.base_root_in(artifact_dir) is not None:
            return self.base_start
        return self.import_manifest(artifact_dir)
