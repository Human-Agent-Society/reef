"""The harness training component: each training job is one run of the harness method (RFC #1048, harness supplement).

A harness job runs in the training service, as a weight job does. The scenario's trainer submits a job through
its ``JobLoop``; a training worker runs it here, and control commits the selected tree as an artifact by the
``JobLoop`` rules. A job of this component holds:

- ``start``: the tree that the job builds on. ``base:<checksum>`` is the deployment's seed tree; any other start is a
  directory that holds a tree, which ``import_start`` or an earlier job's checkpoint wrote.
- ``data``: the job data directory of the job's material: the traces of traffic and the train tasks of an
  offline job.
- ``component`` and ``previous_job_id``, the fields of every ``TrainingJob``: the scenario component that the job
  evolves, and the last job whose end the trainer committed, whose harness state the job continues.
- ``parameters``, which the harness stage gives (``TraceJobData.job_parameters``): ``evaluation_tasks``, the eval
  split of the scenario's imported tasks, which the job evaluates on instead of its configured tasks, when the
  stage names it; and ``request``, the training request that the job runs
  (``reef.core.training_request.request_parameter``), or ``None``. A job that an earlier Reef queued holds
  ``component`` and ``previous_job`` in its parameters instead.

A job's directory keeps ``outcome.json``: the job's outcome, with the count of jobs that ended and the
``HarnessState`` after it; and ``tree/``, its selected tree. A job continues the state after ``previous_job_id``,
so the harness state follows the jobs that control committed: a job that ran but whose end no commit recorded
changes no later job.
Under ``EvolutionSettings.component_state_dir(scenario, component)`` the component keeps the history and job records
of the backend, and ``progress.json`` while a job runs.

The component builds itself (RFC #1390). ``HarnessTrainingComponent.from_settings`` builds the method, the scorer,
the collector, the run executor and the agent settings that the ``evolution`` section names; the component then
builds, for each scenario component at its first job, the backend with its history, its evaluator, its tasks and
its bound agent, and for each job the selection policy. No recipe takes part.
"""

from __future__ import annotations

import json
import shutil
import threading
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Self

from reef.artifact.files import ArtifactFiles
from reef.artifact.repository import DefaultStart
from reef.core.errors import ConfigError, InternalError, ReefError
from reef.core.evaluation import CandidateEvaluationPlugin, UpdateCandidate
from reef.core.training_request import request_from_parameters
from reef.core.weight_layout import ArtifactFormat
from reef.evaluation.harness import (
    ComparePolicy,
    EvaluationTasks,
    FloorPolicy,
    HarnessTaskEvaluator,
    History,
    NoEvaluationPolicy,
    TaskCollector,
    TaskScorer,
    evaluation_selection,
)
from reef.execution.sandbox.config import build_executor
from reef.execution.sandbox.process import ProgramExecutor
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.tasks.harness_models import HarnessModels
from reef.harness.tasks.model_binding import ModelBindings
from reef.harness.tasks.vendor_install import AdapterProgram
from reef.harness.tree import HarnessTree
from reef.storage.durable_files import read_json, write_json
from reef.storage.job_data import JobDataKind
from reef.train.backend import ArtifactTextFiles
from reef.train.component import JobStatus, SavedCheckpoint, TrainingJob
from reef.train.harness_backend.agent.settings import AgentSettings, agent_settings
from reef.train.harness_backend.backend import HarnessBackend, JobLimits, JobProgress
from reef.train.harness_backend.contract import Method
from reef.train.harness_backend.facts import HARNESS_COMPONENT, HarnessFacts
from reef.train.harness_backend.records import JobRecord
from reef.train.harness_backend.settings import HarnessSettings, configured_tasks, plugin_name
from reef.train.harness_backend.state import HarnessState
from reef.train.local_jobs import JOB_FILE, LocalJobComponent

#: The backend's history and job records under the state directory of a scenario component.
HISTORY_DIR = "history"
JOB_RECORDS_DIR = "steps"
PROGRESS_FILE = "progress.json"
#: The outcome of a job and the harness state after it: written before the job's checkpoint, so a resumed job
#: does not run the method again.
OUTCOME_FILE = "outcome.json"
TREE_DIR = "tree"


def job_component(job: TrainingJob) -> str:
    """The scenario component that ``job`` evolves; ``ValueError`` without one."""
    component = job.component
    if not isinstance(component, str) or not component:
        raise ValueError("a harness job names the scenario component that it evolves")
    return component


def previous_job(submitted: Mapping[str, Any]) -> str | None:
    """The previous job of the job file ``submitted``."""
    return submitted.get("previous_job_id")


def read_job_progress(path: Path) -> JobProgress | None:
    """The running job's progress that ``path`` holds, or ``None`` when no job runs."""
    value = read_json(path)
    return None if value is None else JobProgress.from_json(value)


class ProgressFile:
    """Writes the progress of ``backend``'s running job to ``path`` when it changed, looking every
    ``interval_seconds``, in a thread.

    Values: ``interval_seconds`` > 0. The thread runs from ``__enter__`` to ``__exit__``; ``__exit__`` removes the
    file. Purpose: ``HarnessTrainingComponent`` runs one around each job, so another process, such as the one that
    serves the request pages, reads where the job stands (``read_job_progress``).
    """

    def __init__(self, backend: HarnessBackend, path: Path, *, interval_seconds: float = 0.1) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.backend = backend
        self.path = path
        self.interval_seconds = interval_seconds
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.run, name="reef-harness-progress", daemon=True)

    def __enter__(self) -> ProgressFile:
        self.thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stopped.set()
        self.thread.join()
        self.path.unlink(missing_ok=True)

    def run(self) -> None:
        written: dict[str, Any] | None = None
        while not self.stopped.wait(self.interval_seconds):
            progress = self.backend.job_progress
            value = None if progress is None else progress.to_json()
            if value == written:
                continue
            if value is None:
                self.path.unlink(missing_ok=True)
            else:
                write_json(self.path, value)
            written = value


class HarnessTrainingComponent(LocalJobComponent):
    """The training component whose jobs are harness evolution jobs: each job runs the method once.

    Values: ``settings`` is the deployment's harness configuration, and ``models`` its models. ``method`` is the
    deployment's ``Method``. ``scorer`` scores each run, ``None`` when the evaluation policy is ``none`` and the
    jobs are not offline. ``collector`` picks the prompts that collection adds, ``None`` for the default.
    ``executor`` runs the runs, not yet bound to the adapter's program. ``agent`` is ``evolution.agent`` read,
    ``None`` when no coding agent runs; a method that ``runs_agent`` needs one unless it ``runs_without_agent``.
    ``facts`` are the ``HarnessFacts`` of ``settings``: the name, the artifact format, the sharing of scenarios, the
    default start's inputs and the tree checks are its answers. ``descriptor`` is ``facts.descriptor``, the adapter
    that ``settings.evolution.adapter`` names. The component's root is
    ``settings.evolution.training_dir()``: it holds ``jobs/``, one directory per job (``LocalJobComponent``), and
    ``starts/``, the trees that ``import_start`` took. A job's weights never exist: its snapshots are trees, so
    ``artifact_format`` is ``None``. A job depends only on its start, data and parameters, and on the state of its
    scenario component, which this component keeps per scenario; so scenarios share it. ``status`` of a job that
    raised carries the job's metrics and error.

    The model under test reaches the method as ``services.models.served``, and the adapter's ``model_binding``
    templates write it into each evaluation run, so neither the seed nor a kept tree ever carries a provider.

    Purpose: the service builds one for each harness recipe of the deployment (``from_settings``, in
    ``reef.service.assembly.with_harness_components``); the dispatcher queues its jobs, and the training service
    runs them. Code that already holds the method and the scorer as objects calls the constructor.
    """

    def __init__(
        self,
        settings: HarnessSettings,
        models: HarnessModels,
        *,
        method: Method,
        executor: ProgramExecutor,
        scorer: TaskScorer | None = None,
        collector: TaskCollector | None = None,
        agent: AgentSettings | None = None,
    ) -> None:
        if not isinstance(method, Method):
            raise ValueError("harness evolution requires a Method")
        if method.runs_agent and not method.runs_without_agent and agent is None:
            raise ConfigError(
                f"{type(method).__name__} runs a coding agent: set evolution.agent, with a sandbox this host "
                "can run (sandbox: none runs the agent without isolation)"
            )
        if scorer is None and settings.evaluation.policy != "none":
            raise ValueError("harness evolution requires a scorer unless the evaluation policy is none")
        if scorer is None and settings.runs_offline:
            # An offline job plays the imported train tasks and scores each run.
            raise ConfigError("offline runs score each train task, so they need evaluation.scorer")
        facts = HarnessFacts.from_settings(settings)
        if scorer is not None:
            evaluation_selection(scorer, settings.worker_executor)
        root = settings.evolution.training_dir()
        super().__init__(root / "jobs")
        self.settings = settings
        self.models = models
        self.method = method
        self.scorer = scorer
        self.collector = collector
        self.executor = executor
        self.agent = agent
        self.facts = facts
        self.descriptor = facts.descriptor
        self.starts_root = root / "starts"
        #: Each scenario component's backend, made at its first job.
        self.backends: dict[tuple[str, str], HarnessBackend] = {}

    @classmethod
    def from_settings(cls, settings: HarnessSettings, models: HarnessModels, *, environ: Mapping[str, str]) -> Self:
        """The component of the deployment whose configuration is ``settings``, on ``models``.

        Build what the sections name: the ``Method`` of ``evolution.method``, the ``TaskScorer`` and the
        ``TaskCollector`` of ``evolution.evaluation``, the run executor of ``evolution.executor`` and
        ``evolution.sandbox``, and the agent settings of ``evolution.agent``. ``environ`` holds the variables that
        a sandbox and the agent's isolation read. Raise ``ConfigError`` for a setting that names no usable plugin,
        executor or agent.
        """
        evolution, evaluation = settings.evolution, settings.evaluation
        try:
            if evolution.method is None:
                raise ValueError("evolution.method must name the Method that evolves the tree")
            method = Method.from_setting(evolution.method, setting="evolution.method")
            scorer = (
                None
                if evaluation.scorer is None
                else TaskScorer.from_setting(evaluation.scorer, setting="evolution.evaluation.scorer")
            )
            collector = (
                None
                if evaluation.collector is None
                else TaskCollector.from_setting(evaluation.collector, setting="evolution.evaluation.collector")
            )
            executor = build_executor({"executor": evolution.executor, "sandbox": evolution.sandbox}, environ=environ)
            agent = agent_settings(evolution.agent, environ, state_dir=Path(evolution.state_dir).expanduser())
            return cls(
                settings, models, method=method, executor=executor, scorer=scorer, collector=collector, agent=agent
            )
        except (InternalError, ConfigError):
            raise
        except (TypeError, ValueError, ReefError) as exc:
            raise ConfigError(str(exc)) from exc

    @property
    def name(self) -> str:
        return self.facts.name

    @property
    def artifact_format(self) -> ArtifactFormat | None:
        return self.facts.artifact_format

    @property
    def shares_scenarios(self) -> bool:
        return self.facts.shares_scenarios

    @property
    def concurrent_jobs(self) -> bool:
        return self.facts.concurrent_jobs

    def parameters_schema(self, kind: JobDataKind) -> dict[str, Any] | None:
        return self.facts.parameters_schema(kind)

    def default_start(self) -> DefaultStart:
        """The seed tree, as ``HarnessEvolutionRecipe.default_start`` gives it."""
        return DefaultStart(files=dict(self.facts.seed.files), inputs=self.facts.default_start_inputs)

    def check_task_jobs(self, parameters: Mapping[str, Any]) -> None:
        """Accept: an offline job runs the head tree on its train tasks."""
        return

    def check_artifact(self, files: ArtifactFiles) -> None:
        self.check_tree(HarnessTree({path: files.read_bytes(path).decode("utf-8") for path in files.paths()}))

    def check_job(self, job: TrainingJob) -> None:
        job_component(job)
        tasks = job.parameters.get("evaluation_tasks")
        if tasks is not None and (not isinstance(tasks, list) or not all(isinstance(task, str) for task in tasks)):
            raise ValueError("parameters.evaluation_tasks of a harness job must be a list of task paths or null")
        request_from_parameters(job.parameters)

    def changed_parts(self, start: str, checkpoint: SavedCheckpoint) -> frozenset[str]:
        """The parts of the tree that the job changed: the parts of the paths of the diff from the start's tree to
        the checkpoint's."""
        changes = HarnessTree.read(Path(start)).diff(HarnessTree.read(Path(checkpoint.artifact_path))).changes
        return frozenset(self.descriptor.layout.part_names(changes).values())

    def import_start(self, artifact_dir: Path) -> str:
        """Copy the tree of ``release`` to ``starts/<checksum>`` and return that directory; ``ValueError`` when the
        tree fails validation.

        The tree is the snapshot's text files without the repository's own files (``TextFileTree``), as a client
        pulls it: a Git LFS release directory also holds ``.gitattributes``.
        """
        tree = self.facts.tree_in(artifact_dir)
        directory = self.starts_root / tree.checksum()
        if not directory.is_dir():
            staging = self.starts_root / f".{tree.checksum()}.partial"
            shutil.rmtree(staging, ignore_errors=True)
            tree.write(staging)
            staging.replace(directory)
        return str(directory)

    def status(self, job_id: str) -> JobStatus:
        status = super().status(job_id)
        if status.state != "failed":
            return status
        # A job that raised failed with its metrics and error, which its outcome keeps.
        outcome = read_json(self.job_directory(job_id) / OUTCOME_FILE)
        if outcome is None or outcome["error"] is None:
            return status
        return JobStatus(job_id, "failed", status.checkpoints, metrics=outcome["metrics"], error=outcome["error"])

    def backend_for(self, scenario: str, component: str) -> HarnessBackend:
        """The backend of ``component``'s jobs in ``scenario``: made at the first job, then kept."""
        key = (scenario, component)
        if key not in self.backends:
            self.backends[key] = self.build_backend(scenario, component)
        return self.backends[key]

    def check_tree(self, tree: HarnessTree) -> None:
        """Raise ``ValueError`` for a tree that fails validation for the deployment's adapter."""
        self.facts.check_tree(tree)

    def build_policy(self, backend: HarnessBackend) -> CandidateEvaluationPlugin:
        """The selection policy that ``evolution.evaluation.policy`` names, measuring through ``backend``."""
        evaluation = self.settings.evaluation
        if evaluation.policy == "compare":
            return ComparePolicy(backend, statistic=evaluation.statistic, min_margin=evaluation.min_margin)
        if evaluation.policy == "floor":
            return FloorPolicy(backend, floor_score=evaluation.floor_score)
        return NoEvaluationPolicy()

    def build_backend(self, scenario: str, component: str) -> HarnessBackend:
        """A new backend of ``component`` in ``scenario``: the binary installed, the history, the evaluator, the
        tasks, the records and the coding agent bound. It reads no snapshots: each job names its head.

        Raise ``ConfigError`` when the state directory cannot hold the history, when the deployment has no
        inference runtime, and for a task manifest that names no eval task.
        """
        evolution, evaluation = self.settings.evolution, self.settings.evaluation
        # Each component of a composed scenario keeps its own history and records.
        root = evolution.component_state_dir(scenario, component)
        try:
            history = History.open(root / HISTORY_DIR)
        except OSError as exc:
            raise ConfigError(f"evolution.state_dir cannot hold {root}: {exc}") from exc
        models = self.models.bindings(scenario)
        evaluator = None if self.scorer is None else self.build_evaluator(models, history, self.scorer)
        return HarnessBackend(
            descriptor=self.descriptor,
            method=self.method,
            # Only a method that runs the agent through its services gets it bound here.
            agent=self.agent.bind() if self.agent is not None and self.method.runs_agent else None,
            evaluator=evaluator,
            tasks=EvaluationTasks(
                seed_tasks=configured_tasks(evaluation),
                collect_failures=evaluation.collect_failures,
                collector=self.collector,
                max_collected=evaluation.max_collected_tasks,
                max_collected_per_client=evaluation.max_collected_per_client,
            ),
            history=history,
            models=models,
            limits=JobLimits(
                max_steps=evolution.max_steps,
                max_failure_streak=evolution.max_failure_streak,
                max_model_calls_per_step=evolution.max_model_calls_per_step,
                max_runs_per_step=evolution.max_runs_per_step,
            ),
            records=JobRecord(root / JOB_RECORDS_DIR),
            task_directories=self.settings.task_directories(self.descriptor),
        )

    def build_evaluator(self, models: ModelBindings, history: History, scorer: TaskScorer) -> HarnessTaskEvaluator:
        """The run evaluator: the adapter's binary installed, the worker pool placed, ``scorer`` bound."""
        evolution, evaluation = self.settings.evolution, self.settings.evaluation
        descriptor: AdapterDescriptor = self.descriptor
        executor, binary = self.executor.for_program(AdapterProgram(descriptor), binary=evolution.binary)
        executor.preflight()
        selection, requirements = evaluation_selection(scorer, self.settings.worker_executor)
        # The evaluation context names the scorer by its setting. A scorer built in Python has no setting: its
        # class and the instance name it, so no other scorer reuses its scores, and a restart reuses none of them.
        setting = evaluation.scorer
        if setting is None:
            scorer_name = f"{type(scorer).__module__}:{type(scorer).__qualname__}@{id(scorer):x}"
        else:
            scorer_name = plugin_name(setting)
        return HarnessTaskEvaluator(
            descriptor=descriptor,
            scorer=scorer,
            scorer_name=scorer_name,
            models=models,
            binary=binary,
            binary_setting=evolution.binary,
            executor_setting=json.dumps(
                {
                    "executor": evolution.executor,
                    "sandbox": None if evolution.sandbox is None else dict(evolution.sandbox),
                },
                sort_keys=True,
                default=str,
            ),
            task_timeout_s=evaluation.task_timeout_s,
            task_repeats=evaluation.repeats,
            executor=executor,
            forbid_residue=evaluation.forbid_residue,
            worker_selection=selection,
            worker_requirements=requirements,
            history=history,
            task_directories=self.settings.task_directories(descriptor),
            agent_host=evaluation.agent_host,
        )

    def state_after(self, job_id: str | None) -> tuple[int, HarnessState]:
        """The count of jobs that ended and the ``HarnessState`` after the job ``job_id``, as its outcome keeps them;
        the initial ``(0, HarnessState())`` for ``None``.

        A job that has no outcome, such as one cancelled before it settled, changed nothing: its state is the
        state after its previous job (``previous_job_id``).
        """
        while job_id is not None:
            directory = self.job_directory(job_id)
            outcome = read_json(directory / OUTCOME_FILE)
            if outcome is not None:
                return int(outcome["steps"]), HarnessState.from_json(outcome["harness"])
            submitted = read_json(directory / JOB_FILE)
            job_id = None if submitted is None else previous_job(submitted)
        return 0, HarnessState()

    def run(self, job: TrainingJob, directory: Path) -> Mapping[str, Any]:
        """Run the job from the state after its previous job, then save its selected tree as the
        job's checkpoint.

        The job's outcome is durable before its checkpoint, so a resumed job does not run the method again. The
        returned metrics are the job's metrics when the job kept no tree; a kept tree's checkpoint carries them
        instead. When the method or the evaluation raised, the job fails: ``status`` then reports the metrics of
        what the job reached, ``skipped`` (``step failed``, or ``request failed`` for a request's job) and ``error``.
        """
        outcome = read_json(directory / OUTCOME_FILE)
        if outcome is None:
            outcome = self.run_harness_job(job, directory)
            write_json(directory / OUTCOME_FILE, outcome)
        if outcome["error"] is not None:
            raise RuntimeError(outcome["error"])
        checkpoint = outcome["checkpoint"]
        if checkpoint is None:
            return outcome["metrics"]
        if not self.saved(directory):
            self.save_checkpoint(
                directory,
                SavedCheckpoint(0, checkpoint["path"], checkpoint["path"], outcome["metrics"]),
            )
        return {}

    def run_harness_job(self, job: TrainingJob, directory: Path) -> dict[str, Any]:
        """The outcome of ``job``: ``{steps, harness, metrics, checkpoint, error}``. ``steps`` and ``harness``
        are the count of jobs that ended and the state after this one; ``checkpoint`` is ``None`` or ``{path}`` of
        the tree it kept in ``directory/tree``; ``error`` is ``None`` or the error of a job that
        raised, which counts as a job and keeps the state."""
        parameters = job.parameters
        request = request_from_parameters(parameters)
        data = replace(job.read_data(), request=request)
        # The row of a request's job names the request, so its request page settles on that row.
        named = {} if request is None else {"training_request": {"id": request.id, **request.to_dict()}}
        component = job_component(job)
        backend = self.backend_for(job.scenario, component)
        ended_jobs, state = self.state_after(job.previous_job_id or parameters.get("previous_job"))
        tree = HarnessTree.read(Path(job.start))
        # The tree's checksum names the head: a job needs no artifact ID of its start.
        head = ArtifactTextFiles(tree.checksum(), dict(tree.files))
        tasks = parameters.get("evaluation_tasks")
        progress = self.settings.evolution.component_state_dir(job.scenario, component) / PROGRESS_FILE
        try:
            with ProgressFile(backend, progress):
                prepared = backend.prepare_job(
                    data, state.to_json(), ended_jobs, head=head, seed_tasks=None if tasks is None else tuple(tasks)
                )
                if prepared.outcome == "skip":
                    return {
                        "steps": ended_jobs + 1,
                        "harness": dict(prepared.state),
                        "metrics": {**named, **prepared.metrics},
                        "checkpoint": None,
                        "error": None,
                    }
                candidate = prepared.candidate
                if not isinstance(candidate, UpdateCandidate):
                    raise TypeError("a harness job's candidate must be an UpdateCandidate")
                try:
                    result = backend.settle_job(prepared, self.build_policy(backend).evaluate_and_decide(candidate))
                except BaseException:
                    backend.abort_job(prepared)
                    raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            # The catalog row of a failed job reads as a skip with an error, which published nothing.
            reason = "step failed" if request is None else "request failed"
            return {
                "steps": ended_jobs + 1,
                "harness": state.to_json(),
                "metrics": {**named, **backend.failed_job_metrics(), "skipped": reason, "error": error},
                "checkpoint": None,
                "error": error,
            }
        if result.state is None:
            raise TypeError("a settled harness job carries the state after it")
        checkpoint = None
        if result.artifact is not None and result.artifact.local_path is not None:
            kept = directory / TREE_DIR
            shutil.rmtree(kept, ignore_errors=True)
            shutil.copytree(result.artifact.local_path, kept)
            backend.discard_kept_tree()
            checkpoint = {"path": str(kept)}
        return {
            "steps": ended_jobs + 1,
            "harness": dict(result.state),
            "metrics": {**named, **result.metrics},
            "checkpoint": checkpoint,
            "error": None,
        }

    def close_resources(self) -> None:
        for backend in self.backends.values():
            backend.close()
        self.backends.clear()


__all__ = [
    "HARNESS_COMPONENT",
    "HISTORY_DIR",
    "JOB_RECORDS_DIR",
    "PROGRESS_FILE",
    "HarnessTrainingComponent",
    "ProgressFile",
    "job_component",
    "read_job_progress",
]
