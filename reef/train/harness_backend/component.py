"""The harness training component: each training job is one run of the harness method (RFC #1048, harness supplement).

A harness job runs in the training service, as a weight job does. The scenario's trainer submits a job through
its ``JobLoop``; a training worker runs it here, and control commits the selected tree as an artifact by the
``JobLoop`` rules. A job of this component holds:

- ``start``: the tree that the job builds on. ``base:<checksum>`` is the recipe's seed tree; any other start is a
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
Under ``HarnessJobs.state_root(scenario, component)`` the component keeps the history and job records of the
backend, and ``progress.json`` while a job runs.
"""

from __future__ import annotations

import shutil
import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from reef.artifact.artifact import Artifact
from reef.artifact.components import TextFileTree
from reef.artifact.files import ArtifactFiles
from reef.artifact.repository import DefaultStart
from reef.core.evaluation import CandidateEvaluationPlugin, UpdateCandidate
from reef.core.training_request import request_from_parameters
from reef.core.weight_layout import ArtifactFormat
from reef.harness.adapters.parts import TreeLayout
from reef.harness.tree import HarnessTree
from reef.storage.durable_files import read_json, write_json
from reef.storage.job_data import read_job_data
from reef.train.backend import ArtifactTextFiles
from reef.train.component import JobStatus, SavedCheckpoint, TrainingJob
from reef.train.harness_backend.backend import HarnessBackend, JobProgress
from reef.train.harness_backend.state import HarnessState
from reef.train.local_jobs import JOB_FILE, LocalJobComponent

#: The name of the component, which its jobs carry.
HARNESS_COMPONENT = "harness"
#: The start of a job from the recipe's seed tree: this prefix and the tree's checksum.
#: The backend's history and job records under the state directory of a scenario component.
HISTORY_DIR = "history"
JOB_RECORDS_DIR = "steps"
PROGRESS_FILE = "progress.json"
#: The outcome of a job and the harness state after it: written before the job's checkpoint, so a resumed job
#: does not run the method again.
OUTCOME_FILE = "outcome.json"
TREE_DIR = "tree"


class HarnessJobs(ABC):
    """What the harness training component reads from its recipe to run the jobs of a scenario component.

    Values: ``seed`` is the tree of a new scenario. ``state_root(scenario, component)`` is the directory of the
    component's state. ``backend(scenario, component)`` makes the backend of its jobs on the recipe's models; it
    reads no snapshots (each job names its head). ``policy(backend)`` is the selection policy that measures through
    ``backend``.
    ``check_tree(tree)`` raises ``ValueError`` for a tree that fails validation for the recipe's adapter.
    ``layout`` is the adapter's tree layout, which names the parts of a tree.

    Purpose: ``HarnessEvolutionRecipe`` makes one for its ``HarnessTrainingComponent``.
    """

    @property
    @abstractmethod
    def seed(self) -> HarnessTree: ...

    @abstractmethod
    def state_root(self, scenario: str, component: str) -> Path: ...

    @abstractmethod
    def backend(self, scenario: str, component: str) -> HarnessBackend: ...

    @abstractmethod
    def policy(self, backend: HarnessBackend) -> CandidateEvaluationPlugin: ...

    @abstractmethod
    def check_tree(self, tree: HarnessTree) -> None: ...

    @property
    @abstractmethod
    def layout(self) -> TreeLayout: ...


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

    Values: ``jobs`` is the recipe's view (``HarnessJobs``). ``root`` holds ``jobs/``, one directory per job
    (``LocalJobComponent``), and ``starts/``, the trees that ``import_start`` took. A job's weights never exist:
    its snapshots are trees, so ``artifact_format`` is ``None``. A job depends only on its start, data and
    parameters, and on the state of its scenario component, which this component keeps per scenario; so scenarios
    share it. ``status`` of a job that raised carries the job's metrics and error.

    Purpose: ``HarnessEvolutionRecipe`` makes one, and names it in ``job_components``; the dispatcher queues its
    jobs, and the training service runs them.
    """

    def __init__(self, jobs: HarnessJobs, root: Path) -> None:
        super().__init__(root / "jobs")
        self.jobs = jobs
        self.starts_root = root / "starts"
        #: Each scenario component's backend, made at its first job.
        self.backends: dict[tuple[str, str], HarnessBackend] = {}

    @property
    def name(self) -> str:
        return HARNESS_COMPONENT

    @property
    def artifact_format(self) -> ArtifactFormat | None:
        return None

    @property
    def shares_scenarios(self) -> bool:
        return True

    def default_start(self) -> DefaultStart:
        """The seed tree, as ``HarnessEvolutionRecipe.default_start`` gives it."""
        seed = self.jobs.seed
        return DefaultStart(files=dict(seed.files), inputs={"seed": seed.checksum()})

    def check_task_jobs(self, parameters: Mapping[str, Any]) -> None:
        """Accept: an offline job runs the head tree on its train tasks."""
        return

    def check_artifact(self, files: ArtifactFiles) -> None:
        self.jobs.check_tree(HarnessTree({path: files.read_bytes(path).decode("utf-8") for path in files.paths()}))

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
        return frozenset(self.jobs.layout.part_names(changes).values())

    def import_start(self, artifact_dir: Path) -> str:
        """Copy the tree of ``release`` to ``starts/<checksum>`` and return that directory; ``ValueError`` when the
        tree fails validation.

        The tree is the snapshot's text files without the repository's own files (``TextFileTree``), as a client
        pulls it: a Git LFS release directory also holds ``.gitattributes``.
        """
        tree = HarnessTree(dict(TextFileTree().read_files(Artifact.local(artifact_dir)) or {}))
        self.jobs.check_tree(tree)
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
            self.backends[key] = self.jobs.backend(scenario, component)
        return self.backends[key]

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
        data = replace(read_job_data(job.data_dir, job.data_id), request=request)
        # The row of a request's job names the request, so its request page settles on that row.
        named = {} if request is None else {"training_request": {"id": request.id, **request.to_dict()}}
        component = job_component(job)
        backend = self.backend_for(job.scenario, component)
        ended_jobs, state = self.state_after(job.previous_job_id or parameters.get("previous_job"))
        tree = HarnessTree.read(Path(job.start))
        # The tree's checksum names the head: a job needs no artifact ID of its start.
        head = ArtifactTextFiles(tree.checksum(), dict(tree.files))
        tasks = parameters.get("evaluation_tasks")
        progress = self.jobs.state_root(job.scenario, component) / PROGRESS_FILE
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
                    result = backend.settle_job(prepared, self.jobs.policy(backend).evaluate_and_decide(candidate))
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
    "HarnessJobs",
    "HarnessTrainingComponent",
    "ProgressFile",
    "job_component",
    "read_job_progress",
]
