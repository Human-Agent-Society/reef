"""The harness training component: one harness evolution step per training job (RFC #1048, harness supplement).

A harness step runs in the training service, as a weight job does. The scenario's trainer submits a job through
its ``JobLoop``; a training worker runs it here, and control commits the selected tree as an artifact by the
``JobLoop`` rules. A job of this component holds:

- ``start``: the tree that the step builds on. ``base:<checksum>`` is the recipe's seed tree; any other start is a
  directory that holds a tree, which ``import_start`` or an earlier job's checkpoint wrote.
- ``data``: the step's material: the traces of traffic, the train tasks of an offline step, and the instruction.
- ``parameters``: ``component``, the scenario component that the step evolves; ``head_release_id``, the release
  of the start; ``evaluation_tasks``, the eval split of the scenario's imported tasks, or ``None`` for the
  configured tasks; ``previous_job``, the last job whose end the trainer committed, or ``None`` before the first.

A job's directory keeps ``outcome.json``: the settled step, with the steps that ended and the ``HarnessState``
after it; and ``tree/``, its selected tree. A job continues the state after ``previous_job``, so the harness state
follows the jobs that control committed: a job that ran but whose end no commit recorded changes no later step.
Under ``HarnessSteps.state_root(scenario, component)`` the component keeps the history and step records of the
backend, and ``progress.json`` while a step runs.
"""

from __future__ import annotations

import shutil
import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reef.artifact.artifact import Artifact
from reef.artifact.files import ArtifactFiles
from reef.artifact.repository import DefaultStart
from reef.core.evaluation import CandidateEvaluationPlugin, UpdateCandidate
from reef.harness.tree import HarnessTree
from reef.runtime.durable_files import read_json, write_json
from reef.runtime.jobs import JobStatus, ReleaseFormat, SavedCheckpoint, TrainingJob
from reef.surface.files import TextFileTree
from reef.train.backend import ReleaseFiles
from reef.train.harness_backend.backend import HarnessBackend, StepProgress
from reef.train.harness_backend.state import HarnessState
from reef.train.local_jobs import JOB_FILE, LocalJobComponent

#: The name of the component, which its jobs carry.
HARNESS_COMPONENT = "harness"
#: The start of a job from the recipe's seed tree: this prefix and the tree's checksum.
#: The backend's history and step records under the state directory of a scenario component.
HISTORY_DIR = "history"
STEPS_DIR = "steps"
PROGRESS_FILE = "progress.json"
#: The settled step of a job and the harness state after it: written before the job's checkpoint, so a resumed job
#: runs no second step.
OUTCOME_FILE = "outcome.json"
TREE_DIR = "tree"


class HarnessSteps(ABC):
    """What the harness training component reads from its recipe to run the steps of a scenario component.

    Values: ``seed`` is the tree of a new scenario. ``state_root(scenario, component)`` is the directory of the
    component's state. ``backend(scenario, component)`` makes the backend of its steps on the recipe's models; it
    reads no releases (each job names its head). ``policy(backend)`` is the selection policy that measures through
    ``backend``.
    ``check_tree(tree)`` raises ``ValueError`` for a tree that fails validation for the recipe's adapter.

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


def read_step_progress(path: Path) -> StepProgress | None:
    """The running step's progress that ``path`` holds, or ``None`` when no step runs."""
    value = read_json(path)
    return None if value is None else StepProgress.from_json(value)


class ProgressFile:
    """Writes the progress of ``backend``'s running step to ``path`` when it changed, looking every
    ``interval_seconds``, in a thread.

    Values: ``interval_seconds`` > 0. The thread runs from ``__enter__`` to ``__exit__``; ``__exit__`` removes the
    file. Purpose: ``HarnessTrainingComponent`` runs one around each step, so another process, such as the one that
    serves the request pages, reads where the step stands (``read_step_progress``).
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
            progress = self.backend.step_progress
            value = None if progress is None else progress.to_json()
            if value == written:
                continue
            if value is None:
                self.path.unlink(missing_ok=True)
            else:
                write_json(self.path, value)
            written = value


class HarnessTrainingComponent(LocalJobComponent):
    """The training component whose jobs are harness evolution steps, one step per job.

    Values: ``steps`` is the recipe's view (``HarnessSteps``). ``root`` holds ``jobs/``, one directory per job
    (``LocalJobComponent``), and ``starts/``, the trees that ``import_start`` took. A job's weights never exist:
    its releases are trees, so ``release_format`` is ``None``. A job depends only on its start, data and
    parameters, and on the state of its scenario component, which this component keeps per scenario; so scenarios
    share it. ``status`` of a job whose step raised carries the step's metrics and error.

    Purpose: ``HarnessEvolutionRecipe`` makes one, and names it in ``job_components``; the dispatcher queues its
    jobs, and the training service runs them.
    """

    def __init__(self, steps: HarnessSteps, root: Path) -> None:
        super().__init__(root / "jobs")
        self.steps = steps
        self.starts_root = root / "starts"
        #: Each scenario component's backend, made at its first job.
        self.backends: dict[tuple[str, str], HarnessBackend] = {}

    @property
    def name(self) -> str:
        return HARNESS_COMPONENT

    @property
    def release_format(self) -> ReleaseFormat | None:
        return None

    @property
    def shares_scenarios(self) -> bool:
        return True

    def default_start(self) -> DefaultStart:
        """The seed tree, as ``HarnessEvolutionRecipe.default_start`` gives it."""
        seed = self.steps.seed
        return DefaultStart(files=dict(seed.files), inputs={"seed": seed.checksum()})

    def check_task_jobs(self, parameters: Mapping[str, Any]) -> None:
        """Accept: an offline step runs the head tree on its train tasks."""
        return

    def check_release(self, files: ArtifactFiles) -> None:
        self.steps.check_tree(HarnessTree({path: files.read_bytes(path).decode("utf-8") for path in files.paths()}))

    def check_job(self, job: TrainingJob) -> None:
        parameters = job.parameters
        component = parameters.get("component")
        if not isinstance(component, str) or not component:
            raise ValueError("a harness job names the scenario component that it evolves in parameters.component")
        head = parameters.get("head_release_id")
        if not isinstance(head, str) or not head:
            raise ValueError("a harness job names the release of its start in parameters.head_release_id")
        tasks = parameters.get("evaluation_tasks")
        if tasks is not None and (not isinstance(tasks, list) or not all(isinstance(task, str) for task in tasks)):
            raise ValueError("parameters.evaluation_tasks of a harness job must be a list of task paths or null")
        previous = parameters.get("previous_job")
        if previous is not None and (not isinstance(previous, str) or not previous):
            raise ValueError("parameters.previous_job of a harness job must be a job ID or null")

    def import_start(self, release: Path) -> str:
        """Copy the tree of ``release`` to ``starts/<checksum>`` and return that directory; ``ValueError`` when the
        tree fails validation.

        The tree is the release's text files without the repository's own files (``TextFileTree``), as a client
        pulls it: a Git LFS release directory also holds ``.gitattributes``.
        """
        tree = HarnessTree(dict(TextFileTree().read_files(Artifact.local(release)) or {}))
        self.steps.check_tree(tree)
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
        # A job whose step raised failed with the step's metrics and error, which its outcome keeps.
        outcome = read_json(self.job_directory(job_id) / OUTCOME_FILE)
        if outcome is None or outcome["error"] is None:
            return status
        return JobStatus(job_id, "failed", status.checkpoints, metrics=outcome["metrics"], error=outcome["error"])

    def backend_for(self, scenario: str, component: str) -> HarnessBackend:
        """The backend of ``component``'s steps in ``scenario``: made at the first job, then kept."""
        key = (scenario, component)
        if key not in self.backends:
            self.backends[key] = self.steps.backend(scenario, component)
        return self.backends[key]

    def state_after(self, job_id: str | None) -> tuple[int, HarnessState]:
        """The steps that ended and the ``HarnessState`` after the job ``job_id``, as its outcome keeps them; the
        initial ``(0, HarnessState())`` for ``None``.

        A job that has no outcome, such as one cancelled before its step settled, changed nothing: its state is the
        state after the job that it names as ``previous_job``.
        """
        while job_id is not None:
            directory = self.job_directory(job_id)
            outcome = read_json(directory / OUTCOME_FILE)
            if outcome is not None:
                return int(outcome["steps"]), HarnessState.from_json(outcome["harness"])
            submitted = read_json(directory / JOB_FILE)
            job_id = None if submitted is None else submitted["parameters"].get("previous_job")
        return 0, HarnessState()

    def run(self, job: TrainingJob, directory: Path) -> Mapping[str, Any]:
        """Run the job's step from the state after ``parameters.previous_job``, then save its selected tree as the
        job's checkpoint.

        The step's outcome is durable before its checkpoint, so a resumed job runs no second step. The job's metrics
        are the step's metrics when the step kept no tree; a kept tree's checkpoint carries them instead. When the
        step raised, the job fails: ``status`` then reports the metrics of what the step reached, ``skipped``
        (``step failed``, or ``instruction failed`` for an instruction's step) and ``error``.
        """
        outcome = read_json(directory / OUTCOME_FILE)
        if outcome is None:
            outcome = self.run_step(job, directory)
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

    def run_step(self, job: TrainingJob, directory: Path) -> dict[str, Any]:
        """The settled step of ``job``: ``{steps, harness, metrics, checkpoint, error}``. ``steps`` and ``harness``
        are the steps that ended and the state after this one; ``checkpoint`` is ``None`` or ``{path}`` of
        the tree it kept in ``directory/tree``; ``error`` is ``None`` or the error of a step that
        raised, which counts as a step and keeps the state."""
        parameters = job.parameters
        request = job.data.request
        # The row of an instruction's step names the instruction, so its request page settles on that row.
        named = {} if request is None else {"training_request": {"id": request.id, **request.to_dict()}}
        component = str(parameters["component"])
        backend = self.backend_for(job.scenario, component)
        steps, state = self.state_after(parameters.get("previous_job"))
        head = ReleaseFiles(str(parameters["head_release_id"]), dict(HarnessTree.read(Path(job.start)).files))
        tasks = parameters.get("evaluation_tasks")
        progress = self.steps.state_root(job.scenario, component) / PROGRESS_FILE
        try:
            with ProgressFile(backend, progress):
                prepared = backend.prepare_step(
                    job.data, state.to_json(), steps, head=head, seed_tasks=None if tasks is None else tuple(tasks)
                )
                if prepared.outcome == "skip":
                    return {
                        "steps": steps + 1,
                        "harness": dict(prepared.state),
                        "metrics": {**named, **prepared.metrics},
                        "checkpoint": None,
                        "error": None,
                    }
                candidate = prepared.candidate
                if not isinstance(candidate, UpdateCandidate):
                    raise TypeError("a harness step's candidate must be an UpdateCandidate")
                try:
                    result = backend.settle_step(prepared, self.steps.policy(backend).evaluate_and_decide(candidate))
                except BaseException:
                    backend.abort_step(prepared)
                    raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            # The catalog row of a failed step reads as a skip with an error, which published nothing.
            reason = "step failed" if request is None else "instruction failed"
            return {
                "steps": steps + 1,
                "harness": state.to_json(),
                "metrics": {**named, **backend.failed_step_metrics(), "skipped": reason, "error": error},
                "checkpoint": None,
                "error": error,
            }
        if result.state is None:
            raise TypeError("a settled harness step carries the state after it")
        checkpoint = None
        if result.artifact is not None and result.artifact.local_path is not None:
            tree = directory / TREE_DIR
            shutil.rmtree(tree, ignore_errors=True)
            shutil.copytree(result.artifact.local_path, tree)
            backend.discard_kept_tree()
            checkpoint = {"path": str(tree)}
        return {
            "steps": steps + 1,
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
    "PROGRESS_FILE",
    "STEPS_DIR",
    "HarnessSteps",
    "HarnessTrainingComponent",
    "ProgressFile",
    "read_step_progress",
]
