"""Run and score harness rollouts in a pool of worker processes.

:class:`RolloutEvaluationWorker` runs one rollout of a rendered tree and
scores it with the deployment's :class:`RolloutScorer`.
:class:`EvaluationWorkerPool` runs those workers on the executor that
:func:`evaluation_selection` chooses from ``execution.evolution``. The pool
returns results in submission order and never runs failed work again.
"""

from __future__ import annotations

import json
import logging
import math
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from threading import Lock

from reef.core.evaluation import RolloutScore
from reef.evaluation.harness.scoring import FailureObservation, RolloutScorer, ScoredRollout
from reef.execution.sandbox.process import ROLLOUT_OWNER_LEASE, ProgramExecutor
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.rollouts.harbor_task import run_task_rollout
from reef.harness.rollouts.model_binding import ModelBindings
from reef.harness.rollouts.run import RolloutError, RolloutResult, run_rollout
from reef.harness.rollouts.trajectory import (
    TrajectoryError,
    agent_work,
    failed_trial_error,
    final_assistant_text,
    root_header,
    stage_path,
)
from reef.harness.tree.checks import clip_record_text
from reef.runtime.executor import Executor, ExecutorConfig, WorkerSpec
from reef.runtime.executor.config import (
    ExecutorSelection,
    ExecutorSettings,
    local_gpu_assignments,
    select_executor,
    worker_requirements,
)
from reef.runtime.executor.ray import RayExecutor
from reef.runtime.executor.requirements import ExecutionRequirements

#: The record file beside one evaluation rollout's kept trajectory.
RECORD_ROLLOUT_FILE = "rollout.json"


@dataclass(frozen=True)
class RolloutRun:
    """One rollout to run: a rendered tree, the task, and the directory that keeps its record.

    Values: ``files`` maps each rendered path to its text; ``task`` is the
    prompt or the task directory the rollout plays; ``record_dir`` is ``None``,
    which keeps no record, or an absent or empty directory that receives the
    rollout's trajectory files and ``rollout.json``.
    Purpose: the evaluator makes one per ``RolloutRequest`` it runs; the worker
    pool runs it.
    """

    files: Mapping[str, str]
    task: str
    record_dir: Path | None


@dataclass(repr=False)
class RolloutEvaluationWorker:
    """Run and score one rollout in a worker process.

    Values: the adapter ``descriptor``, the deployment's ``scorer``, the
    installed ``binary`` (``None`` for the adapter's own), the rollout
    ``timeout`` in seconds, the ``executor`` the rollout runs in, and
    ``forbid_residue``, which scores a littering rollout 0.
    ``owner_lease`` marks the rollout's processes as owned by this worker.
    ``transfer_records`` returns the record as an archive, for a worker that
    does not share the driver's file system. ``task_containers`` runs each
    task, a Harbor task directory, in the task's own container
    (``run_task_rollout``) instead of in ``executor``.
    Purpose: the worker pool builds one per worker; only rollout and scorer
    state crosses the process boundary, never the recipe.
    """

    descriptor: AdapterDescriptor
    scorer: RolloutScorer
    binary: str | None
    timeout: float
    executor: ProgramExecutor
    forbid_residue: bool
    owner_lease: bool = False
    transfer_records: bool = False
    task_containers: bool = False

    def __post_init__(self) -> None:
        self.executor.preflight()

    def run(self, rollout: RolloutRun, models: ModelBindings) -> ScoredRollout:
        """The scored rollout; a remote worker keeps the record on its side and returns it as an archive."""
        if not self.transfer_records or rollout.record_dir is None:
            return self.run_and_score(rollout.files, rollout.task, rollout.record_dir, models)
        # A remote worker must not read the driver's path as a local path.
        with tempfile.TemporaryDirectory(prefix="reef-worker-record-") as temporary:
            record_dir = Path(temporary) / "rollout"
            scored = self.run_and_score(rollout.files, rollout.task, record_dir, models)
            buffer = BytesIO()
            with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
                archive.add(record_dir, arcname=".")
            return replace(scored, record_archive=buffer.getvalue())

    def run_and_score(
        self, files: Mapping[str, str], task: str, record_dir: Path | None, models: ModelBindings
    ) -> ScoredRollout:
        """Score one rollout; a failed score marks a rollout that the environment could not run.

        The failure observation keeps the stage and the cause that the
        exception would otherwise discard. A native turn that ended on an
        error scores 0; any other nonzero exit still scores and is observed
        beside the score. ``record_dir`` receives the trajectory files before
        the rollout root is removed, unless it is ``None``; a copy that fails is
        not a rollout failure and propagates, so the step stops instead of
        scoring a result that a disk error shaped.
        """
        token = ROLLOUT_OWNER_LEASE.set(self.owner_lease)
        try:
            if self.task_containers:
                result = run_task_rollout(self.descriptor, files, task, timeout=self.timeout, keep_dir=record_dir)
            else:
                result = run_rollout(
                    self.descriptor,
                    files,
                    task,
                    binary=self.binary,
                    timeout=self.timeout,
                    executor=self.executor,
                    keep_dir=record_dir,
                )
        except RolloutError as error:
            scored = ScoredRollout(
                RolloutScore.failed(), FailureObservation(task=task, stage="launch", cause=str(error))
            )
            write_rollout_record(record_dir, task, None, scored)
            return scored
        except TrajectoryError as error:
            scored = ScoredRollout(
                RolloutScore.failed(), FailureObservation(task=task, stage="trajectory", cause=str(error))
            )
            write_rollout_record(record_dir, task, None, scored)
            return scored
        finally:
            ROLLOUT_OWNER_LEASE.reset(token)
        scored = self.score_result(result, task, models)
        write_rollout_record(record_dir, task, result, scored)
        return scored

    def score_result(self, result: RolloutResult, task: str, models: ModelBindings) -> ScoredRollout:
        """The score and the observations of a rollout that ran."""
        residue = len(result.residue)
        agents = agent_work(result.trajectory)
        path = stage_path(result.trajectory)
        if residue and self.forbid_residue:
            cause = f"{residue} file(s) outside the cleanup whitelist: {result.residue[0]}"
            return ScoredRollout(
                RolloutScore.scored(0.0),
                FailureObservation(task=task, stage="residue", cause=cause),
                residue,
                agents,
                path,
            )
        trial_error = failed_trial_error(result.trajectory)
        if trial_error:
            # The trial never ran (the image did not build, the agent could not
            # start): the environment did not run it, so it counts for neither side.
            return ScoredRollout(
                RolloutScore.failed(),
                FailureObservation(task=task, stage="trial", cause=trial_error),
                residue,
                agents,
                path,
            )
        if path.get("error") is not None:
            # The native loop ended its turn on an error (a tree that cannot load, a graph that cannot run, an
            # agent whose failure ended the run): nothing it wrote is an answer, and the tree caused it, so it
            # scores 0.
            error = path["error"]
            cause = f"{error.get('code', 'error')}: {error.get('message', '')}".strip(": ")
            if path.get("errored_agent"):
                cause = f"agent {path['errored_agent']}: {cause}"
            # A loop turn walks no graph: the failure names the loop when the root's header does.
            stage = "loop" if root_header(result.trajectory).get("loop") else "graph"
            return ScoredRollout(
                RolloutScore.scored(0.0),
                FailureObservation(task=task, stage=stage, cause=cause),
                residue,
                agents,
                path,
            )
        score = float(self.scorer.score_with_models(task, result, models))
        if not math.isfinite(score):
            raise ValueError(f"rollout scorer returned a non-finite score {score!r} for task {task!r}")
        reply = final_assistant_text(result.trajectory)
        if result.exit_code != 0:
            stderr_lines = result.stderr.strip().splitlines()
            cause = f"exit {result.exit_code}: {stderr_lines[-1] if stderr_lines else ''}".strip()
            return ScoredRollout(
                RolloutScore.scored(score),
                FailureObservation(task=task, stage="exit", cause=cause),
                residue,
                agents,
                path,
                reply,
            )
        # An empty trajectory is no failure: a grader that reads files scored the run as it stands. The record
        # still says no transcript was read, so a low score that a text grader gave is not blamed on the request.
        return ScoredRollout(
            RolloutScore.scored(score), None, residue, agents, path, reply, transcript_read=bool(result.trajectory)
        )


def write_rollout_record(
    record_dir: Path | None, task: str, result: RolloutResult | None, scored: ScoredRollout
) -> None:
    """``rollout.json`` beside the kept trajectory: what the scorer saw, so a result can be derived again.

    A rollout without a record directory keeps no record."""
    if record_dir is None:
        return
    record = {
        "task": clip_record_text(task),
        "score": scored.score.to_json(),
        "failure": None if scored.failure is None else scored.failure.to_dict(),
        "path": scored.path,
        "exit_code": None if result is None else result.exit_code,
        "stdout": None if result is None else clip_record_text(result.stdout),
        "stderr": None if result is None else clip_record_text(result.stderr),
        "residue": None if result is None else [clip_record_text(str(path)) for path in result.residue],
    }
    # A rollout that never wrote an event has no trajectory copy, so the directory may not exist yet.
    record_dir.mkdir(parents=True, exist_ok=True)
    with open(record_dir / RECORD_ROLLOUT_FILE, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(record, indent=2, default=str) + "\n")


class EvaluationWorkerPool:
    """One lazy, fixed-size group of rollout workers per scenario; it never runs failed work again.

    Values: ``selection`` and ``requirements`` choose the executor and its
    worker count; ``worker`` is the worker each slot runs. The executor starts
    on the first call. After a failure that a later call cannot recover from,
    the pool is closed and refuses work.
    Purpose: the evaluator owns one and runs every rollout through it.
    """

    def __init__(self, selection: ExecutorSelection, requirements: ExecutionRequirements, worker: WorkerSpec):
        self.selection = selection
        self.requirements = requirements
        self.worker = worker
        self.executor: Executor | None = None
        self.lock = Lock()
        self.closed = False

    def run(self, rollouts: Sequence[RolloutRun], models: ModelBindings) -> list[ScoredRollout]:
        """Run every rollout and return the scored rollouts in submission order.

        A batch drains before the next one starts, scorer errors included. The
        first error of a batch is raised after every rollout of the batch has
        finished.
        """
        with self.lock:
            if self.closed:
                if self.executor is not None and self.executor.failure is not None:
                    self.executor.check_health()
                raise RuntimeError("evolution worker pool is closed")
            if not rollouts:
                return []
            if self.executor is None:
                config = evaluation_executor_config(self.selection, self.requirements, self.worker)
                try:
                    self.executor = Executor.create(config)
                except BaseException:
                    self.closed = True
                    raise
            pending = []
            try:
                for index, rollout in enumerate(rollouts):
                    pending.append(
                        self.executor.rpc(
                            index % self.requirements.workers,
                            "run",
                            args=(rollout, models),
                            non_block=True,
                        )
                    )
            except BaseException:
                # A partial submission cannot be run again safely, so the pool retires.
                self.closed = True
                self.executor.shutdown()
                raise
            results: list[ScoredRollout] = []
            first_error: Exception | None = None
            try:
                for future in pending:
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        if first_error is None:
                            first_error = exc
                if first_error is not None:
                    raise first_error
                return results
            except BaseException as exc:
                if not isinstance(exc, Exception):
                    self.closed = True
                    self.executor.shutdown()
                raise

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True
            if self.executor is not None:
                self.executor.shutdown()


def evaluation_selection(
    scorer: RolloutScorer, settings: ExecutorSettings
) -> tuple[ExecutorSelection, ExecutionRequirements]:
    """The executor and the worker requirements of the rollout pool, from ``execution.evolution`` and the scorer."""
    declared = scorer.execution_requirements()
    if not isinstance(declared, ExecutionRequirements):
        raise TypeError("RolloutScorer.execution_requirements must return ExecutionRequirements")
    requirements = worker_requirements(settings, declared)
    gpus = requirements.gpus_per_worker
    configured_gpus = settings.options.get("num_gpus", gpus)
    if configured_gpus != gpus:
        raise ValueError(
            "declare GPU needs through execution.evolution.resources.gpus_per_worker, not executor.options.num_gpus"
        )
    configured_cpus = settings.options.get("num_cpus", requirements.cpus_per_worker)
    if settings.resources.cpus_per_worker is not None and configured_cpus != requirements.cpus_per_worker:
        raise ValueError("executor.options.num_cpus conflicts with execution.evolution.resources.cpus_per_worker")
    # Keep the older Ray profiles, which request CPUs through options.num_cpus.
    requirements = replace(requirements, cpus_per_worker=configured_cpus)
    runtime_env = settings.options.get("runtime_env", {})
    if not isinstance(runtime_env, dict):
        raise ValueError("evolution executor runtime_env must be an object")
    if "CUDA_VISIBLE_DEVICES" in runtime_env.get("env_vars", {}):
        raise ValueError(
            "Ray owns evolution worker CUDA visibility; declare execution.evolution.resources.gpus_per_worker"
        )
    if settings.options.get("max_restarts", 0) != 0 or settings.options.get("max_task_retries", 0) != 0:
        raise ValueError("evolution executors must not replay evaluations")
    return select_executor(settings, role="evolution", requirements=requirements), requirements


def evaluation_executor_config(
    selection: ExecutorSelection, requirements: ExecutionRequirements, worker: WorkerSpec
) -> ExecutorConfig:
    """The executor configuration of the rollout pool: one copy of ``worker`` per worker slot."""
    backend = Executor.get_class(selection.settings.backend)
    options = dict(selection.settings.options)
    if issubclass(backend, RayExecutor) or (
        selection.settings.backend not in ("uni", "mp")
        and (requirements.gpus_per_worker or selection.settings.resources.cpus_per_worker is not None)
    ):
        options = {**options, "num_cpus": requirements.cpus_per_worker, "num_gpus": requirements.gpus_per_worker}
    logging.getLogger(__name__).info(
        "evolution: executor=%s workers=%s gpus_per_worker=%s (%s)",
        selection.settings.backend,
        requirements.workers,
        requirements.gpus_per_worker,
        selection.reason,
    )
    workers = (worker,) * requirements.workers
    if selection.settings.backend == "mp" and requirements.gpus_per_worker:
        workers = tuple(
            replace(worker, options={"cuda_visible_devices": ",".join(devices)})
            for devices in local_gpu_assignments(requirements)
        )
    return ExecutorConfig(backend=backend, options=options, workers=workers)


__all__ = [
    "RECORD_ROLLOUT_FILE",
    "EvaluationWorkerPool",
    "RolloutEvaluationWorker",
    "RolloutRun",
    "evaluation_executor_config",
    "evaluation_selection",
    "write_rollout_record",
]
