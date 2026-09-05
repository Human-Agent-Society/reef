"""Evolution worker scheduling, separate from episode sandbox isolation."""

from __future__ import annotations

import logging
from dataclasses import replace
from threading import Lock

from reef.runtime.executor import Executor, ExecutorConfig, WorkerSpec
from reef.runtime.executor.config import ExecutorSelection, ExecutorSettings, select_executor
from reef.runtime.executor.ray import RayExecutor
from reef.runtime.executor.requirements import ExecutionRequirements
from reef.train.cordis_backend.strategies import EpisodeScorer


class EvaluationWorkerPool:
    """One lazy, fixed-size worker group per backend; never replay failed work."""

    def __init__(self, selection: ExecutorSelection, requirements: ExecutionRequirements, worker: WorkerSpec):
        self._selection = selection
        self._requirements = requirements
        self._worker = worker
        self._executor: Executor | None = None
        self._lock = Lock()
        self._closed = False

    def evaluate(self, pairings):
        # Drain a batch before admitting the next, including ordinary scorer errors.
        with self._lock:
            if self._closed:
                raise RuntimeError("evolution worker pool is closed")
            if not pairings:
                return []
            if self._executor is None:
                config = evaluation_executor_config(
                    self._selection, self._requirements, self._worker, self._requirements.workers
                )
                try:
                    self._executor = Executor.create(config)
                except BaseException:
                    self._closed = True
                    raise
            pending = []
            try:
                for index, pairing in enumerate(pairings):
                    pending.append(
                        self._executor.rpc(index % self._requirements.workers, "run", args=pairing, non_block=True)
                    )
            except BaseException:
                # Partial submission cannot be replayed safely. Retire the pool.
                self._closed = True
                self._executor.shutdown()
                raise
            results = []
            first_error: Exception | None = None
            try:
                for future in pending:
                    try:
                        results.append(future.result())
                    except Exception as exc:  # noqa: PERF203 -- drain every submitted RPC before raising
                        if first_error is None:
                            first_error = exc
                if first_error is not None:
                    raise first_error
                return results
            except BaseException as exc:
                if not isinstance(exc, Exception):
                    self._closed = True
                    self._executor.shutdown()
                raise

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._executor is not None:
                self._executor.shutdown()


def evaluation_selection(
    scorer: EpisodeScorer,
    workers: int,
    settings: ExecutorSettings,
    gpus_per_worker: float | None = None,
) -> tuple[ExecutorSelection, ExecutionRequirements]:
    declared = scorer.execution_requirements()
    if not isinstance(declared, ExecutionRequirements):
        raise TypeError("EpisodeScorer.execution_requirements must return ExecutionRequirements")
    gpus = declared.gpus_per_worker if gpus_per_worker is None else gpus_per_worker
    requirements = replace(declared, workers=workers, gpus_per_worker=gpus)
    if requirements.gpus_per_worker < declared.gpus_per_worker:
        raise ValueError("evolution worker GPU allocation is below the scorer's declared requirement")
    configured_gpus = settings.options.get("num_gpus", gpus)
    if configured_gpus != gpus:
        raise ValueError(
            "declare evolution GPU needs through worker_resources.num_gpus, not executor.options.num_gpus"
        )
    runtime_env = settings.options.get("runtime_env", {})
    if not isinstance(runtime_env, dict):
        raise ValueError("evolution executor runtime_env must be an object")
    if "CUDA_VISIBLE_DEVICES" in runtime_env.get("env_vars", {}):
        raise ValueError("Ray owns evolution worker CUDA visibility; declare worker_resources.num_gpus")
    if settings.options.get("max_restarts", 0) != 0 or settings.options.get("max_task_retries", 0) != 0:
        raise ValueError("evolution executors must not replay evaluations")
    return select_executor(settings, role="evolution", requirements=requirements), requirements


def evaluation_executor_config(
    selection: ExecutorSelection, requirements: ExecutionRequirements, worker: WorkerSpec, count: int
) -> ExecutorConfig:
    backend = Executor.get_class(selection.settings.backend)
    options = dict(selection.settings.options)
    if issubclass(backend, RayExecutor):
        options = {"num_cpus": 1, **options, "num_gpus": requirements.gpus_per_worker}
    elif requirements.gpus_per_worker:
        options = {**options, "num_gpus": requirements.gpus_per_worker}
    logging.getLogger(__name__).info(
        "evolution: executor=%s workers=%s gpus_per_worker=%s (%s)",
        selection.settings.backend,
        count,
        requirements.gpus_per_worker,
        selection.reason,
    )
    return ExecutorConfig(backend=backend, options=options, workers=(worker,) * count)
