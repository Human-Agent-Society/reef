"""Evolution worker scheduling, separate from episode sandbox isolation."""

from __future__ import annotations

import logging
from dataclasses import replace
from threading import Lock

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
                if self._executor is not None and self._executor.failure is not None:
                    self._executor.check_health()
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


def legacy_worker_settings(
    settings: ExecutorSettings, workers: int | None = None, gpus: float | None = None
) -> ExecutorSettings:
    """Compatibility boundary for the old recipe/Python resource arguments."""
    if workers is not None:
        if settings.workers is not None and settings.workers != workers:
            raise ValueError("episode_workers conflicts with execution.evolution.workers")
        settings = replace(settings, workers=workers)
    if gpus is not None:
        if settings.resources.gpus_per_worker is not None and settings.resources.gpus_per_worker != gpus:
            raise ValueError("worker_resources.num_gpus conflicts with execution.evolution.resources.gpus_per_worker")
        settings = replace(settings, resources=replace(settings.resources, gpus_per_worker=gpus))
    return settings


def evaluation_selection(
    scorer: EpisodeScorer,
    workers: int | None,
    settings: ExecutorSettings,
    gpus_per_worker: float | None = None,
) -> tuple[ExecutorSelection, ExecutionRequirements]:
    declared = scorer.execution_requirements()
    if not isinstance(declared, ExecutionRequirements):
        raise TypeError("EpisodeScorer.execution_requirements must return ExecutionRequirements")
    settings = legacy_worker_settings(settings, workers, gpus_per_worker)
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
    # Preserve old Ray profiles which request CPUs through options.num_cpus.
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
    selection: ExecutorSelection, requirements: ExecutionRequirements, worker: WorkerSpec, count: int
) -> ExecutorConfig:
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
        count,
        requirements.gpus_per_worker,
        selection.reason,
    )
    workers = (worker,) * count
    if selection.settings.backend == "mp" and requirements.gpus_per_worker:
        workers = tuple(
            replace(worker, options={"cuda_visible_devices": ",".join(devices)})
            for devices in local_gpu_assignments(requirements)
        )
    return ExecutorConfig(backend=backend, options=options, workers=workers)
