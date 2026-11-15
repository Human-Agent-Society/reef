"""Let a training component of a test run its jobs on the trainer's own thread, with no jobs queue.

Control commits a checkpoint by the name of its stored content, which the training worker stores
(``TrainingJobWorker.run_item``). A test component that a ``JobLoop`` calls directly has no worker, so
``store_checkpoints`` makes the component do the worker's part.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from reef.artifact.repository import Repository, RepositoryBackend
from reef.train.component import JobStatus, SavedCheckpoint, ScenarioRoot, TrainingComponent, TrainingJob
from reef.train.job_queue import checkpoint_content


def store_checkpoints(component: TrainingComponent, scenario: str, root: ScenarioRoot | None) -> TrainingComponent:
    """Return ``component`` after its ``status`` was made to report every checkpoint as the training worker
    records it: named by ``checkpoint_content``, with the parts that the checkpoint changed, and with its files
    stored in the artifact store of the job's scenario.

    ``root`` is the root of ``scenario``, whose repository is that store. A root with no repository, or None,
    stores nothing: the trainer of such a test commits to no scenario. Calling this again for another scenario adds
    that scenario's store.
    """
    repository = None if root is None else root.artifact.repository
    stores: dict[str, RepositoryBackend | None] = component.__dict__.setdefault("checkpoint_stores", {})
    patched = bool(stores)
    stores[scenario] = repository.backend if isinstance(repository, Repository) else None
    if patched:
        return component
    jobs: dict[str, TrainingJob] = {}
    stored: dict[tuple[str, int], SavedCheckpoint] = {}
    submit, status, changed_parts = component.submit, component.status, component.changed_parts

    def submit_and_remember(job: TrainingJob) -> None:
        jobs[job.job_id] = job
        submit(job)

    def stored_status(job_id: str) -> JobStatus:
        reported = status(job_id)
        job = jobs.get(job_id)
        if job is None or job.scenario not in stores:
            return reported
        checkpoints = []
        for checkpoint in reported.checkpoints:
            key = (job_id, checkpoint.step)
            if key not in stored and checkpoint.artifact_path is not None:
                content = checkpoint_content(job_id, checkpoint.step)
                store = stores[job.scenario]
                if store is not None:
                    store.store_content(content, Path(checkpoint.artifact_path))
                stored[key] = replace(
                    checkpoint,
                    artifact_path=None,
                    content=content,
                    changed_parts=changed_parts(job.start, checkpoint),
                )
            checkpoints.append(stored.get(key, checkpoint))
        return replace(reported, checkpoints=tuple(checkpoints))

    def recorded_changed_parts(start: str, checkpoint: SavedCheckpoint) -> frozenset[str]:
        return checkpoint.changed_parts

    component.submit = submit_and_remember  # type: ignore[method-assign]
    component.status = stored_status  # type: ignore[method-assign]
    component.changed_parts = recorded_changed_parts  # type: ignore[method-assign]
    return component
