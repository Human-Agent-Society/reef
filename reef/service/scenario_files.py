"""A scenario's artifact files and training jobs: the reads that the data service serves, and the cancel of a job,
which control serves.

No aiohttp types appear here. ``reef.service.routes.scenario_files`` and ``reef.service.routes.scenarios`` adapt it
to HTTP.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reef.artifact.artifact import ArtifactNotFound
from reef.artifact.files import DirectoryFiles
from reef.core.errors import UnknownScenario
from reef.dispatcher import Dispatcher
from reef.service.auth import MAX_DOWNLOAD_URL_SECONDS
from reef.storage.work_queue import WorkItem
from reef.train.job_queue import JOBS_QUEUE


def artifact_directory(dispatcher: Dispatcher, scenario: str, artifact_id: str, component: str | None) -> Path:
    """The local directory of the files of ``artifact_id``, ``component``'s content in ``scenario``.

    Raise as ``Scenario.resolve_artifact`` does: ``ValueError`` for a missing or unknown component and
    ``ArtifactNotFound`` for an artifact that the scenario does not list.
    """
    with dispatcher.locked_scenario(scenario) as current:
        selected, release = current.resolve_artifact(artifact_id, component)
        view = current.surface.component_artifact(release.materialize(), selected.component).materialize()
    if view.local_path is None:
        raise ValueError(f"artifact {artifact_id!r} of {scenario!r} has no local files")
    return view.local_path


def artifact_files(
    dispatcher: Dispatcher, scenario: str, artifact_id: str, component: str | None, expires_in_seconds: int
) -> dict[str, Any]:
    """``{scenario, artifact_id, files}``: each file of the artifact as ``{path, bytes, url, url_expires_at}``, in
    path order.

    When the repository backend signs URLs (``RepositoryBackend.file_downloads``), ``url`` opens the file in its
    storage until ``url_expires_at``, ``expires_in_seconds`` from now, and the release is not materialized. Otherwise
    both are null, and the files are those of the materialized release. Raise ``ValueError`` for an
    ``expires_in_seconds`` outside 1 to ``MAX_DOWNLOAD_URL_SECONDS``, and as ``Scenario.resolve_artifact`` does.
    """
    if not 1 <= expires_in_seconds <= MAX_DOWNLOAD_URL_SECONDS:
        raise ValueError(f"expires_in must be 1 to {MAX_DOWNLOAD_URL_SECONDS} seconds")
    with dispatcher.locked_scenario(scenario) as current:
        selected, ref = current.artifact_release_ref(artifact_id, component)
        downloads = current.repository.file_downloads(ref, expires_in_seconds)
        # A release of several components keeps each one in a directory named after it.
        prefix = "" if current.surface.single else f"{selected}/"
    if downloads is not None:
        files = [
            {
                "path": download.path.removeprefix(prefix),
                "bytes": download.byte_count,
                "url": download.url,
                "url_expires_at": download.expires_at,
            }
            for download in downloads
            if download.path.startswith(prefix)
        ]
    else:
        local = DirectoryFiles(artifact_directory(dispatcher, scenario, artifact_id, component))
        files = [
            {"path": path, "bytes": local.file(path).stat().st_size, "url": None, "url_expires_at": None}
            for path in local.paths()
        ]
    return {"scenario": scenario, "artifact_id": artifact_id, "files": files}


def artifact_file(dispatcher: Dispatcher, scenario: str, artifact_id: str, component: str | None, path: str) -> Path:
    """The file at ``path`` of the artifact; ``ArtifactNotFound`` unless ``path`` is one of its files."""
    files = DirectoryFiles(artifact_directory(dispatcher, scenario, artifact_id, component))
    if path not in files.paths():
        raise ArtifactNotFound(f"artifact {artifact_id!r} of {scenario!r} holds no file {path!r}")
    return files.file(path)


#: The error of a job that a client cancelled.
JOB_CANCELLED_ERROR = "cancelled by request"


def job_json(item: WorkItem) -> dict[str, Any]:
    """The jobs-queue ``item`` as ``{id, state, attempt_count, checkpoints, result, error}``; ``checkpoints`` are the
    saved checkpoints that the job reported."""
    return {
        "id": item.id,
        "state": item.state,
        "attempt_count": item.attempt_count,
        "checkpoints": [dict(progress) for progress in item.progress],
        "result": None if item.result is None else dict(item.result),
        "error": item.error,
    }


def scenario_jobs(dispatcher: Dispatcher, scenario: str) -> dict[str, Any]:
    """``{scenario, jobs}``: the training jobs of ``scenario`` in the jobs queue, oldest first, each as ``job_json``
    gives it. Raise ``UnknownScenario`` for a scenario that does not exist."""
    if not dispatcher.has_scenario(scenario):
        raise UnknownScenario(f"unknown scenario {scenario!r}")
    return {
        "scenario": scenario,
        "jobs": [job_json(item) for item in dispatcher.queue.items(JOBS_QUEUE, scenario=scenario)],
    }


def cancel_scenario_job(dispatcher: Dispatcher, scenario: str, job_id: str) -> dict[str, Any] | None:
    """Cancel the training job ``job_id`` of ``scenario``; return the job, as ``job_json`` gives it, after the cancel.

    A job that has not ended fails with ``JOB_CANCELLED_ERROR``: a queued job at once, and a running job's training
    worker cancels the run at its next poll. The trainer reads the failed job and records its end. A job that ended
    stays as it is. Return None when ``scenario`` has no job ``job_id``; raise ``UnknownScenario`` for a scenario that
    does not exist.
    """
    if not dispatcher.has_scenario(scenario):
        raise UnknownScenario(f"unknown scenario {scenario!r}")
    item = dispatcher.queue.get(JOBS_QUEUE, job_id)
    if item is None or item.scenario != scenario:
        return None
    cancelled = dispatcher.queue.cancel(JOBS_QUEUE, job_id, JOB_CANCELLED_ERROR)
    return job_json(item if cancelled is None else cancelled)


__all__ = [
    "JOB_CANCELLED_ERROR",
    "artifact_directory",
    "artifact_file",
    "artifact_files",
    "cancel_scenario_job",
    "job_json",
    "scenario_jobs",
]
