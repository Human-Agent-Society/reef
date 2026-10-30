"""Reads of a scenario that the data service serves: the files of an artifact, and the training jobs in the queue.

No aiohttp types appear here. ``reef.service.routes.scenario_files`` adapts it to HTTP.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reef.artifact.artifact import ArtifactNotFound
from reef.artifact.files import DirectoryFiles
from reef.core.errors import UnknownScenario
from reef.dispatcher import Dispatcher
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


def artifact_files(dispatcher: Dispatcher, scenario: str, artifact_id: str, component: str | None) -> dict[str, Any]:
    """``{scenario, artifact_id, files}``: each file of the artifact as ``{path, bytes}``, in path order."""
    directory = artifact_directory(dispatcher, scenario, artifact_id, component)
    files = DirectoryFiles(directory)
    return {
        "scenario": scenario,
        "artifact_id": artifact_id,
        "files": [{"path": path, "bytes": files.file(path).stat().st_size} for path in files.paths()],
    }


def artifact_file(dispatcher: Dispatcher, scenario: str, artifact_id: str, component: str | None, path: str) -> Path:
    """The file at ``path`` of the artifact; ``ArtifactNotFound`` unless ``path`` is one of its files."""
    files = DirectoryFiles(artifact_directory(dispatcher, scenario, artifact_id, component))
    if path not in files.paths():
        raise ArtifactNotFound(f"artifact {artifact_id!r} of {scenario!r} holds no file {path!r}")
    return files.file(path)


def scenario_jobs(dispatcher: Dispatcher, scenario: str) -> dict[str, Any]:
    """``{scenario, jobs}``: the training jobs of ``scenario`` in the jobs queue, oldest first, each as
    ``{id, state, attempt_count, checkpoints, result, error}``; ``checkpoints`` are the saved checkpoints that the
    job reported. Raise ``UnknownScenario`` for a scenario that does not exist."""
    if not dispatcher.has_scenario(scenario):
        raise UnknownScenario(f"unknown scenario {scenario!r}")
    return {
        "scenario": scenario,
        "jobs": [
            {
                "id": item.id,
                "state": item.state,
                "attempt_count": item.attempt_count,
                "checkpoints": [dict(progress) for progress in item.progress],
                "result": None if item.result is None else dict(item.result),
                "error": item.error,
            }
            for item in dispatcher.queue.items(JOBS_QUEUE, scenario=scenario)
        ],
    }


__all__ = ["artifact_directory", "artifact_file", "artifact_files", "scenario_jobs"]
