"""A ``TrainingComponent`` as a ``TrainingBackend``: the backend that runs Tinker, and the other components that
continue from a start directory, behind the backend server (RFC #1146).

A job's artifact maps to the component's start:

- ``state/`` is the component's own start directory, which a checkpoint of this backend wrote;
- else ``release/``, or the artifact itself when it holds no ``release/``, is a release that ``import_start`` takes;
- the artifact URI ``base:`` is the component's ``base_start``, such as a fresh model of its configured base.

Each checkpoint that the component saves becomes an artifact directory: its ``path`` (the start that a later job
continues from) as ``state/``, and its ``release_path`` as ``release/``.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Self

from reef.storage.job_data import JOB_FILE, JobDataRef
from reef.train.component import SavedCheckpoint, TrainingComponent, TrainingJob, build_training_component
from reef.train.remote.contract import RELEASE_DIR, STATE_DIR, BackendInfo, RemoteJob
from reef.train.remote.server import JobRun, TrainingBackend

#: The artifact URI of a component's own default start.
BASE_URI = "base:"
#: Seconds between two reads of a running job's status.
POLL_SECONDS = 1.0


class ComponentBackend(TrainingBackend):
    """Run the jobs of ``component``, one ``TrainingComponent``, from artifact directories.

    Values: ``component`` is a started component (``start_jobs``). ``poll_seconds`` is the wait between two reads of
    a running job.
    """

    def __init__(self, component: TrainingComponent, *, poll_seconds: float = POLL_SECONDS) -> None:
        self.component = component
        self.poll_seconds = poll_seconds
        component.start_jobs()

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Self:
        """``{component, model_path}``: the component's config, whose ``type`` names its factory, as a training
        deployment gives it, and the local model directory that the component reads."""
        component = build_training_component(
            config["component"], model_path=str(config.get("model_path", "")), environ=os.environ
        )
        return cls(component)

    @property
    def info(self) -> BackendInfo:
        return BackendInfo(self.component.name, self.component.release_format, self.component.shares_scenarios)

    def start(self, job: RemoteJob, artifact: Path | None) -> str:
        """The component's start of ``job``'s artifact."""
        if job.artifact_uri == BASE_URI:
            base_start = getattr(self.component, "base_start", None)
            if not isinstance(base_start, str):
                raise ValueError(f"the {self.component.name} component has no base start")
            return base_start
        if artifact is None:
            raise ValueError(f"the backend cannot read the artifact {job.artifact_uri!r}")
        if (artifact / STATE_DIR).is_dir():
            return str(artifact / STATE_DIR)
        release = artifact / RELEASE_DIR if (artifact / RELEASE_DIR).is_dir() else artifact
        return self.component.import_start(release)

    def run(self, job: RemoteJob, artifact: Path | None, data: Path, run: JobRun) -> Mapping[str, Any]:
        return run_on_component(
            self.component, self.start(job, artifact), job, data, run, poll_seconds=self.poll_seconds
        )


def run_on_component(
    component: TrainingComponent,
    start: str,
    job: RemoteJob,
    data: Path,
    run: JobRun,
    *,
    add_state: Callable[[SavedCheckpoint, Path], None] | None = None,
    poll_seconds: float = POLL_SECONDS,
) -> Mapping[str, Any]:
    """Run ``job`` on ``component`` from ``start`` to its end, and save each of its checkpoints as an artifact:
    the checkpoint's ``path`` as ``state/`` and its ``release_path`` as ``release/``. ``add_state``, when given,
    adds more to ``state/`` of each checkpoint, such as a framework's own checkpoint. A cancel of ``run`` cancels the
    component's job. Return the job's metrics; ``RuntimeError`` with its error when it failed."""
    header = json.loads((data / JOB_FILE).read_text(encoding="utf-8"))
    ref = JobDataRef(data.as_uri(), data, job.data["data_id"], header["sample_count"], header["kind"])
    component.submit(TrainingJob(job.job_id, component.name, start, ref, job.parameters, scenario=job.scenario))
    saved = 0
    while True:
        if run.cancelled.is_set():
            component.cancel(job.job_id)
        status = component.status(job.job_id)
        for checkpoint in status.checkpoints[saved:]:
            with tempfile.TemporaryDirectory() as staging:
                directory = Path(staging)
                shutil.copytree(checkpoint.release_path, directory / RELEASE_DIR)
                shutil.copytree(checkpoint.path, directory / STATE_DIR)
                if add_state is not None:
                    add_state(checkpoint, directory / STATE_DIR)
                run.save(checkpoint.step, directory, checkpoint.metrics)
            saved += 1
        if status.state == "failed":
            raise RuntimeError(status.error)
        if status.state == "finished":
            return status.metrics
        time.sleep(poll_seconds)


__all__ = ["BASE_URI", "ComponentBackend", "run_on_component"]
