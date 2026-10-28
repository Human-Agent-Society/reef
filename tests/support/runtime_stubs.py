"""Test doubles for Reef's training component and inference runtime contracts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reef.artifact import Artifact
from reef.inference.http import InferenceProxyRuntime
from reef.runtime.jobs import JobStatus, ReleaseFormat, SavedCheckpoint, TrainingComponent, TrainingJob
from reef.runtime.serving import holds_weights
from reef.train.pipeline import PipelineConfig, PipelineSpec

#: The release format of a stub component that does not name one.
FULL_WEIGHTS = ReleaseFormat("full")
#: The file a stub checkpoint holds: the version name that serving the checkpoint reports.
VERSION_FILE = "model.txt"


class StubTrainingComponent(TrainingComponent):
    """A training component that runs each job in ``submit``, in this thread.

    ``name`` is the backend name of its jobs. Without ``checkpoint_root`` a submitted job
    fails, so a test that only builds a recipe never trains. With it, a job
    saves the checkpoints that :meth:`train` returns; the default saves one
    checkpoint, ``<checkpoint_root>/<n>``, that holds ``model.txt`` with the
    version name ``w<n>`` for the ``n``-th job. ``submitted`` lists the jobs
    in submission order and ``imported`` the releases given to
    :meth:`import_start`. ``shares_scenarios`` is what the component reports
    for :attr:`TrainingComponent.shares_scenarios`, and ``release_format`` for
    :attr:`TrainingComponent.release_format` (full weights by default). Like
    the Tinker component, it runs task jobs that carry a ``rollout`` object.
    """

    def __init__(
        self,
        checkpoint_root: Path | None = None,
        *,
        name: str = "stub",
        shares_scenarios: bool = False,
        release_format: ReleaseFormat | None = FULL_WEIGHTS,
    ) -> None:
        self.checkpoint_root = checkpoint_root
        self.backend_name = name
        self.scenarios_shared = shares_scenarios
        self.releases = release_format
        self.submitted: list[TrainingJob] = []
        self.imported: list[Path] = []
        self.statuses: dict[str, JobStatus] = {}
        self.closed = False

    @property
    def name(self) -> str:
        return self.backend_name

    @property
    def shares_scenarios(self) -> bool:
        return self.scenarios_shared

    @property
    def release_format(self) -> ReleaseFormat | None:
        return self.releases

    def check_task_jobs(self, parameters: Mapping[str, Any]) -> None:
        if not isinstance(parameters.get("rollout"), Mapping):
            raise ValueError("a task job needs the rollout parameter as an object")

    def import_start(self, release: Path) -> str | None:
        self.imported.append(release)
        return str(release) if holds_weights(release) else None

    def train(self, job: TrainingJob) -> tuple[SavedCheckpoint, ...]:
        """The checkpoints that ``job`` saves; raising fails the job."""
        if self.checkpoint_root is None:
            raise NotImplementedError("StubTrainingComponent has no checkpoint root")
        index = len(self.submitted)
        checkpoint = self.checkpoint_root / str(index)
        checkpoint.mkdir(parents=True)
        (checkpoint / VERSION_FILE).write_text(f"w{index}")
        return (SavedCheckpoint(0, str(checkpoint), str(checkpoint)),)

    def submit(self, job: TrainingJob) -> None:
        if job.job_id in self.statuses:
            return
        self.submitted.append(job)
        try:
            self.statuses[job.job_id] = JobStatus(job.job_id, "finished", self.train(job))
        except Exception as exc:
            self.statuses[job.job_id] = JobStatus(job.job_id, "failed", error=f"{type(exc).__name__}: {exc}")

    def status(self, job_id: str) -> JobStatus:
        return self.statuses[job_id]

    def cancel(self, job_id: str) -> None:
        status = self.statuses.get(job_id)
        if status is not None and status.state == "running":
            self.statuses[job_id] = JobStatus(job_id, "failed", status.checkpoints, error="cancelled")

    def close(self) -> None:
        self.closed = True


class StubInferenceRuntime(InferenceProxyRuntime):
    """An inference proxy that records the releases it serves.

    A release serves under the version its ``model.txt`` names, or under its
    release ID when it holds no such file. ``activated`` lists the versions
    of the releases it served, at commits, activations and starts, in order;
    ``served`` is the last one, ``None`` before the first one.
    """

    def __init__(self, base_url: str = "http://inference-runtime", **kwargs) -> None:
        super().__init__(base_url=base_url, **kwargs)
        self.activated: list[str] = []
        self.served: str | None = None

    def current_runtime_load_id(self) -> str | None:
        return self.served

    def activate_checkpoint(self, artifact: Artifact, *, scenario: str | None = None) -> str:
        version = self.serve(artifact)
        self.activated.append(version)
        return version

    def serve(self, artifact: Artifact) -> str:
        path = artifact.local_path
        marker = None if path is None else Path(path) / VERSION_FILE
        version = marker.read_text() if marker is not None and marker.is_file() else artifact.ref.release_id
        self.served = version
        return version


def runtime_bindings(
    component: TrainingComponent | None = None, runtime: StubInferenceRuntime | None = None
) -> dict[str, object]:
    """The recipe arguments for a training component and its inference runtime; stubs fill what is missing."""
    return {
        "training_component": StubTrainingComponent() if component is None else component,
        "runtime": StubInferenceRuntime() if runtime is None else runtime,
    }


def job_sized_stages(
    spec: PipelineSpec, section: Mapping[str, Any] | None = None, *, job_samples: int = 1
) -> PipelineConfig:
    """The stages of ``spec`` configured by a deployment ``pipeline`` section, with ``job_samples`` set.

    A weight recipe needs ``pipeline.sample2job.config.job_samples``; a ``job_samples`` in ``section`` wins.
    """
    stages = {key: dict(value) for key, value in (section or {}).items()}
    sample2job = stages.setdefault("sample2job", {})
    sample2job["config"] = {"job_samples": job_samples, **sample2job.get("config", {})}
    return PipelineConfig.from_setting(stages, spec=spec, environ={})
