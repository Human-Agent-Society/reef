"""Test doubles for Reef's training component and inference runtime contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reef.artifact import Artifact, DefaultStart
from reef.core.errors import ReefError
from reef.core.evaluation import PolicyModel
from reef.core.job_data import Sample
from reef.core.served_models import ServedArtifact
from reef.core.snapshot import RootSource
from reef.core.weight_layout import ArtifactFormat
from reef.inference.http import HttpInferenceHandler
from reef.inference.interfaces import InferenceBackend, InferenceHandler, InferenceRuntime
from reef.inference.serving import OpenedEngines, ServingEngines
from reef.scenario.live_objects import LiveObjects
from reef.storage.data import DataStore
from reef.storage.job_data import JobDataKind
from reef.train.component import (
    JobStatus,
    SavedCheckpoint,
    ScenarioRoot,
    TrainingBackendFacts,
    TrainingComponent,
    TrainingJob,
)
from reef.train.pipeline import PipelineSettings, PipelineSpec

#: The artifact format of a stub component that does not name one.
FULL_WEIGHTS = ArtifactFormat("full")
#: The file a stub checkpoint holds: the version name that serving the checkpoint reports.
VERSION_FILE = "model.txt"
#: The root snapshot of a trainer that a test builds outside a scenario; a stub's ``root_start`` does not read it.
STUB_ROOT = ScenarioRoot(Artifact.local(Path("stub-root")), RootSource("default"))


class StubTrainingComponent(TrainingComponent):
    """A training component that runs each job in ``submit``, in this thread.

    ``name`` is the backend name of its jobs. Without ``checkpoint_root`` a submitted job
    fails, so a test that only builds a recipe never trains. With it, a job
    saves the checkpoints that :meth:`train` returns; the default saves one
    checkpoint, ``<checkpoint_root>/<n>``, that holds ``model.txt`` with the
    version name ``w<n>`` for the ``n``-th job. ``submitted`` lists the jobs
    in submission order and ``imported`` the snapshots given to
    :meth:`import_start`. ``shares_scenarios`` is what the component reports
    for :attr:`TrainingComponent.shares_scenarios`, and ``artifact_format`` for
    :attr:`TrainingComponent.artifact_format` (full weights by default). Like
    the Tinker component, it runs task jobs that carry a ``rollout`` object.
    """

    def __init__(
        self,
        checkpoint_root: Path | None = None,
        *,
        name: str = "stub",
        shares_scenarios: bool = False,
        artifact_format: ArtifactFormat | None = FULL_WEIGHTS,
    ) -> None:
        self.checkpoint_root = checkpoint_root
        self.backend_name = name
        self.scenarios_shared = shares_scenarios
        self.snapshot_format = artifact_format
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
    def artifact_format(self) -> ArtifactFormat | None:
        return self.snapshot_format

    def check_task_jobs(self, parameters: Mapping[str, Any]) -> None:
        if not isinstance(parameters.get("rollout"), Mapping):
            raise ValueError("a task job needs the rollout parameter as an object")

    def stub_default_start(self) -> DefaultStart:
        """The default start that the stub's facts write (``StubFacts``): an empty root, since a stub checkpoint is
        any directory. A subclass overrides it to start its scenarios from other files."""
        return DefaultStart(inputs={"model": self.backend_name})

    @property
    def base_start(self) -> str:
        return f"{self.backend_name}:base"

    def root_start(self, root: ScenarioRoot) -> str:
        """Every scenario root starts a stub's jobs at ``base_start``; a train-from goes through ``import_start``."""
        return self.base_start

    def check_import_start(self, artifact_dir: Path) -> None:
        """Raise ``ValueError`` for a snapshot that the stub cannot start from; the stub's facts ask it too. A stub
        starts from every snapshot."""
        return

    def import_start(self, artifact_dir: Path) -> str:
        self.check_import_start(artifact_dir)
        self.imported.append(artifact_dir)
        return str(artifact_dir)

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


@dataclass(frozen=True)
class RecordedJob:
    """What a component got for one job: the data of its job data directory, and the ID of its training request."""

    data_id: str
    samples: tuple[Sample, ...]
    request_id: str | None


class RecordingComponent(StubTrainingComponent):
    """A training component that keeps what it got for every job it ran, in ``batches``, and saves no checkpoint, so
    each job ends at once in ``submit``.

    Purpose: a test of how a trainer reads, batches and commits records, which trains nothing worth publishing.
    """

    def __init__(self, batches: list[RecordedJob] | None = None, *, name: str = "recording") -> None:
        super().__init__(name=name, shares_scenarios=True)
        self.batches: list[RecordedJob] = [] if batches is None else batches

    def train(self, job: TrainingJob) -> tuple[SavedCheckpoint, ...]:
        data = job.read_data()
        self.batches.append(RecordedJob(data.data_id, data.samples, job.request_id))
        return ()


class StubInferenceRuntime(InferenceRuntime):
    """A runtime that serves weight snapshots without engines: it records each snapshot that a test activates.

    Values: ``configured_model_path`` is the model name of the engine at ``base_url``; a test answers that engine
    or leaves it unreachable. A snapshot's version is the text of its ``model.txt``, or its revision without that
    file. ``activated`` lists the versions of the activated snapshots, in order. For each activation,
    ``activated_ids`` holds the artifact ID and ``activated_scenarios`` the scenario that the activation named, or
    None. ``served`` is the last version and ``active_artifact_id`` is the ID of that snapshot's artifact; both are
    None before the first activation. ``answers_each_artifact`` is False at first. A test sets it, and the runtime
    then answers each call from the artifact that the call names, as Tinker does (``serves_any_artifact``).
    ``per_scenario_adapters`` and ``shared_by_training_scenarios`` are what the runtime reports for the properties
    of the same names.

    Purpose: scenario, dispatcher and service tests bind it to a recipe in place of a runtime with engines.

    The runtime keeps one binding: ``served_artifact`` gives a route only for ``active_artifact_id``. Its capacity
    is the base default. So it deploys an artifact beside the active one (``serve_artifact``) only with
    ``answers_each_artifact``, or in a subclass whose ``served_artifact_capacity`` is above 1. A snapshot without
    weights routes to the configured model, as on an inference proxy (``endpoint_model_name``).
    """

    def __init__(
        self,
        base_url: str = "http://inference-runtime",
        *,
        model_path: str = "stub-model",
        inference_timeout_s: float = 300.0,
        per_scenario_adapters: bool = False,
        shared_by_training_scenarios: bool = False,
    ) -> None:
        super().__init__(base_url=base_url, inference_timeout_s=inference_timeout_s)
        self.configured_model_path = model_path
        self.handler = HttpInferenceHandler(self.base_url, timeout_s=self.inference_timeout_s)
        self.activated: list[str] = []
        self.activated_ids: list[str] = []
        self.activated_scenarios: list[str | None] = []
        self.served: str | None = None
        self.active_artifact_id: str | None = None
        self.answers_each_artifact = False
        self.adapters_per_scenario = per_scenario_adapters
        self.training_scenarios_share = shared_by_training_scenarios

    @property
    def model_path(self) -> str:
        return self.configured_model_path

    @property
    def inference_handler(self) -> InferenceHandler:
        """A plain HTTP handler for ``base_url``, as an inference proxy has."""
        return self.handler

    @property
    def serves_any_artifact(self) -> bool:
        return self.answers_each_artifact

    @property
    def per_scenario_adapters(self) -> bool:
        return self.adapters_per_scenario

    @property
    def shared_by_training_scenarios(self) -> bool:
        return self.training_scenarios_share

    def artifact_model(self, artifact: Artifact) -> PolicyModel:
        if not self.answers_each_artifact:
            raise ReefError("the stub runtime answers only from the snapshot that it activated")
        return PolicyModel("base", artifact.ref.revision)

    def serve_artifact(self, artifact: Artifact, *, scenario: str) -> ServedArtifact:
        """Deploy ``artifact`` under its own model name, ``sampler:<artifact_id>``.

        Raise ``ReefError`` at a capacity of 1."""
        if self.served_artifact_capacity == 1:
            raise ReefError("the stub runtime keeps one snapshot, so it deploys no artifact beside the active one")
        return ServedArtifact(artifact.artifact_id, self.base_url, f"sampler:{artifact.artifact_id}")

    def activate_checkpoint(self, artifact: Artifact, *, scenario: str | None = None) -> str:
        self.activated_ids.append(artifact.artifact_id)
        self.activated_scenarios.append(scenario)
        version = self.serve(artifact)
        self.activated.append(version)
        self.active_artifact_id = artifact.artifact_id
        return version

    def served_artifact(self, artifact_id: str, scenario: str) -> ServedArtifact | None:
        """The route of the last activated artifact, at ``base_url`` under ``served_model_name``. None for any other
        artifact, and when the stub has no model name or no HTTP(S) ``base_url``."""
        if artifact_id != self.active_artifact_id:
            return None
        try:
            return ServedArtifact(artifact_id, self.base_url, self.served_model_name(scenario))
        except ValueError:
            return None

    def endpoint_model_name(self, scenario: str) -> str | None:
        """The configured model for a snapshot without weights; None when the stub names no model."""
        return self.served_model_name(scenario) or None

    def serve(self, artifact: Artifact) -> str:
        path = artifact.local_path
        marker = None if path is None else Path(path) / VERSION_FILE
        version = marker.read_text() if marker is not None and marker.is_file() else artifact.ref.revision
        self.served = version
        return version


class StubFacts(TrainingBackendFacts):
    """The facts of the test component ``component``: each member answers from the component itself.

    ``default_start`` is ``component.stub_default_start()``, ``check_root_start`` raises exactly when
    ``component.root_start`` raises, and ``check_import_start`` exactly when ``component.import_start`` raises.
    A test gives the value to ``LiveObjects.training_facts`` beside the component, as a deployment's assembly
    gives the facts of the backend's configuration.
    """

    def __init__(self, component: StubTrainingComponent) -> None:
        if not isinstance(component, StubTrainingComponent):
            raise TypeError("StubFacts states the facts of a StubTrainingComponent")
        self.component = component

    @property
    def name(self) -> str:
        return self.component.name

    @property
    def artifact_format(self) -> ArtifactFormat | None:
        return self.component.artifact_format

    @property
    def shares_scenarios(self) -> bool:
        return self.component.shares_scenarios

    @property
    def concurrent_jobs(self) -> bool:
        return self.component.concurrent_jobs

    @property
    def default_start_inputs(self) -> Mapping[str, str]:
        return self.component.stub_default_start().inputs

    def default_start(self) -> DefaultStart:
        return self.component.stub_default_start()

    def parameters_schema(self, kind: JobDataKind) -> dict[str, Any] | None:
        return self.component.parameters_schema(kind)

    def check_root_start(self, root: ScenarioRoot) -> None:
        self.component.root_start(root)

    def check_import_start(self, artifact_dir: Path) -> None:
        self.component.check_import_start(artifact_dir)


def live_objects(
    component: TrainingComponent | None = None,
    runtime: StubInferenceRuntime | None = None,
    *,
    data_store: DataStore | None = None,
) -> LiveObjects:
    """The live objects of a test deployment that trains weights: the training component ``component``, its facts
    when it is a stub (``StubFacts``), and the inference runtime ``runtime``; stubs fill what is missing.
    ``data_store`` is the deployment's data store."""
    training_component = StubTrainingComponent() if component is None else component
    return LiveObjects(
        runtime=StubInferenceRuntime() if runtime is None else runtime,
        training_component=training_component,
        training_facts=(
            StubFacts(training_component) if isinstance(training_component, StubTrainingComponent) else None
        ),
        data_store=data_store,
    )


def harness_live_objects(component: StubTrainingComponent, *, data_store: DataStore | None = None) -> LiveObjects:
    """The live objects of a test deployment that evolves a harness: ``component`` as the harness training
    component, and its facts (``StubFacts``). The deployment has no runtime."""
    return LiveObjects(harness_component=component, harness_facts=StubFacts(component), data_store=data_store)


def job_sized_stages(
    spec: PipelineSpec, section: Mapping[str, Any] | None = None, *, job_samples: int = 1
) -> PipelineSettings:
    """The stages of ``spec`` configured by a deployment ``pipeline`` section, with ``job_samples`` set.

    A weight recipe needs ``pipeline.sample2job.config.job_samples``; a ``job_samples`` in ``section`` wins.
    """
    stages = {key: dict(value) for key, value in (section or {}).items()}
    sample2job = stages.setdefault("sample2job", {})
    sample2job["config"] = {"job_samples": job_samples, **sample2job.get("config", {})}
    return PipelineSettings.from_setting(stages, spec=spec, environ={})


class RunningEngines(ServingEngines):
    """Engines that a test started behind ``backend`` and stops: opening returns them as new engines, and closing
    stops nothing. They declare ``request_pinning``, as a receiver that keeps each request on its weights does;
    the fake backend admits no request, so the tests cover the runtime's own calls only.

    Purpose: the serving runtime tests run ``AdapterServingRuntime`` and ``FullWeightServingRuntime`` on fake engines.
    """

    def __init__(self, backend: InferenceBackend) -> None:
        self.backend = backend

    @property
    def base_url(self) -> str:
        return self.backend.inference_url()

    @property
    def request_pinning(self) -> bool:
        return True

    def open(self) -> OpenedEngines:
        return OpenedEngines(self.backend, attached=False)

    def check_health(self) -> None:
        return
