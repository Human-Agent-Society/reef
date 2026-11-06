"""Bounded training jobs and the training components that run them.

Reef submits a ``TrainingJob`` to a ``TrainingComponent``. The component runs
its framework's own training loop over the job's data, starting from the job's
start, and saves checkpoints. Reef reads the job's ``JobStatus``, evaluates each
saved checkpoint as a candidate, and serves the ones that pass. The job's
``parameters`` belong to the backend: Reef reads none of them, and they
override the backend's own defaults. A component keeps its own training state,
such as an objective's counters or an optimizer's search state, inside the
checkpoints it saves, so a job that starts from a checkpoint continues it.

A ``TrainingComponentFactory`` builds a component from the component config
of a training deployment (``build_training_component``).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from reef.artifact.artifact import Artifact
from reef.artifact.components import ArtifactLayout
from reef.artifact.files import ArtifactFiles
from reef.artifact.repository import DefaultStart
from reef.core.config import ConfiguredFactory
from reef.core.errors import RuntimeConfigError
from reef.core.snapshot import RootSource
from reef.core.weight_layout import ADAPTER_DIR, ArtifactFormat
from reef.storage.job_data import JobDataKind, JobDataRef

JobState = Literal["running", "finished", "failed"]


@dataclass(frozen=True)
class TrainingJob:
    """One bounded training job: its data directory, the weights it starts from, and the backend's parameters.

    A value has a non-empty ``job_id``, a non-empty ``backend`` name, ``start``
    (the non-empty start that the component reads, ``TrainingComponent.root_start`` for a scenario's root snapshot),
    ``data`` (the job data directory that the job trains, ``reef.storage.job_data``; the component reads it
    with ``read_job_data(job.data_dir, job.data_id)`` and cuts it into optimizer batches by its own batch size),
    ``parameters``, a mapping that encodes as JSON, and ``scenario``, the non-empty name of the scenario whose
    trainer submits the job. The jobs of one scenario run one at a time, in the order of their submission.
    ``request_id`` is the ID of the training request that the job trains, or None. Reef's job listing reads it; a
    component does not need it. ``component`` is the scenario component that the job trains, or None when the
    trainer names none. ``previous_job_id`` is the last job whose end the trainer committed, or None before the
    first: a component whose state advances also in a job that saves no checkpoint, such as a harness, continues the
    state after it. ``train_from_artifact_id`` is the artifact that ``start`` came from, or None when the trainer did
    not give it. Reef's job listing reads it; a component does not need it.

    Reef makes the value and names the job with ``job_id_for``, so a retry of
    the same request has the same ID. A component reads it in ``submit``.
    """

    job_id: str
    backend: str
    start: str
    data: JobDataRef
    parameters: Mapping[str, Any] = field(default_factory=dict)
    scenario: str = field(kw_only=True)
    request_id: str | None = field(default=None, kw_only=True)
    component: str | None = field(default=None, kw_only=True)
    previous_job_id: str | None = field(default=None, kw_only=True)
    train_from_artifact_id: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, str) or not self.job_id:
            raise ValueError("TrainingJob.job_id must be a non-empty string")
        if not isinstance(self.scenario, str) or not self.scenario:
            raise ValueError("TrainingJob.scenario must be a non-empty string")
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("TrainingJob.backend must be a non-empty string")
        if not isinstance(self.start, str) or not self.start:
            raise ValueError("TrainingJob.start must be a non-empty string")
        if not isinstance(self.data, JobDataRef):
            raise TypeError("TrainingJob.data must be a JobDataRef")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("TrainingJob.parameters must be a mapping")
        for name in ("request_id", "component", "previous_job_id", "train_from_artifact_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"TrainingJob.{name} must be a non-empty string or None")
        json.dumps(dict(self.parameters), allow_nan=False)

    @property
    def data_dir(self) -> Path:
        """The local job data directory: read it with ``read_job_data``."""
        return self.data.path

    @property
    def data_uri(self) -> str:
        """The URI of the job data directory, which a component that runs a container mounts."""
        return self.data.uri

    @property
    def data_id(self) -> str:
        return self.data.data_id

    @property
    def sample_count(self) -> int:
        return self.data.sample_count

    @property
    def kind(self) -> JobDataKind | None:
        """The kind of the job's samples, or None for a job without samples."""
        return self.data.kind


def job_id_for(backend: str, start: str, data_identity: object, parameters: Mapping[str, Any]) -> str:
    """The ID of a job: a hash of its backend, start, data and parameters, so a retry gets the same ID.

    ``data_identity`` is a JSON value that names the data, not the job data's
    process-local number: a reload numbers the same rows again.
    """
    encoded = json.dumps(
        {"backend": backend, "start": start, "data": data_identity, "parameters": dict(parameters)},
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SavedCheckpoint:
    """A checkpoint a job saved: the step that saved it, where it is, what Reef publishes of it, and its metrics.

    ``step`` is a non-negative int, counted within the job. ``path`` is a
    non-empty directory path as the component wrote it, which a later job
    can start from. ``artifact_path`` is the non-empty directory that Reef
    publishes as a snapshot when it serves the checkpoint: the weights that
    serving loads, without the training state that only the component reads
    (for example ``path`` itself, or its Hugging Face export).
    ``metrics`` is the backend's telemetry for the training that led to this
    checkpoint, a mapping that encodes as JSON, which Reef records without
    reading. A component makes the value; ``JobLoop`` reads it.
    """

    step: int
    path: str
    artifact_path: str
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.step, int) or isinstance(self.step, bool) or self.step < 0:
            raise ValueError("SavedCheckpoint.step must be a non-negative int")
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("SavedCheckpoint.path must be a non-empty string")
        if not isinstance(self.artifact_path, str) or not self.artifact_path:
            raise ValueError("SavedCheckpoint.artifact_path must be a non-empty string")
        if not isinstance(self.metrics, Mapping):
            raise TypeError("SavedCheckpoint.metrics must be a mapping")
        json.dumps(dict(self.metrics), allow_nan=False)


@dataclass(frozen=True)
class JobStatus:
    """What a component reports about one job.

    ``job_id`` is the job's non-empty ID. ``state`` is ``running``,
    ``finished`` or ``failed``. ``checkpoints`` are the checkpoints saved so
    far, in increasing step order; once ``state`` is not ``running``, the list
    is complete. ``metrics`` is the backend's telemetry, a mapping that encodes
    as JSON, which Reef records without reading. ``error`` is a non-empty
    string exactly when the job failed. A finished job without checkpoints
    trained nothing, for example because its objective skipped the job's data.
    ``TrainingComponent.status`` makes the value; ``JobLoop.poll`` reads it.
    """

    job_id: str
    state: JobState
    checkpoints: tuple[SavedCheckpoint, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, str) or not self.job_id:
            raise ValueError("JobStatus.job_id must be a non-empty string")
        if self.state not in ("running", "finished", "failed"):
            raise ValueError(f"JobStatus.state must be running, finished or failed, got {self.state!r}")
        if (self.error is not None) != (self.state == "failed"):
            raise ValueError("JobStatus.error must be set exactly when the job failed")
        if self.error is not None and (not isinstance(self.error, str) or not self.error):
            raise ValueError("JobStatus.error must be a non-empty string")
        if not isinstance(self.metrics, Mapping):
            raise TypeError("JobStatus.metrics must be a mapping")
        json.dumps(dict(self.metrics), allow_nan=False)
        if not all(isinstance(checkpoint, SavedCheckpoint) for checkpoint in self.checkpoints):
            raise TypeError("JobStatus.checkpoints must hold SavedCheckpoint values")
        steps = [checkpoint.step for checkpoint in self.checkpoints]
        if steps != sorted(set(steps)):
            raise ValueError("JobStatus.checkpoints must be in increasing step order")


@dataclass(frozen=True)
class ScenarioRoot:
    """A scenario's root snapshot, as one trainer's component sees it, and where the root came from.

    Values: ``artifact`` is the trainer's component of the scenario's root snapshot (the snapshot itself when the
    snapshot is flat), which ``materialize`` makes local. ``source`` is the root's ``RootSource``.

    Purpose: ``ScenarioFactory`` makes it when it builds a scenario's trainers, and ``Trainer.build`` gives it to
    ``TrainingComponent.root_start`` for the first job of a trainer that has no committed state.
    """

    artifact: Artifact
    source: RootSource

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, Artifact) or not isinstance(self.source, RootSource):
            raise TypeError("a scenario root is an Artifact and a RootSource")


class TrainingComponent(ABC):
    """A training backend that runs whole jobs with its framework's own training loop.

    A component never learns which scenario a job is for: a job carries
    everything it trains (its start, data directory and parameters). ``submit`` records
    a job durably and returns; the job then runs in the background.
    ``status`` reports a submitted job, also after the component restarts: a
    restarted component resumes each unfinished job or reports it failed.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The backend name that jobs for this component carry."""

    @property
    @abstractmethod
    def artifact_format(self) -> ArtifactFormat | None:
        """The weight files that each saved checkpoint's snapshot holds.

        ``None`` means the snapshots hold no weight files that an engine
        loads: only the backend's own serving runtime reads them (for
        example a Tinker manifest that Tinker's sampler serves).
        """

    @abstractmethod
    def default_start(self) -> DefaultStart:
        """The component's untrained checkpoint, laid out as one of its snapshots, and the configuration it comes from.

        A scenario that is created without an artifact item holds it in its root snapshot, in the component that
        this component trains. It has the layout of every other snapshot (``artifact_format``), so serving and
        training read it as they read any snapshot. Its ``inputs`` name the model and the settings that decide its
        content. Every service process can call this method, so a component whose model files cost a remote call
        gives them as ``BaseModelFiles`` that make them on the first ``directory`` call.
        """

    def root_start(self, root: ScenarioRoot) -> str:
        """The start of the first job of a trainer whose scenario has the root snapshot ``root``.

        The default is ``import_start`` of the root's files. A component that imports no snapshot
        overrides it, and raises ``ValueError`` for a root that it cannot start
        from.
        """
        local_path = root.artifact.materialize().local_path
        if local_path is None:
            raise ValueError(f"the root snapshot {root.artifact.ref.revision} has no local files to train from")
        return self.import_start(local_path)

    @property
    def shares_scenarios(self) -> bool:
        """Whether jobs of several scenarios can run on this component.

        True exactly when a job's weights depend only on the job: its
        ``start``, data and parameters, and never on state that an earlier job
        left in the component, such as weights held in memory. Then one
        scenario's job never trains another scenario's weights, and the
        scenario registry lets several training scenarios share the
        component. The default is False.
        """
        return False

    def check_job(self, job: TrainingJob) -> None:
        """Raise ``ValueError`` for a job this component cannot train. The default accepts every job.

        A queue-backed component calls it before it adds the job to the jobs queue, so a refused job never waits
        in the queue.
        """
        return

    def parameters_schema(self, kind: JobDataKind) -> dict[str, Any] | None:
        """The JSON Schema of the complete parameters of a job of ``kind`` samples, or None when it declares none.

        A job's parameters are the recipe's parameters of that kind with the request's merged in. The default
        declares none. The scenario contract serves the schema, so a client can validate parameters against it.
        ``check_job`` still checks each job.
        """
        return None

    def check_task_jobs(self, parameters: Mapping[str, Any]) -> None:
        """Raise ``ValueError`` unless the component can train jobs whose data are ``TaskSample`` values, with ``parameters``.

        A recipe whose jobs train tasks calls it when it is built, so a deployment that cannot run them fails before
        any job. ``parameters`` are the recipe's parameters of a task job. The default refuses: the
        component trains trajectories only. ``check_job`` still checks each job.
        """
        raise ValueError(f"the {self.name} training component trains no task jobs")

    def check_artifact(self, files: ArtifactFiles) -> None:
        """Raise ``ValueError`` unless ``files`` are a snapshot in the layout of this component's snapshots.

        An artifact import calls it (RFC #980) on the directory it copied or on the revision it keeps by reference
        (RFC #995). The default reads ``artifact_format``: ``"full"`` needs ``config.json`` and weight files at the
        root; ``"adapter"`` needs a PEFT adapter in ``adapter/`` whose rank is at most ``adapter_rank`` and whose
        target modules are within ``adapter_target_modules``. It reads only file names and
        ``adapter/adapter_config.json``, so a revision is checked without downloading its weights. ``None``
        refuses every snapshot. A component whose snapshots name weights on a service overrides it and also checks
        that each path resolves there.
        """
        artifact_format = self.artifact_format
        if artifact_format is None:
            raise ValueError(f"the {self.name} training component takes no imported snapshots")
        paths = set(files.paths())
        if artifact_format.layout == "full":
            if "config.json" not in paths or not any(
                "/" not in path and path.endswith((".safetensors", ".bin")) for path in paths
            ):
                raise ValueError("a full-weight artifact holds config.json and weight files at its root")
            return
        if not paths & {f"{ADAPTER_DIR}/adapter_model.safetensors", f"{ADAPTER_DIR}/adapter_model.bin"}:
            raise ValueError(f"an adapter artifact holds a PEFT adapter in {ADAPTER_DIR}/")
        try:
            config = json.loads(files.read_bytes(f"{ADAPTER_DIR}/adapter_config.json"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"{ADAPTER_DIR}/adapter_config.json is no readable JSON object: {exc}") from exc
        rank = config.get("r") if isinstance(config, dict) else None
        # ArtifactFormat's invariant: an adapter format has a rank.
        max_rank = cast(int, artifact_format.adapter_rank)
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 < rank <= max_rank:
            raise ValueError(f"the adapter's rank must be a positive int at most {max_rank}")
        targets = artifact_format.adapter_target_modules
        modules = config.get("target_modules")
        if targets != ("all",) and (not isinstance(modules, list) or not set(modules) <= set(targets)):
            raise ValueError(f"the adapter's target modules must be a list within {list(targets)}, got {modules!r}")

    def changed_parts(self, start: str, checkpoint: SavedCheckpoint) -> frozenset[str]:
        """The parts of the scenario component that ``checkpoint`` changed from ``start``, which the scenario's
        activation rule reads (``TrainingCommit.changed_kinds``). The default is none."""
        return frozenset()

    @abstractmethod
    def import_start(self, artifact_dir: Path) -> str:
        """Take the published snapshot in ``release`` as the start of later jobs; return that start.

        Reef calls this when a trainer starts training from an artifact
        (``train-from``). The component copies what it trains
        from into its own durable directory, since ``release`` can be
        discarded later. Raise ``ValueError``, and change nothing, when the
        component cannot start from these weights.
        """

    @abstractmethod
    def submit(self, job: TrainingJob) -> None:
        """Record ``job`` durably and run it in the background; a second submit of its ID does nothing.

        Raise ``ValueError`` for a job the component refuses (for example a
        start it cannot continue from, or parameters it does not accept), and
        ``RuntimeError`` after ``close``; nothing is recorded then.
        """

    @abstractmethod
    def status(self, job_id: str) -> JobStatus:
        """Report the job; raise ``KeyError`` for a job this component never saw."""

    @abstractmethod
    def cancel(self, job_id: str) -> None:
        """Stop the job, which then reports ``failed``; nothing for an unknown or ended job."""

    def start_jobs(self) -> None:
        """Start running jobs in this process: take the component's job state and resume the jobs that have not ended.

        Only the process that runs the jobs calls it: the training service's ``TrainingJobWorker``, which a
        one-process deployment runs too. Another process, such as control, uses the component to describe jobs and
        to import starts, and never runs a job. A component whose jobs run elsewhere does nothing, the default.
        """
        return

    def close(self) -> None:
        """Release the component's resources; safe to call repeatedly."""
        return


@dataclass(frozen=True)
class TrainingArtifactLayout(ArtifactLayout):
    """The layout of the snapshots of ``component``: an import of the component it trains must meet it.

    ``check`` is ``component.check_artifact``. The snapshots resolve a reference when they hold weight files
    (``artifact_format``), which the serving runtimes read through ``ArtifactWeightFiles``. A weight recipe binds it on
    its ``weights`` component.
    """

    component: TrainingComponent

    def check(self, files: ArtifactFiles) -> None:
        self.component.check_artifact(files)

    @property
    def resolves_references(self) -> bool:
        return self.component.artifact_format is not None


class TrainingComponentFactory(ConfiguredFactory):
    """Build the training component that a training deployment's component config describes.

    ``kind`` is the dotted reference ``package.module:factory`` that the
    config's ``type`` holds (``TrainingDeployment.component_type``).
    ``__call__`` receives ``(config, model_path, environ)`` and returns a
    :class:`TrainingComponent`; ``build_training_component`` calls it.
    """

    @abstractmethod
    def __call__(self, config: Mapping[str, Any], model_path: str, environ: Mapping[str, str]) -> TrainingComponent:
        """Build the training component that ``config`` describes."""


def training_component_factory_for(reference: str) -> TrainingComponentFactory:
    """The ``TrainingComponentFactory`` that the dotted ``reference`` (``package.module:factory``) names.

    The reference names a factory instance or a subclass, which is made with no
    arguments. ``RuntimeConfigError`` for another reference.
    """
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise RuntimeConfigError(f"training component type {reference!r} must be 'package.module:factory'")
    try:
        candidate = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise RuntimeConfigError(f"cannot import training component type {reference!r}: {exc}") from exc
    if isinstance(candidate, type) and issubclass(candidate, TrainingComponentFactory):
        candidate = candidate()
    if not isinstance(candidate, TrainingComponentFactory):
        raise RuntimeConfigError(f"training component type {reference!r} is not a TrainingComponentFactory")
    return candidate


def build_training_component(
    config: Mapping[str, Any], *, model_path: str, environ: Mapping[str, str] | None = None
) -> TrainingComponent:
    """The training component of ``config``, whose ``type`` names its factory (``training_component_factory_for``).

    ``TypeError`` when the factory builds anything but a ``TrainingComponent``.
    """
    reference = config.get("type")
    if not isinstance(reference, str) or not reference:
        raise RuntimeConfigError("a training component config needs its factory's dotted reference as type")
    factory = training_component_factory_for(reference)
    values = os.environ if environ is None else environ
    component = factory(factory.parse_config(config, values), model_path, values)
    if not isinstance(component, TrainingComponent):
        raise TypeError(f"training component type {reference!r} must build a TrainingComponent")
    return component
