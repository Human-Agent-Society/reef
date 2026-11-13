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
of a training deployment (``build_training_component``). The same factory states the backend's
``TrainingBackendFacts`` from that config, without a component (``training_backend_facts``): a service that does not
run the component reads them, and a new scenario's root snapshot holds the default start that they write.

Only the training service builds the component. Control holds the facts and the jobs queue
(``reef.train.job_queue``): it names the snapshot that a job starts from (``SnapshotStart``), and the training
service makes the component's start of it when it takes the job.
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
from reef.artifact.base_root import base_root_of
from reef.artifact.components import ArtifactLayout
from reef.artifact.files import ArtifactFiles
from reef.artifact.repository import DefaultStart
from reef.core.config import ConfiguredFactory
from reef.core.errors import InternalError, ReefError, RuntimeConfigError
from reef.core.job_data import JobData
from reef.core.snapshot import RootSource
from reef.core.values import JsonValue
from reef.core.weight_layout import ADAPTER_DIR, ArtifactFormat
from reef.storage.job_data import JobDataKind, JobDataRef

JobState = Literal["running", "finished", "failed"]
#: The first characters of a start that names a snapshot (``SnapshotStart.to_text``). No component's own start
#: begins with them.
SNAPSHOT_START_PREFIX = "snapshot:"


class JobCheckPendingError(ReefError):
    """``TrainingComponent.submit`` recorded nothing, because the training service has not answered the check of
    the job yet. A later ``submit`` of a job with the same ID reads the answer."""


@dataclass(frozen=True)
class SnapshotStart:
    """A job's start that names a snapshot of the job's scenario, for the training service to resolve.

    Values: ``revision`` is the non-empty revision of the snapshot. ``root`` is the ``RootSource`` of the job's
    component when the snapshot is the scenario's root and the job is the first of a new trainer; else None, for a
    snapshot that a ``train-from`` or a request's ``train_from`` names.

    ``to_text()`` is ``SNAPSHOT_START_PREFIX`` and the value as JSON; ``from_text`` of it is an equal value.
    ``from_text(text)`` is None for a text without the prefix, which is a component's own start, and raises
    ``ValueError`` for a text with the prefix that ``to_text`` did not write.

    Purpose: control's ``reef.train.job_queue.JobQueueComponent`` makes the text as the start of a job. The
    training service reads it when it takes the job (``reef.train.job_queue.resolved_job``): it gives the job's
    component of the snapshot to ``TrainingComponent.root_start`` (with ``root``) or ``import_start``.
    """

    revision: str
    root: RootSource | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.revision, str) or not self.revision:
            raise ValueError("SnapshotStart.revision must be a non-empty string")
        if self.root is not None and not isinstance(self.root, RootSource):
            raise TypeError("SnapshotStart.root must be a RootSource or None")

    def to_text(self) -> str:
        value = {"revision": self.revision, "root": None if self.root is None else self.root.to_dict()}
        return SNAPSHOT_START_PREFIX + json.dumps(value, separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_text(cls, text: str) -> SnapshotStart | None:
        if not text.startswith(SNAPSHOT_START_PREFIX):
            return None
        value = json.loads(text.removeprefix(SNAPSHOT_START_PREFIX))
        if not isinstance(value, dict) or set(value) != {"revision", "root"}:
            raise ValueError(f"a snapshot start holds revision and root, not {text!r}")
        return cls(value["revision"], None if value["root"] is None else RootSource.from_dict(value["root"]))


@dataclass(frozen=True)
class TrainingJob:
    """One bounded training job: its data directory, the weights it starts from, and the backend's parameters.

    A value has a non-empty ``job_id``, a non-empty ``backend`` name, ``start``
    (the non-empty start: one that the component made, such as ``SavedCheckpoint.path``, or in a job that control
    queued the name of a snapshot, ``SnapshotStart.to_text``, which the training service replaces by the component's
    start of that snapshot before the component reads the job),
    ``data`` (the reference to the job data directory that the job trains, ``reef.storage.job_data``; a component
    that trains on Reef's host reads it with ``job.read_data()`` and cuts it into optimizer batches by its own batch
    size, and a component that runs the job elsewhere gives that place ``data_uri``),
    ``parameters``, a mapping that encodes as JSON, and ``scenario``, the non-empty name of the scenario whose
    trainer submits the job. The jobs of one scenario can run at the same time when the component has
    ``concurrent_jobs``; each starts from its own ``start``.
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
        """The local job data directory: read it with ``read_data``. The first read copies it from a remote store."""
        return self.data.path

    def read_data(self) -> JobData:
        """The job's data, without a request: its samples in job order, each task with its local directory. The
        read copies the directory and the tasks' directories from a remote store when the host does not hold them."""
        return self.data.read()

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
    reading. ``changed_parts`` is a frozenset of non-empty strings: empty in the value that a component makes, and
    ``TrainingComponent.changed_parts`` of the checkpoint in the value that the training service writes into the
    jobs queue. A component makes the value; ``JobLoop`` reads it.
    """

    step: int
    path: str
    artifact_path: str
    metrics: Mapping[str, Any] = field(default_factory=dict)
    changed_parts: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not isinstance(self.step, int) or isinstance(self.step, bool) or self.step < 0:
            raise ValueError("SavedCheckpoint.step must be a non-negative int")
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("SavedCheckpoint.path must be a non-empty string")
        if not isinstance(self.artifact_path, str) or not self.artifact_path:
            raise ValueError("SavedCheckpoint.artifact_path must be a non-empty string")
        if not isinstance(self.metrics, Mapping):
            raise TypeError("SavedCheckpoint.metrics must be a mapping")
        if not isinstance(self.changed_parts, frozenset) or not all(
            isinstance(part, str) and part for part in self.changed_parts
        ):
            raise TypeError("SavedCheckpoint.changed_parts must be a frozenset of non-empty strings")
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

    def local_directory(self) -> Path:
        """The directory of the root's files, made local (``Artifact.materialize``); ``ValueError`` when the
        artifact has no local files."""
        local_path = self.artifact.materialize().local_path
        if local_path is None:
            raise ValueError(f"the root snapshot {self.artifact.ref.revision} has no local files to train from")
        return local_path


def component_root(root: ScenarioRoot, component: str) -> ScenarioRoot:
    """``component``'s view of the composite's root ``root``: the component's directory of the root snapshot, and,
    for a default start, the inputs of that component's own default start (the composite's default start names
    them ``<component>.<input>``)."""
    source = root.source
    if source.kind == "default":
        prefix = f"{component}."
        source = RootSource(
            "default",
            inputs={
                name.removeprefix(prefix): value for name, value in source.inputs.items() if name.startswith(prefix)
            },
        )
    return ScenarioRoot(root.artifact.component(component), source)


class TrainingBackendFacts(ABC):
    """What a training backend states about itself from its configuration alone, without a training component.

    Values: a backend defines one concrete type in its own package and makes a value from its own settings
    (``TrainingComponentFactory.facts``). Making a value starts no process, opens no connection and imports no
    training framework or service SDK. Each concrete type checks its fields when a value is made.

    Relation to the component: let ``component`` be the backend's ``TrainingComponent`` of the same settings. Then
    ``name``, ``artifact_format``, ``shares_scenarios`` and ``concurrent_jobs`` equal the component's,
    ``parameters_schema(kind)`` equals ``component.parameters_schema(kind)``, ``check_root_start(root)`` raises
    exactly when ``component.root_start(root)`` raises, and ``check_import_start(artifact_dir)`` exactly when
    ``component.import_start(artifact_dir)`` raises, each with the same message. The component holds a value and
    answers these members from it, so the two cannot differ. Only the facts state the default start: the
    component has no ``default_start`` and no ``default_start_inputs``. ``default_start().inputs`` equals
    ``default_start_inputs``, and ``component.root_start`` accepts a root that holds ``default_start()``: the
    component starts training from the files that the facts write.

    A member reads at most the settings, the files that a caller hands it (``check_root_start`` reads the root
    snapshot's files), and local paths that the settings name. No member writes a file, calls the backend's service
    or needs a running component. What needs one stays on ``TrainingComponent`` and runs in the training service:
    the start of a job (``root_start``, ``import_start``), ``check_remote_artifact``, ``check_task_jobs``,
    ``check_job``, ``changed_parts`` and the jobs themselves.

    Purpose: control builds no training component. The scenario factory writes the root snapshot of a new scenario
    from ``default_start`` and checks an item root with ``check_root_start`` (``reef.scenario.live_objects``).
    Control's view of the jobs queue (``reef.train.job_queue.JobQueueComponent``) answers a trainer from the facts
    and asks the training service for the rest. The inference service reads ``artifact_format``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The backend name that jobs for the component carry (``TrainingComponent.name``)."""

    @property
    @abstractmethod
    def artifact_format(self) -> ArtifactFormat | None:
        """The weight files that each saved checkpoint's snapshot holds, or None when an engine loads none
        (``TrainingComponent.artifact_format``)."""

    @property
    @abstractmethod
    def shares_scenarios(self) -> bool:
        """Whether jobs of several scenarios can run on the component (``TrainingComponent.shares_scenarios``)."""

    @property
    @abstractmethod
    def concurrent_jobs(self) -> bool:
        """Whether several jobs of one scenario can run at the same time (``TrainingComponent.concurrent_jobs``)."""

    @property
    @abstractmethod
    def default_start_inputs(self) -> Mapping[str, str]:
        """The ``inputs`` of the component's default start: the model and the settings that decide its content.

        A new scenario without an artifact item records them in its ``RootSource``.
        """

    @abstractmethod
    def default_start(self) -> DefaultStart:
        """The content of the component's part of a new scenario's root snapshot, when no artifact item starts it.

        The value holds a few small text files that the settings decide, and ``default_start_inputs`` as its
        ``inputs``. A backend whose artifacts are weights of a base model writes a base root
        (``reef.artifact.base_root``) or a reference to the base model's commit
        (``reef.artifact.sources.REFERENCE_FILE``), and no weight file. The backend's component starts a
        scenario's first job from a root that holds these files (``TrainingComponent.root_start``).
        """

    @abstractmethod
    def parameters_schema(self, kind: JobDataKind) -> dict[str, Any] | None:
        """The JSON Schema of the complete parameters of a job of ``kind`` samples, or None when the backend
        declares none (``TrainingComponent.parameters_schema``)."""

    @abstractmethod
    def check_root_start(self, root: ScenarioRoot) -> None:
        """Raise ``ValueError`` unless a trainer of the backend can start from the scenario root ``root``.

        It refuses exactly the roots that ``TrainingComponent.root_start`` refuses. It writes nothing: the
        component makes the start itself when the first job needs it.
        """

    @abstractmethod
    def check_import_start(self, artifact_dir: Path) -> None:
        """Raise ``ValueError`` unless a job of the backend can start from the published snapshot in ``artifact_dir``.

        It refuses exactly the directories that ``TrainingComponent.import_start`` refuses, and writes nothing.
        Control calls it when a trainer starts training from an artifact (``train-from``, a request's
        ``train_from``), so the refusal reaches the caller at once.
        """

    def check_artifact(self, files: ArtifactFiles) -> Mapping[str, JsonValue] | None:
        """Check ``files`` as a snapshot in the layout of the backend's snapshots, as far as the files and the
        settings decide; return what remains to check on the backend's service.

        Raise ``ValueError`` naming what does not hold. Return None when the check is complete. Otherwise return a
        JSON object that names what must resolve on the backend's service; the training service gives it to
        ``TrainingComponent.check_remote_artifact``, and the artifact is admitted only after that returns.

        An artifact import calls it (RFC #980) on the directory it copied or on the revision it keeps by reference
        (RFC #995). The default is complete without a service. It first accepts a base root of the backend's default
        start (``is_own_base_root``). Otherwise it reads ``artifact_format``: ``"full"`` needs ``config.json`` and
        weight files at the root; ``"adapter"`` needs a PEFT adapter in ``adapter/`` whose rank is at most
        ``adapter_rank`` and whose target modules are within ``adapter_target_modules``. It reads only file names
        and ``adapter/adapter_config.json``, so a revision is checked without downloading its weights. ``None``
        refuses every snapshot.
        """
        artifact_format = self.artifact_format
        if artifact_format is None:
            raise ValueError(f"the {self.name} training component takes no imported snapshots")
        if self.is_own_base_root(files):
            return None
        paths = set(files.paths())
        if artifact_format.layout == "full":
            if "config.json" not in paths or not any(
                "/" not in path and path.endswith((".safetensors", ".bin")) for path in paths
            ):
                raise ValueError("a full-weight artifact holds config.json and weight files at its root")
            return None
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
        return None

    def is_own_base_root(self, files: ArtifactFiles) -> bool:
        """Whether ``files`` are a base root (``reef.artifact.base_root``) of the backend's default start.

        False for files that are no base root. Raise ``ValueError`` for a base root that is not readable, that
        holds other files, or whose inputs differ from ``default_start_inputs``: it stands for another base model
        or another adapter configuration than this deployment's.
        """
        base = base_root_of(files)
        if base is None:
            return False
        base.check_inputs(self.default_start_inputs)
        return True


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

    def root_start(self, root: ScenarioRoot) -> str:
        """The start of the first job of a trainer whose scenario has the root snapshot ``root``.

        The default is ``import_start`` of the root's files. A component that imports no snapshot
        overrides it, and raises ``ValueError`` for a root that it cannot start
        from. A root of a new scenario holds the default start of the backend's facts
        (``TrainingBackendFacts.default_start``): for a backend of weights, a base root or a reference to the base
        model, from which the component starts the job at the base model.
        """
        return self.import_start(root.local_directory())

    def snapshot_start(self, artifact: Artifact) -> str:
        """The start of a job that trains from ``artifact``, the trainer's component of a published snapshot,
        materialized. The default is ``import_start`` of its directory. Raise ``ValueError`` when the component
        cannot start from it, or when the artifact has no local files."""
        if artifact.local_path is None:
            raise ValueError(f"artifact {artifact.ref.revision!r} has no local files to train from")
        return self.import_start(artifact.local_path)

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

    @property
    def concurrent_jobs(self) -> bool:
        """Whether several jobs of one scenario can run on this component at the same time.

        True exactly when a job depends only on its own ``start``, data and parameters, and never on a job that
        runs beside it or on ``previous_job_id``. The default is False: the trainer then submits a job only when
        no other job of its scenario runs.
        """
        return False

    def check_job(self, job: TrainingJob) -> None:
        """Raise ``ValueError`` for a job this component cannot train. The default accepts every job.

        The training service calls it for control before control adds the job to the jobs queue
        (``reef.train.job_queue.TrainingCheckWorker``), so a refused job never waits in the queue and consumes no
        sample. ``submit`` checks the job again.
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

        The training service calls it at its start for a recipe whose jobs train tasks
        (``reef.scenario.live_objects.check_recipe_objects``), so a deployment that cannot run them fails before
        any job. ``parameters`` are the recipe's parameters of a task job. The default refuses: the
        component trains trajectories only. ``check_job`` still checks each job.
        """
        raise ValueError(f"the {self.name} training component trains no task jobs")

    def check_remote_artifact(self, names: Mapping[str, JsonValue]) -> None:
        """Raise ``ValueError`` unless everything that ``names`` names resolves on the component's service.

        ``names`` is what the backend's ``TrainingBackendFacts.check_artifact`` returned for the files of an
        import. The training service calls it; the import is refused with the error's message. A backend whose
        facts return None for every artifact is never asked, so the default raises ``InternalError``.
        """
        raise InternalError(f"the {self.name} training component checks no artifact on a service")

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
        ``RuntimeError`` after ``close``; nothing is recorded then. A component that asks another process to check
        the job raises ``JobCheckPendingError`` while the answer is not there; nothing is recorded then either, and
        the trainer submits the same job again later.
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
        one-process deployment runs too. No other process builds the component. A component whose jobs run
        elsewhere does nothing, the default.
        """
        return

    def close(self) -> None:
        """Release the component's resources; safe to call repeatedly."""
        return


@dataclass(frozen=True)
class TrainingArtifactLayout(ArtifactLayout):
    """The layout of the snapshots of the training backend of ``facts``: an import of the component it trains must
    meet it.

    Values: ``facts`` is the ``TrainingBackendFacts`` of the deployment's training backend. ``remote`` is the
    component that answers what the facts leave to the backend's service, or None in a process that has none.

    ``check(files)`` runs ``facts.check_artifact(files)``. When that returns names, it gives them to
    ``remote.check_remote_artifact``; without ``remote`` it raises ``InternalError``, which is no ``ValueError``, so
    no caller reads it as a refusal of the files. ``resolves_references`` is true exactly when
    ``facts.artifact_format`` is not None: the snapshots then hold weight files, which the serving runtimes read
    through ``ArtifactWeightFiles``.

    Purpose: ``reef.scenario.live_objects.scenario_components`` binds it on the weight component of a weight
    recipe. In control ``remote`` is the jobs queue's view, which asks the training service. The inference service
    binds it without ``remote``: it loads artifacts that control admitted and stores no import.
    """

    facts: TrainingBackendFacts
    remote: TrainingComponent | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.facts, TrainingBackendFacts):
            raise TypeError(f"facts must be TrainingBackendFacts, got {type(self.facts).__name__}")
        if self.remote is not None and not isinstance(self.remote, TrainingComponent):
            raise TypeError(f"remote must be a TrainingComponent or None, got {type(self.remote).__name__}")

    def check(self, files: ArtifactFiles) -> None:
        names = self.facts.check_artifact(files)
        if names is None:
            return
        if self.remote is None:
            raise InternalError(
                f"this process has no way to the {self.facts.name} training service, so it checks no imported "
                "artifact; the control service checks an import"
            )
        self.remote.check_remote_artifact(names)

    @property
    def resolves_references(self) -> bool:
        return self.facts.artifact_format is not None


class TrainingComponentFactory(ConfiguredFactory):
    """Build the training component that a training deployment's component config describes.

    ``kind`` is the dotted reference ``package.module:factory`` that the
    config's ``type`` holds (``TrainingDeployment.component_type``).
    ``__call__`` receives ``(config, model_path, environ)`` and returns a
    :class:`TrainingComponent`; ``build_training_component`` calls it. ``facts`` receives the same arguments and
    returns the backend's :class:`TrainingBackendFacts` without building a component; ``training_backend_facts``
    calls it.

    Every factory defines both. The facts are the one source of a new scenario's default start
    (``TrainingBackendFacts.default_start``) and of the check of an item root (``check_root_start``): Reef asks no
    component for a start's files, and a ``TrainingComponent`` has no ``default_start``. The component only starts
    jobs from a root that holds those files (``TrainingComponent.root_start``).
    """

    @abstractmethod
    def __call__(self, config: Mapping[str, Any], model_path: str, environ: Mapping[str, str]) -> TrainingComponent:
        """Build the training component that ``config`` describes."""

    @abstractmethod
    def facts(self, config: Mapping[str, Any], model_path: str, environ: Mapping[str, str]) -> TrainingBackendFacts:
        """The facts of the component that ``__call__`` builds from the same arguments, without building it.

        It starts nothing and imports no training framework. Every process but the training service reads them
        in place of a component, and control writes a new scenario's root from them.
        """


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


def training_backend_facts(
    config: Mapping[str, Any], *, model_path: str, environ: Mapping[str, str] | None = None
) -> TrainingBackendFacts:
    """The facts of the training component of ``config``, without the component.

    ``config`` is a component config as ``build_training_component`` takes it: its ``type`` names the factory, whose
    ``facts`` gets the parsed config. ``TypeError`` when the factory returns anything but a ``TrainingBackendFacts``.
    """
    reference = config.get("type")
    if not isinstance(reference, str) or not reference:
        raise RuntimeConfigError("a training component config needs its factory's dotted reference as type")
    factory = training_component_factory_for(reference)
    values = os.environ if environ is None else environ
    facts = factory.facts(factory.parse_config(config, values), model_path, values)
    if not isinstance(facts, TrainingBackendFacts):
        raise TypeError(f"training component type {reference!r} must state TrainingBackendFacts")
    return facts


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
