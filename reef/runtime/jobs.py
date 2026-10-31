"""Bounded training jobs and the training components that run them.

Reef submits a ``TrainingJob`` to a ``TrainingComponent``. The component runs
its framework's own training loop over the job's data, starting from the job's
start, and saves checkpoints. Reef reads the job's ``JobStatus``, evaluates each
saved checkpoint as a candidate, and serves the ones that pass. The job's
``parameters`` belong to the backend: Reef reads none of them, and they
override the backend's own defaults. A component keeps its own training state,
such as an objective's counters or an optimizer's search state, inside the
checkpoints it saves, so a job that starts from a checkpoint continues it.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from reef.artifact.files import ArtifactFiles
from reef.artifact.repository import BaseArtifactSource
from reef.core.job_data import JobData
from reef.core.weight_layout import ADAPTER_DIR, WEIGHT_LAYOUTS, WeightLayout
from reef.surface.base import ArtifactLayout

JobState = Literal["running", "finished", "failed"]


@dataclass(frozen=True)
class TrainingJob:
    """One bounded training job: a data snapshot, the weights it starts from, and the backend's parameters.

    A value has a non-empty ``job_id``, a non-empty ``backend`` name, ``start``
    (the non-empty start that the component reads, ``TrainingComponent.base_start`` for the base artifact),
    ``data`` (the job data that the job trains; the component cuts it into
    optimizer batches by its own batch size)
    and ``parameters``, a mapping that encodes as JSON.

    Reef makes the value and names the job with ``job_id_for``, so a retry of
    the same request has the same ID. A component reads it in ``submit``.
    """

    job_id: str
    backend: str
    start: str
    data: JobData
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, str) or not self.job_id:
            raise ValueError("TrainingJob.job_id must be a non-empty string")
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("TrainingJob.backend must be a non-empty string")
        if not isinstance(self.start, str) or not self.start:
            raise ValueError("TrainingJob.start must be a non-empty string")
        if not isinstance(self.data, JobData):
            raise TypeError("TrainingJob.data must be a JobData")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("TrainingJob.parameters must be a mapping")
        json.dumps(dict(self.parameters), allow_nan=False)


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
    can start from. ``release_path`` is the non-empty directory that Reef
    publishes as a release when it serves the checkpoint: the weights that
    serving loads, without the training state that only the component reads
    (for example ``path`` itself, or its Hugging Face export).
    ``metrics`` is the backend's telemetry for the training that led to this
    checkpoint, a mapping that encodes as JSON, which Reef records without
    reading. A component makes the value; ``JobLoop`` reads it.
    """

    step: int
    path: str
    release_path: str
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.step, int) or isinstance(self.step, bool) or self.step < 0:
            raise ValueError("SavedCheckpoint.step must be a non-negative int")
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("SavedCheckpoint.path must be a non-empty string")
        if not isinstance(self.release_path, str) or not self.release_path:
            raise ValueError("SavedCheckpoint.release_path must be a non-empty string")
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
class ReleaseFormat:
    """The weight files that each release of a training component holds, which a serving runtime loads.

    ``layout`` is ``"full"``, a Hugging Face checkpoint at the release's root,
    or ``"adapter"``, a PEFT adapter in the release's ``adapter/`` directory
    (the layouts of ``WeightFiles``). ``adapter_rank`` is the adapter's
    positive LoRA rank, and ``adapter_target_modules`` the non-empty names of
    the modules it adapts in the serving engine's naming (``("all",)`` for
    every linear module); both are set exactly for ``"adapter"``. A training
    component declares the value (``TrainingComponent.release_format``), and
    the serving assembly reads it to configure the engines and to refuse a
    serving runtime that cannot load the releases.
    """

    layout: WeightLayout
    adapter_rank: int | None = None
    adapter_target_modules: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.layout not in WEIGHT_LAYOUTS:
            raise ValueError(f"ReleaseFormat.layout must be 'full' or 'adapter', got {self.layout!r}")
        if self.layout == "full":
            if self.adapter_rank is not None or self.adapter_target_modules:
                raise ValueError("a full-weight ReleaseFormat has no adapter rank or target modules")
            return
        rank = self.adapter_rank
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError(f"an adapter ReleaseFormat needs a positive int adapter_rank, got {rank!r}")
        modules = self.adapter_target_modules
        if (
            not isinstance(modules, tuple)
            or not modules
            or any(not isinstance(name, str) or not name for name in modules)
        ):
            raise ValueError("an adapter ReleaseFormat needs a non-empty tuple of module names")

    def to_dict(self) -> dict[str, Any]:
        """The value as a JSON object, which ``from_dict`` reads back to an equal value."""
        return {
            "layout": self.layout,
            "adapter_rank": self.adapter_rank,
            "adapter_target_modules": list(self.adapter_target_modules),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReleaseFormat:
        """Read a ``to_dict`` value; raise ``ValueError`` for another shape."""
        if (
            not isinstance(value, Mapping)
            or "layout" not in value
            or set(value) - {"layout", "adapter_rank", "adapter_target_modules"}
        ):
            raise ValueError("a release format is an object of layout, adapter_rank and adapter_target_modules")
        modules = value.get("adapter_target_modules", ())
        if not isinstance(modules, (list, tuple)):
            raise ValueError("a release format's adapter_target_modules is a list of module names")
        return cls(value["layout"], value.get("adapter_rank"), tuple(modules))


class TrainingComponent(ABC):
    """A training backend that runs whole jobs with its framework's own training loop.

    A component never learns which scenario a job is for: a job carries
    everything it trains (its start, data and parameters). ``submit`` records
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
    def release_format(self) -> ReleaseFormat | None:
        """The weight files that each saved checkpoint's release holds.

        ``None`` means the releases hold no weight files that an engine
        loads: only the backend's own serving runtime reads them (for
        example a Tinker manifest that Tinker's sampler serves).
        """

    @abstractmethod
    def base_artifact_source(self) -> BaseArtifactSource:
        """The component's untrained checkpoint, laid out as one of its releases, and the configuration it comes from.

        Reef imports it as the base artifact of the component that this
        component trains. It has the layout of every other release
        (``release_format``), so serving and training read it as they read
        any release. Its ``inputs`` name the model and the settings that
        decide its content.
        """

    @property
    @abstractmethod
    def base_start(self) -> str:
        """The start of a job from the untrained checkpoint; the first job of every trainer starts from it."""

    @property
    def starts_from_releases(self) -> bool:
        """Whether ``import_start`` can take a published release as the start of later jobs.

        False exactly when the component refuses every release, so no trainer of it can train from an artifact.
        True, the default, when it takes some releases; ``import_start`` still refuses one whose weights it cannot
        start from.
        """
        return True

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

    def for_scenario(self, scenario: str) -> TrainingComponent:
        """The component that runs the jobs of ``scenario``. The default is this component.

        ``JobLoop`` calls it once. A queue-backed component returns a view that adds each job to the jobs queue
        under ``scenario``.
        """
        return self

    def check_task_jobs(self, parameters: Mapping[str, Any]) -> None:
        """Raise ``ValueError`` unless the component can train jobs whose data are ``TaskSample`` values, with ``parameters``.

        A recipe whose jobs train tasks calls it when it is built, so a deployment that cannot run them fails before
        any job. ``parameters`` are the recipe's job parameters without the method. The default refuses: the
        component trains trajectories only. ``check_job`` still checks each job.
        """
        raise ValueError(f"the {self.name} training component trains no task jobs")

    def check_release(self, files: ArtifactFiles) -> None:
        """Raise ``ValueError`` unless ``files`` are a release in the layout of this component's releases.

        An artifact import calls it (RFC #980) on the directory it copied or on the revision it keeps by reference
        (RFC #995). The default reads ``release_format``: ``"full"`` needs ``config.json`` and weight files at the
        root; ``"adapter"`` needs a PEFT adapter in ``adapter/`` whose rank is at most ``adapter_rank`` and whose
        target modules are within ``adapter_target_modules``. It reads only file names and
        ``adapter/adapter_config.json``, so a revision is checked without downloading its weights. ``None``
        refuses every release. A component whose releases name weights on a service overrides it and also checks
        that each path resolves there.
        """
        release_format = self.release_format
        if release_format is None:
            raise ValueError(f"the {self.name} training component takes no imported releases")
        paths = set(files.paths())
        if release_format.layout == "full":
            if "config.json" not in paths or not any(
                "/" not in path and path.endswith((".safetensors", ".bin")) for path in paths
            ):
                raise ValueError("a full-weight release holds config.json and weight files at its root")
            return
        if not paths & {f"{ADAPTER_DIR}/adapter_model.safetensors", f"{ADAPTER_DIR}/adapter_model.bin"}:
            raise ValueError(f"an adapter release holds a PEFT adapter in {ADAPTER_DIR}/")
        try:
            config = json.loads(files.read_bytes(f"{ADAPTER_DIR}/adapter_config.json"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"{ADAPTER_DIR}/adapter_config.json is no readable JSON object: {exc}") from exc
        rank = config.get("r") if isinstance(config, dict) else None
        # ReleaseFormat's invariant: an adapter format has a rank.
        max_rank = cast(int, release_format.adapter_rank)
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 < rank <= max_rank:
            raise ValueError(f"the adapter's rank must be a positive int at most {max_rank}")
        targets = release_format.adapter_target_modules
        modules = config.get("target_modules")
        if targets != ("all",) and (not isinstance(modules, list) or not set(modules) <= set(targets)):
            raise ValueError(f"the adapter's target modules must be a list within {list(targets)}, got {modules!r}")

    @abstractmethod
    def import_start(self, release: Path) -> str:
        """Take the published release in ``release`` as the start of later jobs; return that start.

        Reef calls this when a trainer starts training from an artifact
        (``train-from``). The component copies what it trains
        from into its own durable directory, since ``release`` can be
        discarded later. Raise ``ValueError``, and change nothing, when the
        component cannot start from these weights. A component whose
        ``starts_from_releases`` is False refuses every release.
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
class ReleaseLayout(ArtifactLayout):
    """The layout of the releases of ``component``: an import of the component it trains must meet it.

    ``check`` is ``component.check_release``. The releases resolve a reference when they hold weight files
    (``release_format``), which the serving runtimes read through ``ReleaseFiles``. A weight recipe binds it on its
    ``weights`` component.
    """

    component: TrainingComponent

    def check(self, files: ArtifactFiles) -> None:
        self.component.check_release(files)

    @property
    def resolves_references(self) -> bool:
        return self.component.release_format is not None
