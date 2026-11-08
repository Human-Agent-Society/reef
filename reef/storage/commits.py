"""Scenario registration metadata, commit records, and their persisted formats.

Checkpoint recovery uses CommitRecord, the same value stored in the commit history.
Initial registration has no commit. Encoding and validation depend only on core
value types; training and storage operations remain with their callers.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from reef.core.artifact_selection import ArtifactSelection
from reef.core.errors import ReefError
from reef.core.snapshot import RootSource, SnapshotRef, decode_snapshot_ref, encode_snapshot_ref


@dataclass(frozen=True)
class RecordProgress:
    """Record-consumption watermark pinned by a commit record.

    ``consumed_ids`` names the rows the step's job data consumed.
    """

    high_water_sequence: int
    high_water_offset: int
    consumed_ids: frozenset[str] = frozenset()


def read_consumed_ids(value: Mapping[str, object], *, context: str) -> frozenset[str]:
    """Normalize old progress into the current consumption-only contract."""
    consumed: set[str] = set()
    for key in ("compacted_ids", "consumed_ids"):
        ids = value.get(key, [] if key == "compacted_ids" else None)
        if not isinstance(ids, list) or any(not isinstance(record_id, str) for record_id in ids):
            raise ValueError(f"{context} record_progress.{key} must be a list of strings")
        consumed.update(ids)
    return frozenset(consumed)


def parse_record_progress(value: object, *, context: str) -> RecordProgress:
    """Validate one record_progress mapping; shared by checkpoint metadata and commit-history parsing.

    ``context`` prefixes every error message (e.g. ``"scenario metadata"`` or
    ``"commit record"``). Raises ``ValueError``; callers with their own error
    types translate it.
    """
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} record_progress must be an object")
    for name in ("high_water_sequence", "high_water_offset"):
        field = value.get(name)
        if not isinstance(field, int) or isinstance(field, bool) or field < 0:
            raise ValueError(f"{context} record_progress.{name} must be a non-negative integer")
    return RecordProgress(
        high_water_sequence=value["high_water_sequence"],
        high_water_offset=value["high_water_offset"],
        consumed_ids=read_consumed_ids(value, context=context),
    )


RECORD_KIND = "reef-commit/8"
#: The ``snapshot`` of a record read from storage until ``CommitSnapshots`` locates it: a record with a
#: ``selection`` stores no storage location.
UNLOCATED_SNAPSHOT = SnapshotRef("unlocated:", None)


class CommitHistoryError(ReefError):
    """The commit history is corrupt or a record does not match the schema."""


def to_json_value(value: Any) -> Any:
    """Return ``value`` as a JSON round trip gives it back: tuples become lists, and mapping keys become strings.

    Raises ``CommitHistoryError`` when ``value`` holds something JSON cannot encode.
    """
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise CommitHistoryError(f"commit record values must be JSON values: {exc}") from exc


#: What a commit record does: a training step, the activation of an artifact, a trainer's new training start, an
#: import, a sync of a person's tree that is active and the training start at once, or the start or the end of
#: serving an artifact beside the active selection.
CommitOperation = Literal["training", "activate", "restore", "train_from", "import", "sync", "serve", "stop_serving"]


@dataclass(frozen=True, kw_only=True, eq=False)
class CommitRecord:
    """One committed step of a scenario: a training step, an activation, a new training start, an import, a sync,
    or a change of the artifacts served beside the active selection.

    Values: ``scenario`` is a non-empty name and ``step`` a positive int.
    ``snapshot`` is where the record's release is stored, for use in this process. ``to_dict`` stores it only for a
    record without a ``selection``. A record that ``from_dict`` reads without one holds ``UNLOCATED_SNAPSHOT`` until
    ``CommitSnapshots`` locates it.
    ``recorded_at`` is the time of the commit in seconds. ``operation`` is a
    ``CommitOperation``. ``operation_verified`` is false only for a commit
    read from release metadata that names no operation. ``component`` is
    ``None`` or a non-empty name, and ``components`` is ``None`` or a
    mapping of non-empty component names to non-empty artifact ids
    (``SnapshotComponents.artifact_ids``).
    ``selection`` is None for stored artifacts or an ``ArtifactSelection``
    mapping component names to the artifact IDs needed to rebuild a local view. When
    ``components`` is present, it equals the selection's artifact ids.
    ``high_water_sequence`` and ``high_water_offset`` are non-negative ints.
    Each operation adds its own conditions:

    - ``"training"``: one trainer's step. ``snapshot`` is the snapshot
      the step made when ``checkpoint`` is true, and the active selection when
      the step made none. ``activated`` is true exactly when the scenario's
      activation rule activated the snapshot that the step made. A later
      activation is an ``"activate"`` record. ``algorithm_state``, the
      high-water values, ``consumed_ids``, ``metrics`` and
      ``training_job_id`` are the
      trainer's state and the step's outcome. ``component`` names the
      trainer in a scenario of several components.
    - ``"activate"``: ``snapshot`` is the active view after selecting
      an existing artifact. One component uses its stored artifact directly,
      with ``component`` and ``components`` both None. Several components
      use a local view and persist its complete ``selection``; ``component``
      names the changed component and ``components`` lists every artifact.
    - ``"restore"``: restore a previous complete active selection.
      ``component`` is None, ``selection`` is present, and ``components``
      lists its artifact ids. The operation creates no artifact.
    - ``"train_from"``: ``snapshot`` is the snapshot that holds the new
      training start, as the row that made it records it. ``component``
      names the trainer as its training rows do (``None`` for a lone
      trainer), and ``components`` is ``None``.
    - ``"import"``: ``snapshot`` is a snapshot that the import published
      beside the chain: it binds the imported artifact and the active
      selection's other components. ``checkpoint`` is true. ``component`` and
      ``components`` are as for an ``"activate"`` record: ``None`` in a
      scenario of one component, else the imported component and the
      snapshot's manifest. ``metrics`` is ``{"source": {"uri", "version"}}``,
      where the content was read from, and ``source.item`` names the
      artifact item of an import of one; it is no part of the artifact's
      identity.
    - ``"sync"``: ``snapshot`` is a snapshot that the sync published
      beside the chain from a person's merged tree. ``checkpoint`` is true.
      ``component`` and ``components`` are as for an ``"import"`` record.
      In this one commit the snapshot becomes the active selection and the
      training start of the component's trainer. ``metrics`` is the sync's
      own description (``None`` or an object), not trainer state.
    - ``"serve"``: Reef serves one artifact beside the active selection from
      now on. ``snapshot`` is the snapshot that serves it, and the served
      artifact is the one that it binds as ``component``. In a scenario of
      one component, it is the snapshot that made the artifact, and
      ``component`` and ``components`` are ``None``. In a scenario of
      several, it is a local view with the complete ``selection``, and
      ``component`` and ``components`` are as for an ``"activate"`` record.
    - ``"stop_serving"``: Reef no longer serves the artifact of the
      ``"serve"`` record with the same ``snapshot``, ``component`` and
      ``components``.

    An ``"activate"``, ``"restore"``, ``"train_from"``, ``"import"``, ``"sync"``, ``"serve"`` or
    ``"stop_serving"`` record carries no trainer state and is not
    ``activated``: ``algorithm_state`` and
    ``training_job_id`` are ``None``, ``metrics`` is ``None`` except for an
    import's source and a sync's own metrics, both high-water values are 0 and
    ``consumed_ids`` is empty. Only an ``"import"`` or ``"sync"`` record among them has
    ``checkpoint`` set.

    ``algorithm_state`` and ``metrics`` hold JSON values: the constructor
    converts them as a JSON round trip does, so a record equals itself after
    storage. The constructor raises ``CommitHistoryError`` for a value
    outside these conditions.

    Purpose: the scenario committer makes the records and appends them to the
    commit history. Recovery reads them back: trainer state from training
    records, the active selection from :func:`active_record`, and each
    trainer's training start from its newest record that
    ``sets_training_start``.
    """

    scenario: str
    step: int
    snapshot: SnapshotRef
    checkpoint: bool
    algorithm_state: Mapping[str, Any] | None
    high_water_sequence: int
    high_water_offset: int
    consumed_ids: frozenset[str] = frozenset()
    recorded_at: float = field(default_factory=time.time)
    operation: CommitOperation = "training"
    operation_verified: bool = True
    activated: bool = False
    metrics: Mapping[str, Any] | None = None
    training_job_id: str | None = None
    #: The component of the trainer that committed this step or takes this training start, or the activated
    #: component; ``None`` in a scenario of one component.
    component: str | None = None
    #: The artifact id of every component the published snapshot or local view binds, by
    #: name; ``None`` for a flat snapshot and for a step that published none.
    components: Mapping[str, str] | None = None
    #: Complete component selection for a locally materialized view. These IDs survive cache loss.
    selection: ArtifactSelection | None = None

    def __post_init__(self) -> None:
        if self.selection is not None:
            if not isinstance(self.selection, ArtifactSelection):
                raise CommitHistoryError("commit record selection must be an ArtifactSelection")
            if self.components is not None and self.components != self.selection.artifact_ids:
                raise CommitHistoryError("commit record components must match its selection")
        if self.snapshot.selection_view and self.selection is None:
            raise CommitHistoryError("a materialized selection requires its component artifact IDs")
        if not isinstance(self.scenario, str) or not self.scenario:
            raise CommitHistoryError("commit record scenario must be a non-empty string")
        if not isinstance(self.step, int) or isinstance(self.step, bool) or self.step < 1:
            raise CommitHistoryError("commit record step must be a positive integer")
        if self.component is not None and (not isinstance(self.component, str) or not self.component):
            raise CommitHistoryError("commit record component must be a non-empty string or null")
        if self.components is not None and (
            not isinstance(self.components, Mapping)
            or not all(
                isinstance(name, str) and name and isinstance(artifact_id, str) and artifact_id
                for name, artifact_id in self.components.items()
            )
        ):
            raise CommitHistoryError("commit record components must map component names to artifact IDs")
        for name, value in (
            ("high_water_sequence", self.high_water_sequence),
            ("high_water_offset", self.high_water_offset),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise CommitHistoryError(f"commit record {name} must be a non-negative integer")
        if self.operation not in get_args(CommitOperation):
            raise CommitHistoryError(f"commit record operation must be one of {get_args(CommitOperation)}")
        if not isinstance(self.operation_verified, bool):
            raise CommitHistoryError("commit record operation_verified must be a boolean")
        if not isinstance(self.checkpoint, bool) or not isinstance(self.activated, bool):
            raise CommitHistoryError("commit record checkpoint and activated must be booleans")
        if self.activated and not (self.operation == "training" and self.checkpoint):
            raise CommitHistoryError("only a training commit that made an artifact is activated with it")
        if self.metrics is not None and not isinstance(self.metrics, Mapping):
            raise CommitHistoryError("commit record metrics must be an object or null")
        if self.training_job_id is not None and (
            not isinstance(self.training_job_id, str) or not self.training_job_id
        ):
            raise CommitHistoryError("commit record training_job_id must be a non-empty string or null")
        if self.operation != "training":
            publishes = self.operation in ("import", "sync")
            if self.checkpoint != publishes:
                made = "makes an artifact" if publishes else "makes no artifact"
                raise CommitHistoryError(f"a {self.operation} commit {made}: its checkpoint is {publishes}")
            carried = [
                name
                for name, value in (
                    ("algorithm_state", self.algorithm_state is not None),
                    ("metrics", self.metrics is not None and not publishes),
                    ("training_job_id", self.training_job_id is not None),
                    ("high_water_sequence", self.high_water_sequence != 0),
                    ("high_water_offset", self.high_water_offset != 0),
                    ("consumed_ids", bool(self.consumed_ids)),
                )
                if value
            ]
            if carried:
                raise CommitHistoryError(
                    f"a {self.operation} commit carries no trainer state and must not carry {carried}"
                )
        if self.operation == "import":
            source = None if self.metrics is None else self.metrics.get("source")
            if (
                self.metrics is None
                or set(self.metrics) != {"source"}
                or not isinstance(source, Mapping)
                or set(source) not in ({"uri", "version"}, {"uri", "version", "item"})
                or not all(isinstance(value, str) for value in source.values())
                or not source["uri"]
                or source.get("item") == ""
            ):
                raise CommitHistoryError(
                    'an import commit\'s metrics are {"source": {"uri", "version"}}, with "item" for an artifact item'
                )
        if self.operation == "train_from" and self.components is not None:
            raise CommitHistoryError("a train_from commit publishes no snapshot and carries no components")
        if self.operation == "restore" and (
            self.component is not None or self.selection is None or self.components != self.selection.artifact_ids
        ):
            raise CommitHistoryError(
                "a restore commit requires a complete selection and names no individual component"
            )
        if self.operation in ("activate", "import", "sync", "serve", "stop_serving") and (
            (self.component is None) != (self.components is None)
            or (self.components is not None and self.component not in self.components)
        ):
            raise CommitHistoryError(
                f"a commit of operation {self.operation!r} names a component exactly when it carries the manifest "
                "that binds it"
            )
        # Hold state and metrics as a stored record reads them back, so a record
        # equals itself after storage, and a caller mutating its values
        # afterwards does not change what was recorded.
        object.__setattr__(
            self,
            "algorithm_state",
            None if self.algorithm_state is None else to_json_value(dict(self.algorithm_state)),
        )
        object.__setattr__(self, "metrics", None if self.metrics is None else to_json_value(dict(self.metrics)))
        object.__setattr__(self, "consumed_ids", frozenset(self.consumed_ids))
        object.__setattr__(self, "components", None if self.components is None else dict(self.components))

    @property
    def artifact_ids(self) -> Mapping[str, str] | None:
        """The artifact ID of each component that ``snapshot`` binds, from ``selection`` or ``components``.

        None for a record of a local snapshot, which names no component IDs.
        """
        return self.selection.artifact_ids if self.selection is not None else self.components

    @property
    def made_artifact(self) -> bool:
        """Whether ``snapshot`` made a new artifact: a training step that saved a checkpoint, an import, or a
        sync."""
        return self.checkpoint

    @property
    def trained_artifact(self) -> bool:
        """Whether a training step saved this record's artifact: its snapshot is the new chain tip, and the
        training start moves to it."""
        return self.operation == "training" and self.checkpoint

    @property
    def sets_training_start(self) -> bool:
        """Whether ``snapshot`` becomes the training start of the record's trainer: a new training start, a
        sync, or a training step that saved an artifact."""
        return self.operation in ("train_from", "sync") or self.trained_artifact

    @property
    def activates(self) -> bool:
        """Whether ``snapshot`` is the active selection after this record: an activation, a sync, or a training
        step whose snapshot the activation rule activated."""
        return self.operation in ("activate", "restore", "sync") or self.activated

    def to_dict(self) -> dict[str, Any]:
        record_progress: dict[str, Any] = {
            "high_water_sequence": self.high_water_sequence,
            "high_water_offset": self.high_water_offset,
        }
        record_progress["consumed_ids"] = sorted(self.consumed_ids)
        value = {
            "record": RECORD_KIND,
            "scenario": self.scenario,
            "step": self.step,
            "checkpoint": self.checkpoint,
            "algorithm_state": self.algorithm_state,
            "record_progress": record_progress,
            "recorded_at": self.recorded_at,
        }
        if self.operation != "training":
            value["operation"] = self.operation
        if not self.operation_verified:
            value["operation_verified"] = False
        if self.activated:
            value["activated"] = True
        if self.metrics is not None:
            value["metrics"] = deepcopy(self.metrics)
        if self.training_job_id is not None:
            value["training_job_id"] = self.training_job_id
        if self.component is not None:
            value["component"] = self.component
        if self.components is not None:
            value["components"] = dict(self.components)
        if self.selection is not None:
            value["selection"] = self.selection.to_dict()
        else:
            value["snapshot"] = encode_snapshot_ref(self.snapshot)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CommitRecord:
        if value.get("record") != RECORD_KIND:
            raise CommitHistoryError(
                f"commit record kind {value.get('record')!r} is not {RECORD_KIND}: this Reef reads no older record"
            )
        scenario = value.get("scenario")
        if not isinstance(scenario, str) or not scenario:
            raise CommitHistoryError("commit record requires scenario")
        step = value.get("step")
        if not isinstance(step, int):
            raise CommitHistoryError("commit record step must be an integer")
        raw_ref = value.get("snapshot")
        if not isinstance(raw_ref, Mapping) and (raw_ref is not None or value.get("selection") is None):
            raise CommitHistoryError("commit record requires snapshot or selection")
        checkpoint = value.get("checkpoint")
        if not isinstance(checkpoint, bool):
            raise CommitHistoryError("commit record checkpoint must be a boolean")
        algorithm_state = value.get("algorithm_state")
        if algorithm_state is not None and not isinstance(algorithm_state, Mapping):
            raise CommitHistoryError("commit record algorithm_state must be an object or null")
        raw_progress = value.get("record_progress")
        if not isinstance(raw_progress, Mapping):
            raise CommitHistoryError("commit record requires record_progress")
        try:
            snapshot = UNLOCATED_SNAPSHOT if raw_ref is None else decode_snapshot_ref(raw_ref)
            selection = None if value.get("selection") is None else ArtifactSelection.from_dict(value["selection"])
        except ValueError as exc:
            raise CommitHistoryError(f"commit record {exc}") from exc
        try:
            record_progress = parse_record_progress(raw_progress, context="commit record")
        except ValueError as exc:
            raise CommitHistoryError(str(exc)) from exc
        recorded_at = value.get("recorded_at")
        if not isinstance(recorded_at, int | float) or isinstance(recorded_at, bool):
            raise CommitHistoryError("commit record recorded_at must be a number")
        operation = value.get("operation", "training")
        operation_verified = value.get("operation_verified", True)
        return cls(
            activated=value.get("activated", False),
            scenario=scenario,
            step=step,
            snapshot=snapshot,
            checkpoint=checkpoint,
            algorithm_state=algorithm_state,
            high_water_sequence=record_progress.high_water_sequence,
            high_water_offset=record_progress.high_water_offset,
            consumed_ids=record_progress.consumed_ids,
            recorded_at=float(recorded_at),
            operation=operation,
            operation_verified=operation_verified,
            metrics=value.get("metrics"),
            training_job_id=value.get("training_job_id"),
            component=value.get("component"),
            components=value.get("components"),
            selection=selection,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CommitRecord):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        return f"CommitRecord(scenario={self.scenario!r}, step={self.step}, release={self.snapshot.revision!r})"


class CommitSnapshots(ABC):
    """Finds the storage snapshot of each record read from a commit history.

    Purpose: storage holds no storage location, so the scenario factory gives a store this interface.
    ``locate(records)`` takes a commit history in step order. It returns the same records in the same order, each
    with its ``snapshot``, and leaves a record that holds a snapshot as it is. It raises ``CommitHistoryError`` when
    a record's artifact IDs do not name a snapshot.
    """

    @abstractmethod
    def locate(self, records: Sequence[CommitRecord]) -> tuple[CommitRecord, ...]: ...


def encode_commit(record: CommitRecord) -> str:
    """The JSON text that a commit history stores for ``record``."""
    return json.dumps(record.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def active_record(records: Sequence[CommitRecord]) -> CommitRecord | None:
    """The newest record of ``records``, a commit history in step order, that activates a snapshot; ``None`` when
    no record does (the creation artifact is active)."""
    return next((record for record in reversed(records) if record.activates), None)


def active_snapshot(records: Sequence[CommitRecord], fallback: SnapshotRef) -> SnapshotRef:
    """The snapshot that is active after ``records``: the one :func:`active_record` names, else ``fallback``.

    ``fallback`` is the snapshot that is active when no record activates one:
    the creation artifact, or, for a store that keeps no history across a
    restart, the snapshot that the artifact head's metadata names
    (:func:`served_snapshot`).
    """
    record = active_record(records)
    return fallback if record is None else record.snapshot


SCENARIO_METADATA_KEY = "scenario_commit_record"
SCENARIO_METADATA_KIND = "reef-scenario/7"


def scenario_metadata_for(
    *,
    name: str,
    source: RootSource,
    start_artifact: SnapshotRef | None = None,
    scenario_step: int = 0,
    algorithm_state: Mapping[str, Any] | None = None,
    record_progress: RecordProgress | None = None,
    metrics: Mapping[str, Any] | None = None,
    training_job_id: str | None = None,
    component: str | None = None,
    active_artifact: SnapshotRef | None = None,
    active_selection: ArtifactSelection | None = None,
) -> dict[str, object]:
    """The scenario metadata of a registration (step zero) or of the artifact a training step made.

    ``source`` is where the scenario's root snapshot came from. ``start_artifact`` is the root snapshot. The root
    snapshot carries the registration itself, so a registration names no ``start_artifact`` and every later step
    names it. ``active_artifact`` is None when the activation rule activated the step's own artifact. Otherwise it names
    the artifact that serves after the step. A recovery without the commit history reads it to find the active
    artifact.
    """
    if not isinstance(scenario_step, int) or isinstance(scenario_step, bool) or scenario_step < 0:
        raise ValueError("scenario_step must be non-negative")
    if (start_artifact is None) != (scenario_step == 0):
        raise ValueError("scenario metadata names its start_artifact exactly after step zero")
    if active_artifact is not None and scenario_step == 0:
        raise ValueError("a registration's own artifact serves; it names no active_artifact")
    metadata: dict[str, object] = {
        "format": SCENARIO_METADATA_KIND,
        "scenario": name,
        "scenario_step": scenario_step,
        "source": source.to_dict(),
        "operation": "training",
    }
    if start_artifact is not None:
        metadata["start_artifact"] = encode_snapshot_ref(start_artifact)
    # The trainer that made the checkpoint: a checkpoint adopted from the artifact head
    # after the commit history is lost must still be attributed to its component's trainer.
    if component is not None:
        if not isinstance(component, str) or not component:
            raise ValueError("scenario metadata component must be a non-empty string or None")
        metadata["component"] = component
    if algorithm_state is not None:
        metadata["algorithm_state"] = dict(algorithm_state)
    if record_progress is not None:
        metadata["record_progress"] = {
            "high_water_sequence": record_progress.high_water_sequence,
            "high_water_offset": record_progress.high_water_offset,
            "consumed_ids": sorted(record_progress.consumed_ids),
        }
    if training_job_id is not None:
        metadata["training_job_id"] = training_job_id
    if metrics is not None:
        metadata["metrics"] = deepcopy(dict(metrics))
    if active_artifact is not None:
        metadata["active_artifact"] = encode_snapshot_ref(active_artifact)
    if active_selection is not None:
        if active_artifact is None:
            raise ValueError("active_selection requires active_artifact")
        metadata["active_selection"] = active_selection.to_dict()
    return metadata


def parse_scenario_metadata(
    value: Mapping[str, Any], *, checkpoint_head: SnapshotRef, components: Mapping[str, str] | None = None
) -> tuple[str, SnapshotRef, RootSource, CommitRecord | None]:
    """Read the metadata ``value`` that the snapshot ``checkpoint_head`` carries: the scenario name, the root snapshot,
    the root's source, and the checkpoint commit (``None`` at step zero, which has no commit).

    At step zero ``checkpoint_head`` is the root snapshot. ``components`` is the manifest of ``checkpoint_head``,
    read from the same snapshot metadata, so a commit rebuilt from it carries what a recorded one does. Raise
    ``ValueError`` for metadata of another format, or a malformed value.
    """
    if value.get("format") != SCENARIO_METADATA_KIND:
        raise ValueError(
            f"scenario metadata has the format {value.get('format')!r}; this Reef reads {SCENARIO_METADATA_KIND}, "
            "so create the scenario again"
        )
    scenario = value.get("scenario")
    if not isinstance(scenario, str) or not scenario:
        raise ValueError("scenario metadata requires scenario")
    source = RootSource.from_dict(value.get("source"))
    scenario_step = value.get("scenario_step", 0)
    if not isinstance(scenario_step, int) or isinstance(scenario_step, bool) or scenario_step < 0:
        raise ValueError("scenario metadata scenario_step must be non-negative")
    algorithm_state = value.get("algorithm_state")
    if algorithm_state is not None and not isinstance(algorithm_state, Mapping):
        raise ValueError("scenario metadata algorithm_state must be an object")
    raw_progress = value.get("record_progress")
    record_progress: RecordProgress | None = None
    if raw_progress is not None:
        record_progress = parse_record_progress(raw_progress, context="scenario metadata")
    training_job_id = value.get("training_job_id")
    if training_job_id is not None and (not isinstance(training_job_id, str) or not training_job_id):
        raise ValueError("scenario metadata training_job_id must be a non-empty string or null")
    metrics = value.get("metrics")
    if metrics is not None and not isinstance(metrics, Mapping):
        raise ValueError("scenario metadata metrics must be an object or null")
    operation = value.get("operation")
    if operation is not None and operation != "training":
        raise ValueError("scenario metadata operation must be 'training'")
    component = value.get("component")
    if component is not None and (not isinstance(component, str) or not component):
        raise ValueError("scenario metadata component must be a non-empty string or null")
    raw_start = value.get("start_artifact")
    if scenario_step == 0:
        if raw_start is not None:
            raise ValueError("a registration's scenario metadata names no start_artifact: it is the root snapshot")
        return scenario, checkpoint_head, source, None
    if not isinstance(raw_start, Mapping):
        raise ValueError("scenario metadata after step zero requires start_artifact")
    try:
        start_artifact = decode_snapshot_ref(raw_start)
    except ValueError as exc:
        raise ValueError(f"invalid scenario metadata start_artifact: {exc}") from exc
    if record_progress is None:
        raise ValueError("scenario metadata requires record_progress after step zero")
    commit = CommitRecord(
        scenario=scenario,
        step=scenario_step,
        snapshot=checkpoint_head,
        checkpoint=True,
        algorithm_state=algorithm_state,
        high_water_sequence=record_progress.high_water_sequence,
        high_water_offset=record_progress.high_water_offset,
        consumed_ids=record_progress.consumed_ids,
        training_job_id=training_job_id,
        metrics=metrics,
        operation_verified=operation is not None,
        component=component,
        components=components,
        activated=value.get("active_artifact") is None,
    )
    return scenario, start_artifact, source, commit


def served_snapshot(value: Mapping[str, Any], *, checkpoint_head: SnapshotRef) -> SnapshotRef:
    """The artifact that serves after the commit whose scenario metadata is ``value``.

    ``checkpoint_head`` is the snapshot that carries ``value``. It serves when ``value`` names no
    ``active_artifact``: a registration, or a step whose artifact the activation rule activated. Otherwise the
    named artifact serves. Raise ``ValueError`` when ``active_artifact`` is not an artifact reference.
    """
    raw = value.get("active_artifact")
    if raw is None:
        return checkpoint_head
    if not isinstance(raw, Mapping):
        raise ValueError("scenario metadata active_artifact must be an object")
    try:
        return decode_snapshot_ref(raw)
    except ValueError as exc:
        raise ValueError(f"invalid scenario metadata active_artifact: {exc}") from exc


__all__ = [
    "RECORD_KIND",
    "SCENARIO_METADATA_KEY",
    "SCENARIO_METADATA_KIND",
    "UNLOCATED_SNAPSHOT",
    "CommitHistoryError",
    "CommitOperation",
    "CommitRecord",
    "CommitSnapshots",
    "RecordProgress",
    "active_record",
    "active_snapshot",
    "parse_record_progress",
    "parse_scenario_metadata",
    "scenario_metadata_for",
    "served_snapshot",
]
