"""Scenario registration metadata, commit records, and their persisted formats.

Checkpoint recovery uses CommitRecord, the same value stored in the commit history.
Initial registration has no commit. Encoding and validation depend only on core
value types; training and storage operations remain with their callers.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

from reef.core.artifact_ref import ArtifactRef, decode_artifact_ref, encode_artifact_ref
from reef.core.errors import ReefError


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


RECORD_KIND = "reef-commit/5"


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


#: What a commit record does: a training step, the activation of an artifact, or a trainer's new training start.
CommitOperation = Literal["training", "activate", "train_from"]


@dataclass(frozen=True, kw_only=True, eq=False)
class CommitRecord:
    """One committed step of a scenario: a training step, an activation, or a new training start.

    Values: ``scenario`` is a non-empty name and ``step`` a positive int.
    ``recorded_at`` is the time of the commit in seconds. ``operation`` is a
    ``CommitOperation``. ``operation_verified`` is false only for a commit
    read from release metadata that names no operation. ``component`` is
    ``None`` or a non-empty name, and ``components`` is ``None`` or a
    mapping of non-empty component names to non-empty content ids.
    ``high_water_sequence`` and ``high_water_offset`` are non-negative ints.
    Each operation adds its own conditions:

    - ``"training"``: one trainer's step. ``artifact_ref`` is the release
      the step made when ``checkpoint`` is true, and the active release when
      the step made none. ``pending`` is true only for a release that the
      step did not activate. ``algorithm_state``, the high-water values,
      ``consumed_ids``, ``metrics`` and ``training_job_id`` are the
      trainer's state and the step's outcome. ``component`` names the
      trainer in a scenario of several components.
    - ``"activate"``: ``artifact_ref`` is the release that is active after
      the activation. In a scenario of one component it is the release that
      made the activated artifact, as that row records it, and
      ``component`` and ``components`` are ``None``. In a scenario of
      several components it is a release that the activation published:
      ``component`` names the activated component, and ``components`` is
      the release's manifest, which contains ``component``.
    - ``"train_from"``: ``artifact_ref`` is the release that holds the new
      training start, as the row that made it records it. ``component``
      names the trainer as its training rows do (``None`` for a lone
      trainer), and ``components`` is ``None``.

    An ``"activate"`` or ``"train_from"`` record makes no artifact and
    carries no trainer state: ``checkpoint`` and ``pending`` are false,
    ``algorithm_state``, ``metrics`` and ``training_job_id`` are ``None``,
    both high-water values are 0 and ``consumed_ids`` is empty.

    ``algorithm_state`` and ``metrics`` hold JSON values: the constructor
    converts them as a JSON round trip does, so a record equals itself after
    storage. The constructor raises ``CommitHistoryError`` for a value
    outside these conditions.

    Purpose: the scenario committer makes the records and appends them to the
    commit history. Recovery reads them back: trainer state from training
    records, the active release from :func:`active_record`, and each
    trainer's training start from its newest ``"train_from"`` record.
    """

    scenario: str
    step: int
    artifact_ref: ArtifactRef
    checkpoint: bool
    algorithm_state: Mapping[str, Any] | None
    high_water_sequence: int
    high_water_offset: int
    consumed_ids: frozenset[str] = frozenset()
    recorded_at: float = field(default_factory=time.time)
    operation: CommitOperation = "training"
    operation_verified: bool = True
    pending: bool = False
    metrics: Mapping[str, Any] | None = None
    training_job_id: str | None = None
    #: The component of the trainer that committed this step or takes this training start, or the activated
    #: component; ``None`` in a scenario of one component.
    component: str | None = None
    #: The content id of every component the published release binds, by
    #: name; ``None`` for a flat release and for a step that published none.
    components: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scenario, str) or not self.scenario:
            raise CommitHistoryError("commit record scenario must be a non-empty string")
        if not isinstance(self.step, int) or isinstance(self.step, bool) or self.step < 1:
            raise CommitHistoryError("commit record step must be a positive integer")
        if self.component is not None and (not isinstance(self.component, str) or not self.component):
            raise CommitHistoryError("commit record component must be a non-empty string or null")
        if self.components is not None and (
            not isinstance(self.components, Mapping)
            or not all(
                isinstance(name, str) and name and isinstance(content_id, str) and content_id
                for name, content_id in self.components.items()
            )
        ):
            raise CommitHistoryError("commit record components must map component names to content ids")
        for name, value in (
            ("high_water_sequence", self.high_water_sequence),
            ("high_water_offset", self.high_water_offset),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise CommitHistoryError(f"commit record {name} must be a non-negative integer")
        if self.operation not in ("training", "activate", "train_from"):
            raise CommitHistoryError("commit record operation must be 'training', 'activate' or 'train_from'")
        if not isinstance(self.operation_verified, bool):
            raise CommitHistoryError("commit record operation_verified must be a boolean")
        if not isinstance(self.checkpoint, bool) or not isinstance(self.pending, bool):
            raise CommitHistoryError("commit record checkpoint and pending must be booleans")
        if self.metrics is not None and not isinstance(self.metrics, Mapping):
            raise CommitHistoryError("commit record metrics must be an object or null")
        if self.training_job_id is not None and (
            not isinstance(self.training_job_id, str) or not self.training_job_id
        ):
            raise CommitHistoryError("commit record training_job_id must be a non-empty string or null")
        if self.pending and not (self.operation == "training" and self.checkpoint):
            raise CommitHistoryError("only a training commit that made an artifact may be pending")
        if self.operation != "training":
            carried = [
                name
                for name, value in (
                    ("checkpoint", self.checkpoint),
                    ("algorithm_state", self.algorithm_state is not None),
                    ("metrics", self.metrics is not None),
                    ("training_job_id", self.training_job_id is not None),
                    ("high_water_sequence", self.high_water_sequence != 0),
                    ("high_water_offset", self.high_water_offset != 0),
                    ("consumed_ids", bool(self.consumed_ids)),
                )
                if value
            ]
            if carried:
                raise CommitHistoryError(f"a {self.operation} commit makes no artifact and must not carry {carried}")
        if self.operation == "train_from" and self.components is not None:
            raise CommitHistoryError("a train_from commit publishes no release and carries no components")
        if self.operation == "activate" and (
            (self.component is None) != (self.components is None)
            or (self.components is not None and self.component not in self.components)
        ):
            raise CommitHistoryError(
                "an activate commit names a component exactly when it carries the manifest that binds it"
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
    def made_artifact(self) -> bool:
        """Whether this record made its artifact: a training step that saved a checkpoint."""
        return self.operation == "training" and self.checkpoint

    @property
    def activates(self) -> bool:
        """Whether ``artifact_ref`` is the active release after this record: an activation, or a training step that
        made a release and is not pending."""
        return self.operation == "activate" or (self.made_artifact and not self.pending)

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
            "artifact_ref": encode_artifact_ref(self.artifact_ref),
            "checkpoint": self.checkpoint,
            "algorithm_state": self.algorithm_state,
            "record_progress": record_progress,
            "recorded_at": self.recorded_at,
        }
        if self.operation != "training":
            value["operation"] = self.operation
        if not self.operation_verified:
            value["operation_verified"] = False
        if self.pending:
            value["pending"] = True
        if self.metrics is not None:
            value["metrics"] = deepcopy(self.metrics)
        if self.training_job_id is not None:
            value["training_job_id"] = self.training_job_id
        if self.component is not None:
            value["component"] = self.component
        if self.components is not None:
            value["components"] = dict(self.components)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CommitRecord:
        if value.get("record") != RECORD_KIND:
            raise CommitHistoryError(f"unknown commit record kind: {value.get('record')!r}")
        scenario = value.get("scenario")
        if not isinstance(scenario, str) or not scenario:
            raise CommitHistoryError("commit record requires scenario")
        step = value.get("step")
        if not isinstance(step, int):
            raise CommitHistoryError("commit record step must be an integer")
        raw_ref = value.get("artifact_ref")
        if not isinstance(raw_ref, Mapping):
            raise CommitHistoryError("commit record requires artifact_ref")
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
            artifact_ref = decode_artifact_ref(raw_ref)
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
        pending = value.get("pending", False)
        return cls(
            pending=pending,
            scenario=scenario,
            step=step,
            artifact_ref=artifact_ref,
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
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CommitRecord):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        return f"CommitRecord(scenario={self.scenario!r}, step={self.step}, release={self.artifact_ref.release_id!r})"


def active_record(records: Sequence[CommitRecord]) -> CommitRecord | None:
    """The newest record of ``records``, a commit history in step order, that activates a release; ``None`` when
    no record does (the creation artifact is active)."""
    return next((record for record in reversed(records) if record.activates), None)


def active_artifact_ref(records: Sequence[CommitRecord], fallback: ArtifactRef) -> ArtifactRef:
    """The release that is active after ``records``: the one :func:`active_record` names, else ``fallback``.

    ``fallback`` is the release that is active when no record activates one:
    the creation artifact, or the chain tip of a store that keeps no
    history across a restart.
    """
    record = active_record(records)
    return fallback if record is None else record.artifact_ref


SCENARIO_METADATA_KEY = "scenario_commit_record"
SCENARIO_METADATA_KIND = "reef-scenario/4"


def scenario_metadata_for(
    *,
    name: str,
    base_artifact: ArtifactRef,
    scenario_step: int = 0,
    algorithm_state: Mapping[str, Any] | None = None,
    record_progress: RecordProgress | None = None,
    metrics: Mapping[str, Any] | None = None,
    training_job_id: str | None = None,
    component: str | None = None,
) -> dict[str, object]:
    """The scenario metadata of a registration (step zero) or of the artifact a training step made."""
    if not isinstance(scenario_step, int) or isinstance(scenario_step, bool) or scenario_step < 0:
        raise ValueError("scenario_step must be non-negative")
    metadata: dict[str, object] = {
        "format": SCENARIO_METADATA_KIND,
        "scenario": name,
        "scenario_step": scenario_step,
        "base_artifact": encode_artifact_ref(base_artifact),
        "operation": "training",
    }
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
    return metadata


def parse_scenario_metadata(
    value: Mapping[str, Any], *, checkpoint_head: ArtifactRef, components: Mapping[str, str] | None = None
) -> tuple[str, ArtifactRef, CommitRecord | None]:
    """Read registration and a checkpoint commit; step zero has no commit.

    ``components`` is the manifest of ``checkpoint_head``, read from the same
    release metadata, so a commit rebuilt from it carries what a recorded one
    does.
    """
    if value.get("format") != SCENARIO_METADATA_KIND:
        raise ValueError(f"unsupported scenario metadata format: {value.get('format')!r}")
    scenario = value.get("scenario")
    if not isinstance(scenario, str) or not scenario:
        raise ValueError("scenario metadata requires scenario")
    raw_base = value.get("base_artifact")
    if not isinstance(raw_base, Mapping):
        raise ValueError("scenario metadata requires base_artifact")
    try:
        base_artifact = decode_artifact_ref(raw_base)
    except ValueError as exc:
        raise ValueError(f"invalid scenario metadata base_artifact: {exc}") from exc
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
    if scenario_step == 0:
        return scenario, base_artifact, None
    if record_progress is None:
        raise ValueError("scenario metadata requires record_progress after step zero")
    commit = CommitRecord(
        scenario=scenario,
        step=scenario_step,
        artifact_ref=checkpoint_head,
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
    )
    return scenario, base_artifact, commit


__all__ = [
    "RECORD_KIND",
    "SCENARIO_METADATA_KEY",
    "SCENARIO_METADATA_KIND",
    "CommitHistoryError",
    "CommitOperation",
    "CommitRecord",
    "RecordProgress",
    "active_artifact_ref",
    "active_record",
    "parse_record_progress",
    "parse_scenario_metadata",
    "scenario_metadata_for",
]
