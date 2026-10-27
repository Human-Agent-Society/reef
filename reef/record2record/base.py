"""The record-to-record stage: records in, record sets out, for one pipeline.

``Record2Record`` is the interface, and ``Record2RecordSettings`` is the config
that every stage that holds units takes. ``NoRecordSets`` is the default of a
pipeline that trains nothing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Self

from reef.core.config import config_option
from reef.core.plugins import NoSettings, Plugin, PluginConfigError, PluginSettings
from reef.core.records_types import AgentRecord, RequestType
from reef.pipeline_context import PipelineContext
from reef.record2record.pool import RecordSetUnitPool
from reef.record2record.selection import SelectionPolicy
from reef.record2record.sets import RecordSet, RecordSetUnit, SingleRecordSet, SingleSetUnit
from reef.record2record.staleness import ServedVersions, stale_units
from reef.runtime.interfaces import inference_aborted


class TaskStatus(Enum):
    """What the record-to-task stage made of a record set. The status settles all records of the set together."""

    #: The set has one or more tasks. Its records wait in one record set unit until job data uses it.
    READY = "ready"
    #: The set has no task and cannot train. The stage ends its records.
    SKIPPED = "skipped"


class UnitDropReason(Enum):
    """Why an unreserved unit ends without training.

    Values are BUFFER_LIMIT (the sample buffer exceeded its limit) and STALE
    (the unit failed the serving-version check). The pipeline and record
    stage pass these values to ``drop_held_unit`` so stages can distinguish
    capacity eviction from obsolete training data. Neither value confirms training.
    """

    BUFFER_LIMIT = "buffer_limit"
    STALE = "stale"


def require_max_staleness(value: object) -> None:
    """Raise ``ValueError`` unless ``value`` is a non-negative int (not a bool)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"max_staleness must be a non-negative int, got {value!r}")


@dataclass(frozen=True)
class Record2RecordSettings(PluginSettings):
    """The config of a record-to-record stage: how many versions its training data may lag behind.

    Values: ``max_staleness`` is a non-negative int. The constructor refuses
    other values. With 0, only data of the version that the scenario serves
    trains.
    Purpose: a stage's ``from_config`` parses it from
    ``pipeline.record2record.config`` and gives ``max_staleness`` to
    ``Record2Record.__init__``. A stage settings class that has more keys
    extends it.
    """

    max_staleness: int = config_option(
        0,
        help=(
            "The versions that training data may lag behind the served version and still train. "
            "0 trains only data of the served version."
        ),
    )

    def __post_init__(self) -> None:
        require_max_staleness(self.max_staleness)


class Record2Record(ABC, Plugin):
    """Turn records into record sets for one pipeline, and decide which record set units train and in what order.

    The pipeline pushes every record that it reads, except training
    instructions, in append order (``push``). It pulls the record sets that
    the stage formed (``pull``). When the pipeline looks for job data, it first
    advances the stage to the current time (``advance``): the stage takes in
    background work that ended, such as judgments, and starts work that is due. No record is in two record sets of one stage:
    ``form`` refuses a set that holds a record of another set. The pipeline
    makes tasks from each set and gives the result back (``settle``). Before
    exposing a set, ``form`` rejects it if any inference response reports an
    engine abort. Rejection ends the whole set through the stage's lifecycle,
    including reports that reference an aborted inference. The stage
    holds the sets that have tasks in record set units in its ``pool`` until the
    pipeline reserves them for job data; the pipeline reserves and consumes
    through the pool, and the stage ends the consumed units (``consume``). The
    pool applies the stage's selection policy. The stage never calls the pipeline.

    The stage also runs the staleness check. No other part of Reef checks how
    far the training data lags behind the weights. The pipeline gives the stage
    each version that the scenario serves (``observe_serving_runtime_load_id``).
    Then it tells the stage to drop the held units that are more than
    ``max_staleness`` versions behind (``drop_stale_units``). ``max_staleness``
    is a non-negative int, from ``Record2RecordSettings``; the constructor
    refuses other values.

    A record that the stage ends, for example a record that can no longer
    train, appears in ``releasable_record_ids``. The trainer adds those IDs to
    the next commit, so the record never trains after a restart. The default
    bookkeeping keeps the ended records in ``terminal`` until their release; a
    stage with its own record lifecycle overrides the lifecycle methods.

    The stage keeps only memory that a replay of the retained records
    rebuilds, as pipeline memory does.

    A deployment names a stage class in ``pipeline.record2record`` and gives
    its config. The class parses the config into its settings dataclass and
    builds the stage in ``from_config``.
    """

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, context: PipelineContext, environ: Mapping[str, str]) -> Self:
        """Build the stage of one scenario from its config.

        ``environ`` holds the environment the deployment was read with.
        """
        raise PluginConfigError(f"{cls.__name__} cannot be built from a deployment file: it has no from_config")

    def __init__(self, selection: SelectionPolicy | None = None, *, max_staleness: int = 0) -> None:
        require_max_staleness(max_staleness)
        #: How many versions the training data of a held unit may lag behind the served version and still train.
        self.max_staleness = max_staleness
        #: The held record set units. The pipeline reserves job data from it and consumes the reservation.
        self.pool = RecordSetUnitPool(selection)
        #: Record sets that ended after the task stage saw them. The pipeline forgets their tasks.
        self.ended_set_ids: set[Hashable] = set()
        #: Records that ended and that no live set references, until their release.
        self.terminal: set[str] = set()
        #: Sets formed since the last ``pull``, in the order they formed.
        self.formed: list[RecordSet] = []
        #: The set ID of each record that a formed set holds, until the record's release.
        self.set_of_record: dict[str, Hashable] = {}
        #: The runtime load IDs that produced each held record set's training data, by set ID.
        self.producing_runtime_load_ids: dict[Hashable, frozenset[str | None]] = {}
        #: The versions that the scenario served, oldest first. The last one is the version that it serves now.
        self.served_versions = ServedVersions()

    @abstractmethod
    def push(self, record: AgentRecord) -> None:
        """Read one record. The stage must not block on network or model calls."""

    def form(self, record_set: RecordSet) -> None:
        """Form ``record_set`` for the pipeline. Raises ``ValueError`` if a record of it is in another set.

        A set with the ID of an earlier set takes its records again, as a retry does.
        An engine-aborted inference rejects the whole set before task creation.
        Replaying retained records applies the same check.
        """
        record_ids = record_set.record_ids()
        taken = [
            record_id
            for record_id in record_ids
            if self.set_of_record.get(record_id, record_set.set_id) != record_set.set_id
        ]
        if taken:
            raise ValueError(f"records {taken!r} are in another record set and cannot join {record_set.set_id!r}")
        self.set_of_record.update(dict.fromkeys(record_ids, record_set.set_id))
        for record in record_set.records():
            if record.request_type is not RequestType.INFERENCE:
                continue
            response = record.payload.get("response")
            if isinstance(response, Mapping) and inference_aborted(response):
                self.reject(record_set)
                return
        self.formed.append(record_set)

    def advance(self, now: float) -> None:
        """Take in the background work that ended by ``now``, a monotonic time, and start the work that is due."""
        return

    def pull(self) -> tuple[RecordSet, ...]:
        """Return the record sets formed since the last call, in the order they formed."""
        formed, self.formed = tuple(self.formed), []
        return formed

    def context_changed(self, context: PipelineContext) -> None:
        """The pipeline's context changed: its training mode or admitted report type."""
        return

    def settle(
        self,
        record_set: RecordSet,
        status: TaskStatus,
        producing_runtime_load_ids: frozenset[str | None] = frozenset(),
    ) -> None:
        """Keep a READY record set in the pool, or end a SKIPPED one.

        ``producing_runtime_load_ids`` are the versions that produced the
        training data of the set. The staleness check reads them.
        """
        if status is TaskStatus.READY:
            self.producing_runtime_load_ids[record_set.set_id] = producing_runtime_load_ids
            self.accept(record_set)
        else:
            self.reject(record_set)

    def accept(self, record_set: RecordSet) -> None:
        """Hold a set that has tasks in the pool, as a record set unit by itself."""
        self.pool.put(SingleSetUnit.of(record_set))

    def reject(self, record_set: RecordSet) -> None:
        """End a set that cannot train."""
        self.terminal.update(record_set.record_ids())
        self.ended_set_ids.add(record_set.set_id)

    def take_ended(self) -> frozenset[Hashable]:
        """Return the IDs of the record sets that ended since the last call. The pipeline forgets their tasks."""
        ended, self.ended_set_ids = frozenset(self.ended_set_ids), set()
        for set_id in ended:
            self.producing_runtime_load_ids.pop(set_id, None)
        return ended

    # ------------------------------------------------------------- staleness

    def observe_serving_runtime_load_id(self, runtime_load_id: str) -> None:
        """Add the version that the scenario serves now, if the version changed (``ServedVersions.observe``)."""
        self.served_versions.observe(runtime_load_id)

    def drop_stale_units(self) -> None:
        """End the held units whose training data is more than ``max_staleness`` versions behind.

        The check does not run while job data is reserved, because the job data refers to held units.
        """
        if self.pool.reserved is not None:
            return
        for unit in stale_units(
            self.pool.units(), self.producing_runtime_load_ids, self.served_versions.versions, self.max_staleness
        ):
            self.drop_held_unit(unit, reason=UnitDropReason.STALE)

    def drop_held_unit(self, unit: RecordSetUnit, *, reason: UnitDropReason) -> None:
        """End an unreserved unit without training for the given reason.

        A stage that holds groups overrides this operation to end the whole
        group. The reason does not change the default record cleanup.
        """
        self.pool.remove(unit.unit_id)
        for record_set in unit.record_sets():
            self.reject(record_set)

    def consume(self, units: tuple[RecordSetUnit, ...]) -> frozenset[str]:
        """End the units that acknowledged job data trained on, and return their keys."""
        consumed = frozenset(key for unit in units for key in unit.consumed_ids())
        self.terminal |= consumed
        self.ended_set_ids.update(set_id for unit in units for set_id in unit.set_ids())
        return consumed

    # ------------------------------------------------------------- lifecycle

    def releasable_record_ids(self) -> frozenset[str]:
        """Return the IDs of records that the stage ended and that no live set references."""
        return frozenset(self.terminal)

    def release_records(self, agent_record_ids: frozenset[str]) -> None:
        """Forget released records. Stored records do not change."""
        self.terminal -= agent_record_ids
        for record_id in agent_record_ids:
            self.set_of_record.pop(record_id, None)

    def background_jobs_running(self) -> bool:
        """Return True while background jobs run. A finished job can form sets without a new record."""
        return False

    def status(self) -> Mapping[str, Any]:
        """Return JSON-safe state that the pipeline adds to its status."""
        return {}

    def operational_metrics(self) -> Mapping[str, float | int]:
        """Return numeric state that the pipeline adds to its metrics."""
        return {}

    def close(self) -> None:
        """Release resources that the stage owns. Calling it more than one time is safe."""
        return


class NoRecordSets(Record2Record):
    """Form no record set: the default of a pipeline that trains nothing. No config.

    The stage reads each record and keeps nothing. The pipeline then makes no
    task and no job data, and the records stay stored for audit. The stage
    holds no unit, so it refuses ``max_staleness``.
    """

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, context: PipelineContext, environ: Mapping[str, str]) -> Self:
        NoSettings.parse(config, plugin_class=cls)
        return cls()

    def push(self, record: AgentRecord) -> None:
        return


class InferenceRecord2Record(Record2Record):
    """Hold each inference record as one set, in arrival order, and end every other record untouched.

    Config: ``Record2RecordSettings``.
    """

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, context: PipelineContext, environ: Mapping[str, str]) -> Self:
        settings = Record2RecordSettings.parse(config, plugin_class=cls)
        return cls(max_staleness=settings.max_staleness)

    def push(self, record: AgentRecord) -> None:
        # A record that is not an inference ends at once.
        if record.request_type is not RequestType.INFERENCE:
            self.terminal.add(record.agent_record_id)
            return
        self.form(
            SingleRecordSet(set_id=record.agent_record_id, arrival_index=self.pool.next_arrival_index(), record=record)
        )
