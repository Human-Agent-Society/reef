"""The record-to-record stage: records in, record sets out, for one pipeline.

``Record2Record`` is the interface. ``NoRecordSets`` is the default of a pipeline
that trains nothing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Mapping
from enum import Enum
from typing import Any, Self

from reef.core.plugins import NoSettings, Plugin, PluginConfigError
from reef.core.records_types import AgentRecord, RequestType
from reef.pipeline_context import PipelineContext
from reef.record2record.pool import RecordSetUnitPool
from reef.record2record.selection import SelectionPolicy
from reef.record2record.sets import RecordSet, RecordSetUnit, SingleRecordSet, SingleSetUnit
from reef.record2record.staleness import canonical_runtime_load_id, stale_units


class TaskStatus(Enum):
    """What the record-to-task stage made of a record set. The status settles all records of the set together."""

    #: The set has one or more tasks. Its records wait in one batch unit until a batch uses it.
    READY = "ready"
    #: The set has no task and cannot train. The stage ends its records.
    SKIPPED = "skipped"


class Record2Record(ABC, Plugin):
    """Turn records into record sets for one pipeline, and decide which batch units train and in what order.

    The pipeline pushes every record that it reads, except training
    instructions, in append order (``push``). It pulls the record sets that
    the stage formed (``pull``). When the pipeline looks for a batch, it first
    advances the stage to the current time (``advance``): the stage takes in
    background work that ended, such as judgments, and starts work that is due. No record is in two record sets of one stage:
    ``form`` refuses a set that holds a record of another set. The pipeline
    makes tasks from each set and gives the result back (``settle``). The stage
    holds the sets that have tasks in batch units in its ``pool`` until the
    pipeline reserves them for a batch; the pipeline reserves and consumes
    through the pool, and the stage ends the consumed units (``consume``). The
    pool applies the stage's selection policy. The stage never calls the pipeline.

    The stage also runs the staleness check. No other part of Reef checks how
    far the training data lags behind the weights. The pipeline gives the stage
    each version that the scenario serves (``observe_serving_runtime_load_id``).
    Then it tells the stage to drop the held units that are too many versions
    behind (``drop_stale_units``).

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

    def __init__(self, selection: SelectionPolicy | None = None) -> None:
        #: The held batch units. The pipeline reserves a batch from it and consumes the reservation.
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
        self.served_runtime_load_ids: list[str] = []

    @abstractmethod
    def push(self, record: AgentRecord) -> None:
        """Read one record. The stage must not block on network or model calls."""

    def form(self, record_set: RecordSet) -> None:
        """Form ``record_set`` for the pipeline. Raises ``ValueError`` if a record of it is in another set.

        A set with the ID of an earlier set takes its records again, as a retry does.
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
        self.formed.append(record_set)

    def advance(self, now: float) -> None:
        """Take in the background work that ended by ``now``, a monotonic time, and start the work that is due."""
        return

    def pull(self) -> tuple[RecordSet, ...]:
        """Return the record sets formed since the last call, in the order they formed."""
        formed, self.formed = tuple(self.formed), []
        return formed

    def context_changed(self, context: PipelineContext) -> None:
        """The pipeline's context changed: its training mode, held-unit limit, or admitted report type."""
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
        """Hold a set that has tasks in the pool, as a batch unit by itself."""
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
        """Add the version that the scenario serves now, if the version changed.

        A version of another incarnation starts the list again. The engine
        restarted, so no earlier version can train.
        """
        served = self.served_runtime_load_ids
        if served and served[-1] == runtime_load_id:
            return
        previous = canonical_runtime_load_id(served[-1]) if served else None
        current = canonical_runtime_load_id(runtime_load_id)
        if previous is not None and current is not None and previous.incarnation != current.incarnation:
            served.clear()
        served.append(runtime_load_id)

    def drop_stale_units(self, max_staleness: int) -> None:
        """End the held units whose training data is more than ``max_staleness`` versions behind.

        The check does not run while a batch is reserved, because the batch refers to held units.
        """
        if self.pool.reserved is not None:
            return
        for unit in stale_units(
            self.pool.units(), self.producing_runtime_load_ids, self.served_runtime_load_ids, max_staleness
        ):
            self.drop_held_unit(unit)

    def drop_held_unit(self, unit: RecordSetUnit) -> None:
        """End a held unit that can no longer train. A stage that holds groups overrides it."""
        self.pool.remove(unit.unit_id)
        for record_set in unit.record_sets():
            self.reject(record_set)

    def consume(self, units: tuple[RecordSetUnit, ...], *, trained: bool) -> frozenset[str]:
        """End the units that an acknowledged batch consumed, and return their keys.

        ``trained`` is False when the backend dropped the batch as stale: its
        units are consumed all the same, but nothing trained on them.
        """
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
    """Form no record set: the default of a pipeline that trains nothing.

    The stage reads each record and keeps nothing. The pipeline then makes no
    task and no batch, and the records stay stored for audit.
    """

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, context: PipelineContext, environ: Mapping[str, str]) -> Self:
        NoSettings.parse(config, plugin_class=cls)
        return cls()

    def push(self, record: AgentRecord) -> None:
        return


class InferenceRecord2Record(Record2Record):
    """Hold each inference record as one set, in arrival order, and end every other record untouched."""

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, context: PipelineContext, environ: Mapping[str, str]) -> Self:
        NoSettings.parse(config, plugin_class=cls)
        return cls()

    def push(self, record: AgentRecord) -> None:
        # A record that is not an inference ends at once.
        if record.request_type is not RequestType.INFERENCE:
            self.terminal.add(record.agent_record_id)
            return
        self.form(
            SingleRecordSet(set_id=record.agent_record_id, arrival_index=self.pool.next_arrival_index(), record=record)
        )
