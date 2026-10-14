"""The record-to-record stage: records in, record sets out, for one pipeline.

``Record2Record`` is the interface. ``PassThroughRecord2Record`` is the default implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Mapping
from dataclasses import replace
from enum import Enum
from typing import Any, Self

from reef.core.plugins import NoSettings, Plugin, PluginConfigError
from reef.core.records_types import AgentRecord, RequestType
from reef.pipeline_context import PipelineContext
from reef.record2record.pool import RecordSetPool
from reef.record2record.selection import SelectionPolicy
from reef.record2record.sets import RecordSet, SingleRecordSet


class TaskStatus(Enum):
    """What the record-to-task stage made of a formed record set."""

    #: The set has tasks. The stage holds the set until a batch uses it.
    READY = "ready"
    #: The set cannot train. The stage ends its records.
    SKIPPED = "skipped"
    #: The set makes no task, as in a pipeline that trains no weights. The stage forgets the set and keeps its records.
    INGESTED = "ingested"


class Record2Record(ABC, Plugin):
    """Turn records into record sets for one pipeline, and decide which sets train and in what order.

    The pipeline gives the stage every record that it reads, except training
    instructions, in append order. The stage forms record sets
    (``take_formed``). The pipeline makes tasks from each formed set and gives
    the result back (``settle``). The stage holds each set that has tasks in
    its ``pool`` until the pipeline reserves it for a batch; the pipeline
    reserves and consumes through the pool, and the stage ends the consumed
    sets (``consume``). The pool applies the stage's selection policy. The
    stage never calls the pipeline.

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
        #: The held sets. The pipeline reserves a batch from it and consumes the reservation.
        self.pool = RecordSetPool(selection)
        #: Formed sets that ended after the task stage saw them. The pipeline forgets their tasks.
        self.ended_set_ids: set[Hashable] = set()
        #: Records that ended and that no live set references, until their release.
        self.terminal: set[str] = set()
        #: Sets formed since the last ``take_formed``, in the order they formed.
        self.formed: list[RecordSet] = []

    @abstractmethod
    def ingest(self, record: AgentRecord) -> None:
        """Read one record. The stage must not block on network or model calls."""

    def take_formed(self) -> tuple[RecordSet, ...]:
        """Return the record sets formed since the last call, in the order they formed."""
        formed, self.formed = tuple(self.formed), []
        return formed

    def poll(self, now: float) -> None:
        """Absorb background work, such as finished judgments. ``now`` is a monotonic time."""
        return

    def context_changed(self, context: PipelineContext) -> None:
        """The pipeline's context changed: its training mode, held-set limit, or admitted report type."""
        return

    def settle(self, record_set: RecordSet, status: TaskStatus, runtime_load_id: str | None = None) -> None:
        """Keep or end a formed set from what the task stage made of it.

        ``runtime_load_id`` is the version of the set's training data, when the
        task stage knows it. A stage's staleness check reads it.
        """
        if status is TaskStatus.READY:
            self.accept(record_set, runtime_load_id)
        elif status is TaskStatus.SKIPPED:
            self.reject(record_set)
        elif status is not TaskStatus.INGESTED:
            raise TypeError("a task result must have a TaskStatus")

    def accept(self, record_set: RecordSet, runtime_load_id: str | None) -> None:
        """Hold a set that has tasks in the pool."""
        self.pool.put(replace(record_set, runtime_load_id=runtime_load_id))

    def reject(self, record_set: RecordSet) -> None:
        """End a set that cannot train."""
        self.terminal.update(record_set.record_ids())
        self.ended_set_ids.add(record_set.set_id)

    def take_ended(self) -> frozenset[Hashable]:
        """Return the IDs of the formed sets that ended since the last call. The pipeline forgets their tasks."""
        ended, self.ended_set_ids = frozenset(self.ended_set_ids), set()
        return ended

    def consume(self, record_sets: tuple[RecordSet, ...], *, trained: bool) -> frozenset[str]:
        """End the sets that an acknowledged batch consumed, and return the IDs of their consumed records.

        ``trained`` is False when the backend dropped the batch as stale: its
        sets are consumed all the same, but nothing trained on them.
        """
        consumed = frozenset(record_id for record_set in record_sets for record_id in record_set.record_ids())
        self.terminal |= consumed
        self.ended_set_ids.update(record_set.set_id for record_set in record_sets)
        return consumed

    # ------------------------------------------------------------- lifecycle

    def releasable_record_ids(self) -> frozenset[str]:
        """Return the IDs of records that the stage ended and that no live set references."""
        return frozenset(self.terminal)

    def release_records(self, agent_record_ids: frozenset[str]) -> None:
        """Forget released records. Stored records do not change."""
        self.terminal -= agent_record_ids

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


class PassThroughRecord2Record(Record2Record):
    """Hand each record to the pipeline by itself, in arrival order, and decide nothing.

    With ``NoTasks`` each set is ``TaskStatus.INGESTED``: the stage forgets it.
    The stage holds no set, so it refuses a task stage that makes tasks.
    """

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, context: PipelineContext, environ: Mapping[str, str]) -> Self:
        NoSettings.parse(config, plugin_class=cls)
        return cls()

    def ingest(self, record: AgentRecord) -> None:
        self.formed.append(
            SingleRecordSet(set_id=record.agent_record_id, arrival_index=self.pool.next_arrival_index(), record=record)
        )

    def accept(self, record_set: RecordSet, runtime_load_id: str | None) -> None:
        raise TypeError(f"{type(self).__name__} holds no set: pair it with NoTasks, or hold sets in another stage")

    def reject(self, record_set: RecordSet) -> None:
        raise TypeError(f"{type(self).__name__} ends no set: pair it with NoTasks, or hold sets in another stage")

    def consume(self, record_sets: tuple[RecordSet, ...], *, trained: bool) -> frozenset[str]:
        raise TypeError(f"{type(self).__name__} holds no set, so no batch can consume one")


class InferenceRecord2Record(Record2Record):
    """Hold each inference record as one set, in arrival order, and end every other record untouched."""

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, context: PipelineContext, environ: Mapping[str, str]) -> Self:
        NoSettings.parse(config, plugin_class=cls)
        return cls()

    def ingest(self, record: AgentRecord) -> None:
        # A record that is not an inference ends at once.
        if record.request_type is not RequestType.INFERENCE:
            self.terminal.add(record.agent_record_id)
            return
        self.formed.append(
            SingleRecordSet(set_id=record.agent_record_id, arrival_index=self.pool.next_arrival_index(), record=record)
        )
