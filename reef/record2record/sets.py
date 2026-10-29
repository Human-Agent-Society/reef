"""Record sets and record set units: the input of the record-to-sample stage, and the sets whose samples are stored together.

Notation: ``refs(r)`` is the set of records that a record ``r`` references.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable
from dataclasses import dataclass

from reef.core.records_types import AgentRecord


@dataclass(frozen=True, kw_only=True)
class RecordSet(ABC):
    """The records that the record-to-sample stage makes samples from, together.

    Values: ``X`` is a record set if and only if ``records(X)`` is not empty and
    ``refs(r) ⊆ records(X)`` for each record ``r`` in ``records(X)``. ``set_id``
    identifies ``X`` among the sets of its record stage. No record is in two
    record sets of one stage (``Record2Record.form``).

    Purpose: the record stage forms it, and the pipeline pushes it to the
    record-to-sample stage. Its one result settles all records of the set together.
    """

    set_id: Hashable

    def __post_init__(self) -> None:
        records = self.records()
        if not records:
            raise ValueError("a record set must hold one or more records")
        missing = {reference for record in records for reference in record.references} - set(self.record_ids())
        if missing:
            raise ValueError(f"record set {self.set_id!r} does not hold the referenced records {sorted(missing)!r}")

    @abstractmethod
    def records(self) -> tuple[AgentRecord, ...]:
        """Return the records of the set."""

    def record_ids(self) -> tuple[str, ...]:
        """Return the IDs of the records of the set, in the order of ``records``."""
        return tuple(record.agent_record_id for record in self.records())


@dataclass(frozen=True, kw_only=True)
class SingleRecordSet(RecordSet):
    """One record that references no other record. ``set_id`` is the record ID."""

    record: AgentRecord

    def records(self) -> tuple[AgentRecord, ...]:
        return (self.record,)


@dataclass(frozen=True, kw_only=True)
class RecordSetUnit(ABC):
    """Record sets whose samples train together.

    Values: ``U`` is a record set unit if and only if ``record_sets(U)`` is not
    empty. Its record sets are distinct and share no record, because a record
    stage forms each record into one set only. ``unit_id`` identifies ``U`` among
    the ready units of its record stage; a ready unit replaces the one with its ID.

    Purpose: the record stage makes it ready when the samples of its record sets
    are ready (``Record2Record.take_ready_units``). The pipeline stores those
    samples as one group in one transaction, which consumes ``consumed_ids(U)``,
    the IDs of the records of its record sets. Then the stage ends it (``consume``).
    """

    unit_id: Hashable

    def __post_init__(self) -> None:
        if not self.record_sets():
            raise ValueError(f"record set unit {self.unit_id!r} must hold one or more record sets")

    @abstractmethod
    def record_sets(self) -> tuple[RecordSet, ...]:
        """Return the record sets of the unit, in the order that their samples train."""

    def set_ids(self) -> tuple[Hashable, ...]:
        """Return the IDs of the record sets of the unit, in the order of ``record_sets``."""
        return tuple(record_set.set_id for record_set in self.record_sets())

    def consumed_ids(self) -> frozenset[str]:
        """Return the record IDs that storing the unit's samples consumes."""
        return frozenset(record_id for record_set in self.record_sets() for record_id in record_set.record_ids())


@dataclass(frozen=True, kw_only=True)
class SingleSetUnit(RecordSetUnit):
    """A record set unit of one record set, with the ID of that set. ``of`` makes it."""

    record_set: RecordSet

    @classmethod
    def of(cls, record_set: RecordSet) -> SingleSetUnit:
        """Return the unit of ``record_set``."""
        return cls(unit_id=record_set.set_id, record_set=record_set)

    def record_sets(self) -> tuple[RecordSet, ...]:
        return (self.record_set,)
