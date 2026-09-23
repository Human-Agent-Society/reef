"""Report storage events, accepted records and committed steps, to an observer.

Both events happen at the storage boundary: a record row is inserted for the
first time, or a commit row lands. Wrapping the storage with
:class:`ObservedScenarioStorage` reports them without any coordination code
knowing an observer exists. The wrappers delegate every other call
unchanged and isolate observer failures, so an exporter can never become
part of record acceptance or the commit transaction.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from reef.core.records_types import AgentRecord, RequestType
from reef.storage.commits import CommitRecord
from reef.storage.records import AppendResult, RecordStore, StoredRecord
from reef.storage.scenario import ScenarioStorage, ScenarioStore

logger = logging.getLogger(__name__)


class RecordObserver:
    """Process-level observer of first-time record inserts and committed steps.

    Observation is a side effect. The wrappers below already isolate
    failures; implementations should still keep their own work off the
    caller's thread where it can block, such as network export.
    """

    def record_accepted(self, item: AgentRecord) -> None:
        """One record the store inserted for the first time (retries are not reported)."""

    def record_committed(self, commit: CommitRecord) -> None:
        """One committed step, training or rollback, with the record ids its batch consumed."""

    def close(self) -> None:
        pass


class ObservedRecordStore(RecordStore):
    """A record store that reports first-time inserts."""

    def __init__(self, inner: RecordStore, observer: RecordObserver) -> None:
        self._inner = inner
        self._observer = observer

    def append(self, item: AgentRecord) -> AgentRecord:
        return self.append_result(item).item

    def append_result(self, item: AgentRecord) -> AppendResult:
        appended = self._inner.append_result(item)
        if appended.inserted:
            try:
                self._observer.record_accepted(appended.item)
            except Exception:
                logger.exception("record observer failed on accepted record %s", appended.item.agent_record_id)
        return appended

    def existing_receipt(self, item: AgentRecord) -> AgentRecord | None:
        return self._inner.existing_receipt(item)

    def get(self, scenario: str, agent_record_id: str) -> AgentRecord | None:
        return self._inner.get(scenario, agent_record_id)

    def replay(self, scenario: str, *, offset: int = 0, limit: int | None = None) -> tuple[AgentRecord, ...]:
        return self._inner.replay(scenario, offset=offset, limit=limit)

    def replay_page(
        self, scenario: str, *, after_sequence: int = 0, limit: int = 256
    ) -> tuple[tuple[int, AgentRecord], ...]:
        return self._inner.replay_page(scenario, after_sequence=after_sequence, limit=limit)

    def count(self, scenario: str, *, request_type: RequestType | None = None, after_sequence: int = 0) -> int:
        return self._inner.count(scenario, request_type=request_type, after_sequence=after_sequence)

    def latest_sequence(self, scenario: str) -> int:
        return self._inner.latest_sequence(scenario)

    def get_for_audit(self, scenario: str, agent_record_id: str) -> StoredRecord | None:
        return self._inner.get_for_audit(scenario, agent_record_id)

    def audit_page(self, scenario: str, *, after_sequence: int = 0, limit: int = 256) -> tuple[StoredRecord, ...]:
        return self._inner.audit_page(scenario, after_sequence=after_sequence, limit=limit)

    def compact(
        self,
        scenario: str,
        agent_record_ids: frozenset[str],
        *,
        receipt_id: str | None = None,
        receipt_metadata: Mapping[str, object] | None = None,
    ) -> None:
        self._inner.compact(scenario, agent_record_ids, receipt_id=receipt_id, receipt_metadata=receipt_metadata)

    def purge_compacted(self, scenario: str, *, before: float, limit: int = 256) -> int:
        return self._inner.purge_compacted(scenario, before=before, limit=limit)

    def compaction_receipts(self, scenario: str) -> tuple[dict[str, object], ...]:
        return self._inner.compaction_receipts(scenario)

    def close(self) -> None:
        self._inner.close()


class ObservedScenarioStore(ScenarioStore):
    """A scenario session whose record inserts and landed commits are reported."""

    def __init__(self, inner: ScenarioStore, observer: RecordObserver) -> None:
        self._inner = inner
        self._observer = observer
        self._records = ObservedRecordStore(inner.records, observer)

    @property
    def records(self) -> RecordStore:
        return self._records

    @property
    def durable(self) -> bool:
        return self._inner.durable

    def history(self) -> tuple[CommitRecord, ...]:
        return self._inner.history()

    def training_run_position(self) -> tuple[int, int]:
        return self._inner.training_run_position()

    def commit_step(self, *, expected_step: int, commit: CommitRecord) -> CommitRecord:
        recorded = self._inner.commit_step(expected_step=expected_step, commit=commit)
        try:
            self._observer.record_committed(recorded)
        except Exception:
            logger.exception("record observer failed on committed step %s of %s", recorded.step, recorded.scenario)
        return recorded

    def recover(self, *, checkpoint: CommitRecord | None) -> CommitRecord | None:
        return self._inner.recover(checkpoint=checkpoint)

    def close(self) -> None:
        self._inner.close()


class ObservedScenarioStorage(ScenarioStorage):
    """Storage whose sessions report to one observer; closing the storage closes the observer."""

    def __init__(self, inner: ScenarioStorage, observer: RecordObserver) -> None:
        self._inner = inner
        self._observer = observer

    @property
    def durable(self) -> bool:
        return self._inner.durable

    def open(self, scenario: str) -> ScenarioStore:
        return ObservedScenarioStore(self._inner.open(scenario), self._observer)

    def archive(self, scenario: str) -> tuple[str, ...]:
        return self._inner.archive(scenario)

    def prune(self, *, days: float, max_bytes: int) -> int:
        return self._inner.prune(days=days, max_bytes=max_bytes)

    def close(self) -> None:
        try:
            self._inner.close()
        finally:
            try:
                self._observer.close()
            except Exception:
                logger.exception("record observer failed to close")


__all__ = ["ObservedRecordStore", "ObservedScenarioStorage", "ObservedScenarioStore", "RecordObserver"]
