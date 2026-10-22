"""The commit history in the record database of a scenario, for each SQL backend.

The commit table is in the same database as the records. Each operation uses
one transaction of the record store. The adapters build the table with
:func:`commit_table` and supply their namespace.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from threading import RLock

from sqlalchemy import BigInteger, Column, ForeignKey, Integer, MetaData, String, Table, Text, select
from sqlalchemy.engine import Connection

from reef.storage.commit_history import CommitHistory, same_commit
from reef.storage.commits import CommitHistoryError, CommitRecord
from reef.storage.sql_records import ScopedTable, SQLRecordStore


def commit_table(metadata: MetaData, *, foreign_key: str | None = None) -> Table:
    """Define the ``scenario_commit`` table in ``metadata``: one row per committed step.

    ``commit_json`` holds the canonical JSON object of ``CommitRecord.to_dict``.
    If ``foreign_key`` names a column, the table also has the column
    ``storage_id``, which refers to it. The adapter then gives ``storage_id``
    in ``ScopedTable.scope``. Steps are unique in each namespace.
    """
    if foreign_key is None:
        return Table(
            "scenario_commit",
            metadata,
            Column("step", Integer, primary_key=True, autoincrement=False),
            Column("commit_json", Text, nullable=False),
        )
    return Table(
        "scenario_commit",
        metadata,
        Column("storage_id", String(32), ForeignKey(foreign_key), primary_key=True),
        Column("step", BigInteger, primary_key=True),
        Column("commit_json", Text, nullable=False),
    )


class SQLCommitHistory(CommitHistory):
    """The commits of one scenario generation, as the rows of its namespace in the commit table.

    Values: ``record_store`` is the open record store whose transactions the
    history uses. ``table`` is the commit table and the namespace of this
    generation. ``label`` names the table for error messages, and ``durable``
    is true when the database outlives the process. ``records()`` is the rows
    of the namespace in step order, decoded. The cached records are always a
    prefix of those rows, so a read fetches only the steps after them; a
    locked block that rolls back clears the cache. A row deleted by anything
    but this class is outside the contract.

    Purpose: a SQL record store makes it in ``commit_history()``.
    ``CommitHistoryScenarioStore`` reads and appends through it.

    ``locked`` holds one write transaction of the record store: ``BEGIN
    IMMEDIATE`` in SQLite, the row lock on the store generation in PostgreSQL.
    The reads and appends of the locking thread use that transaction, so an
    append is durable when the locked block ends, and an exception in the
    block stores nothing. Outside a locked block, each call uses its own
    transaction. Once a PostgreSQL store generation is archived, every
    operation raises ``RuntimeError``, as the record operations of the same
    session do.
    """

    def __init__(self, record_store: SQLRecordStore, table: ScopedTable, *, label: str, durable: bool) -> None:
        self.record_store = record_store
        self.table = table
        self._label = label
        self._durable = durable
        self._lock = RLock()
        self._connection: Connection | None = None
        self._cached: tuple[CommitRecord, ...] = ()

    @property
    def label(self) -> str:
        return self._label

    @property
    def durable(self) -> bool:
        return self._durable

    @contextmanager
    def locked(self) -> Iterator[None]:
        with self._lock:
            try:
                with self.record_store._transaction(write=True) as connection:
                    self._connection = connection
                    try:
                        yield
                    finally:
                        self._connection = None
            except BaseException:
                # The transaction rolled back, so the cache may hold a step that was never stored.
                self._cached = ()
                raise

    def records(self) -> tuple[CommitRecord, ...]:
        with self._lock, self._transaction(write=False) as connection:
            return self._read(connection)

    def append(self, record: CommitRecord) -> None:
        encoded = json.dumps(record.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        with self._lock, self._transaction(write=True) as connection:
            records = self._read(connection)
            existing = next((item for item in records if item.step == record.step), None)
            if existing is not None:
                if same_commit(existing, record):
                    return
                raise CommitHistoryError(f"commit step {record.step} conflicts with its existing record")
            if records and record.step < records[-1].step:
                raise CommitHistoryError(f"commit step {record.step} precedes existing step {records[-1].step}")
            connection.execute(
                self.table.table.insert().values(**self.table.scope, step=record.step, commit_json=encoded)
            )

    @contextmanager
    def _transaction(self, *, write: bool) -> Iterator[Connection]:
        if self._connection is not None:
            yield self._connection
            return
        with self.record_store._transaction(write=write) as connection:
            yield connection

    def _read(self, connection: Connection) -> tuple[CommitRecord, ...]:
        table = self.table.table
        last_step = self._cached[-1].step if self._cached else 0
        rows = connection.execute(
            select(table.c.commit_json).where(self.table.condition(), table.c.step > last_step).order_by(table.c.step)
        ).scalars()
        added = tuple(CommitRecord.from_dict(json.loads(row)) for row in rows)
        if added:
            self._cached = (*self._cached, *added)
        return self._cached
