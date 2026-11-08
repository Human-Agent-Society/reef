"""The training state of a scenario in its record database, for each SQL backend (RFC #1140).

The train count table and the consumed record table are in the same database as the commits. The
commit history counts the samples that a commit consumed in the transaction of the commit (``SQLCommitHistory``).
The adapters build the tables with :func:`sample_train_count_table` and :func:`stage_record_table` and supply their
namespace.
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import Column, ForeignKey, Integer, MetaData, String, Table, Text, select

from reef.storage.samples import TrainingState
from reef.storage.sql_records import ScopedTable, SQLScenarioDatabase


def scoped_table(name: str, metadata: MetaData, columns: tuple[Column, ...], foreign_key: str | None) -> Table:
    """The table ``name`` of ``columns``; ``foreign_key`` adds the ``storage_id`` column of a PostgreSQL store
    generation to its primary key."""
    if foreign_key is None:
        return Table(name, metadata, *columns)
    return Table(name, metadata, Column("storage_id", String(32), ForeignKey(foreign_key), primary_key=True), *columns)


def sample_train_count_table(metadata: MetaData, *, foreign_key: str | None = None) -> Table:
    """Define the ``sample_train_count`` table in ``metadata``: how many committed jobs of each trainer held a sample.

    A row ``(sample_id, component, train_count)`` says that ``train_count`` committed jobs of the trainer
    ``component`` listed ``consumed_sample_id(sample_id)``. ``component`` is the component that the commits name, or
    ``""`` for the only trainer of a scenario, whose commits name none. A missing row is a count of 0.
    """
    columns = (
        Column("sample_id", Text, primary_key=True),
        Column("component", Text, primary_key=True),
        Column("train_count", Integer, nullable=False),
    )
    return scoped_table("sample_train_count", metadata, columns, foreign_key)


def stage_record_table(metadata: MetaData, *, foreign_key: str | None = None) -> Table:
    """Define the ``stage_record`` table in ``metadata``: the records that each trainer's stages consumed.

    A row ``(trainer, agent_record_id)`` says that the stages of ``trainer`` turned the record into samples, or ended
    it with them.
    """
    columns = (Column("trainer", Text, primary_key=True), Column("agent_record_id", Text, primary_key=True))
    return scoped_table("stage_record", metadata, columns, foreign_key)


class SQLTrainingState(TrainingState):
    """The training state of one scenario in its record database: ``train_counts`` is its train count table and
    ``consumed`` its stage record table, both in the namespace of the scenario."""

    def __init__(self, database: SQLScenarioDatabase, train_counts: ScopedTable, consumed: ScopedTable) -> None:
        self.database = database
        self.counts = train_counts
        self.consumed = consumed

    def train_counts(self, sample_ids: Sequence[str]) -> dict[str, dict[str, int]]:
        if not sample_ids:
            return {}
        table = self.counts.table
        counts: dict[str, dict[str, int]] = {}
        with self.database._transaction(write=False) as connection:
            for row in connection.execute(
                select(table.c.sample_id, table.c.component, table.c.train_count).where(
                    self.counts.condition(), table.c.sample_id.in_(list(sample_ids)), table.c.train_count > 0
                )
            ):
                counts.setdefault(row.sample_id, {})[row.component] = row.train_count
        return counts

    def consumed_record_ids(self, trainer: str) -> frozenset[str]:
        table = self.consumed.table
        with self.database._transaction(write=False) as connection:
            return frozenset(
                connection.execute(
                    select(table.c.agent_record_id).where(self.consumed.condition(), table.c.trainer == trainer)
                ).scalars()
            )

    def consume_records(self, trainer: str, record_ids: Sequence[str]) -> None:
        rows = [
            {**self.consumed.scope, "trainer": trainer, "agent_record_id": record_id}
            for record_id in dict.fromkeys(record_ids)
        ]
        if rows:
            with self.database._transaction(write=True) as connection:
                self.database._insert_if_absent(connection, self.consumed.table, rows)


__all__ = ["SQLTrainingState", "sample_train_count_table", "stage_record_table"]
