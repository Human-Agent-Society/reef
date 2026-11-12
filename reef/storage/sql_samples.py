"""The training state of a scenario in its record database, for each SQL backend (RFC #1140).

The consumed record table is in the same database as the commits. The train counts have no table: each training
commit lists the samples that it consumed, and ``SQLTrainingState`` counts them from the commit table. The adapters
build the stage record table with :func:`stage_record_table` and supply their namespace.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from threading import Lock

from sqlalchemy import Column, ForeignKey, MetaData, String, Table, Text, select

from reef.storage.commits import CommitRecord
from reef.storage.samples import CONSUMED_SAMPLE_PREFIX, TrainingState
from reef.storage.sql_records import ScopedTable, SQLScenarioDatabase


def scoped_table(name: str, metadata: MetaData, columns: tuple[Column, ...], foreign_key: str | None) -> Table:
    """The table ``name`` of ``columns``; ``foreign_key`` adds the ``storage_id`` column of a PostgreSQL store
    generation to its primary key."""
    if foreign_key is None:
        return Table(name, metadata, *columns)
    return Table(name, metadata, Column("storage_id", String(32), ForeignKey(foreign_key), primary_key=True), *columns)


def stage_record_table(metadata: MetaData, *, foreign_key: str | None = None) -> Table:
    """Define the ``stage_record`` table in ``metadata``: the records that each trainer's stages consumed.

    A row ``(trainer, agent_record_id)`` says that the stages of ``trainer`` turned the record into samples, or ended
    it with them.
    """
    columns = (Column("trainer", Text, primary_key=True), Column("agent_record_id", Text, primary_key=True))
    return scoped_table("stage_record", metadata, columns, foreign_key)


class SQLTrainingState(TrainingState):
    """The training state of one scenario in its record database.

    Values: ``commits`` is the commit table and ``consumed`` the stage record table, both in the namespace of the
    scenario. ``counts`` holds, for each sample ID, the number of commits of each trainer that list the sample
    (``consumed_sample_id``), over the commits up to step ``read_step``; the trainer of a commit is its ``component``,
    or ``""`` when it names none.
    Purpose: a SQL scenario database makes it in ``training_state()``; ``SampleStore`` reads it. ``train_counts``
    reads the commits after ``read_step`` first, so the counts follow the commit history with no table of their own.
    """

    def __init__(self, database: SQLScenarioDatabase, commits: ScopedTable, consumed: ScopedTable) -> None:
        self.database = database
        self.commits = commits
        self.consumed = consumed
        self.counts: dict[str, dict[str, int]] = {}
        self.read_step = 0
        self.lock = Lock()

    def train_counts(self, sample_ids: Sequence[str]) -> dict[str, dict[str, int]]:
        if not sample_ids:
            return {}
        table = self.commits.table
        with self.lock:
            with self.database._transaction(write=False) as connection:
                rows = connection.execute(
                    select(table.c.step, table.c.commit_json)
                    .where(self.commits.condition(), table.c.step > self.read_step)
                    .order_by(table.c.step)
                ).all()
            for step, commit_json in rows:
                record = CommitRecord.from_dict(json.loads(commit_json))
                trainer = record.component or ""
                for consumed in record.consumed_ids:
                    if consumed.startswith(CONSUMED_SAMPLE_PREFIX):
                        by_trainer = self.counts.setdefault(consumed.removeprefix(CONSUMED_SAMPLE_PREFIX), {})
                        by_trainer[trainer] = by_trainer.get(trainer, 0) + 1
                self.read_step = step
            return {sample_id: dict(self.counts[sample_id]) for sample_id in sample_ids if sample_id in self.counts}

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
                self.database._write_if_absent(connection, self.consumed.table, rows)


__all__ = ["SQLTrainingState", "stage_record_table"]
