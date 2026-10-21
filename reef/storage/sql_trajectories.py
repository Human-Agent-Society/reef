"""The trajectory store in the record database of a scenario, for each SQL backend.

The uploaded trajectory table is in the same database as the records. Each
database operation uses one transaction of the record store. The adapters build
the table with :func:`trajectory_table` and supply their namespace.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field

from sqlalchemy import (
    BigInteger,
    Column,
    ForeignKey,
    Identity,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    and_,
    func,
    select,
    true,
)
from sqlalchemy.engine import RowMapping
from sqlalchemy.sql.elements import ColumnElement

from reef.core.batches import TrajectoryItem
from reef.storage.sql_records import SQLRecordStore
from reef.storage.trajectories import StoredTrajectory, TrajectoryConflict, TrajectoryStore


@dataclass(frozen=True)
class TrajectoryTable:
    """The uploaded trajectory table of one database, and the namespace of one store in it.

    Values: ``table`` is a table that ``trajectory_table`` defines. ``scope``
    maps column names of ``table`` to values. It is empty for a table without a
    namespace, and it names ``storage_id`` for a table that has that column.

    Purpose: a SQL adapter makes it. ``SQLTrajectoryStore`` restricts each
    operation to the rows that match ``scope``.
    """

    table: Table
    scope: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.scope) - set(self.table.c.keys())
        if unknown:
            raise ValueError(f"the scope names columns that {self.table.name} does not have: {sorted(unknown)!r}")

    def condition(self) -> ColumnElement[bool]:
        """Restrict an operation to the namespace of the adapter, if any."""
        return and_(true(), *(self.table.c[key] == value for key, value in self.scope.items()))


def trajectory_table(metadata: MetaData, *, foreign_key: str | None = None) -> Table:
    """Define the ``uploaded_trajectory`` table in ``metadata``.

    If ``foreign_key`` names a column, the table also has the column
    ``storage_id``, which refers to it. The adapter then gives ``storage_id``
    in ``TrajectoryTable.scope``. Trajectory IDs are unique in each namespace.
    """
    columns = (
        Column("trajectory_id", Text, nullable=False),
        Column("content_json", Text, nullable=False),
        Column("content_sha256", String(64), nullable=False),
    )
    if foreign_key is None:
        return Table(
            "uploaded_trajectory",
            metadata,
            Column("sequence", Integer, primary_key=True),
            *columns,
            UniqueConstraint("trajectory_id"),
            sqlite_autoincrement=True,
        )
    return Table(
        "uploaded_trajectory",
        metadata,
        Column("sequence", BigInteger, Identity(), primary_key=True),
        Column("storage_id", String(32), ForeignKey(foreign_key), nullable=False),
        *columns,
        UniqueConstraint("storage_id", "trajectory_id"),
        Index("uploaded_trajectory_storage_sequence", "storage_id", "sequence"),
    )


class SQLTrajectoryStore(TrajectoryStore):
    """The uploaded trajectories of one scenario in its record database."""

    def __init__(self, records: SQLRecordStore, table: TrajectoryTable) -> None:
        self.records = records
        self.table = table

    def add(self, trajectories: tuple[tuple[str, TrajectoryItem], ...]) -> bool:
        rows = []
        for trajectory_id, trajectory in trajectories:
            if not isinstance(trajectory_id, str) or not trajectory_id.strip():
                raise ValueError("a trajectory ID must be a non-empty string")
            content = {
                "trajectory": dict(trajectory.trajectory),
                "group_id": trajectory.group_id,
                "source_agent_record_ids": list(trajectory.source_agent_record_ids),
            }
            content_json = json.dumps(content, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            rows.append((trajectory_id, content_json, hashlib.sha256(content_json.encode()).hexdigest()))
        table = self.table.table
        inserted = False
        with self.records._transaction(write=True) as connection:
            for trajectory_id, content_json, content_sha256 in rows:
                stored = connection.execute(
                    select(table.c.content_sha256).where(
                        self.table.condition(), table.c.trajectory_id == trajectory_id
                    )
                ).scalar_one_or_none()
                if stored is not None:
                    if stored != content_sha256:
                        raise TrajectoryConflict(f"trajectory {trajectory_id!r} is already stored with other content")
                    continue
                connection.execute(
                    table.insert().values(
                        **self.table.scope,
                        trajectory_id=trajectory_id,
                        content_json=content_json,
                        content_sha256=content_sha256,
                    )
                )
                inserted = True
        return inserted

    def page(self, *, after_sequence: int = 0, limit: int = 256) -> tuple[StoredTrajectory, ...]:
        if after_sequence < 0 or limit <= 0:
            raise ValueError("after_sequence must be non-negative and limit positive")
        table = self.table.table
        with self.records._transaction(write=False) as connection:
            rows = (
                connection.execute(
                    select(table)
                    .where(self.table.condition(), table.c.sequence > after_sequence)
                    .order_by(table.c.sequence)
                    .limit(limit)
                )
                .mappings()
                .all()
            )
        return tuple(stored_trajectory(row) for row in rows)

    def count(self, *, after_sequence: int = 0) -> int:
        table = self.table.table
        with self.records._transaction(write=False) as connection:
            statement = select(func.count()).select_from(table)
            return int(
                connection.execute(
                    statement.where(self.table.condition(), table.c.sequence > after_sequence)
                ).scalar_one()
            )


def stored_trajectory(row: RowMapping) -> StoredTrajectory:
    """Return the stored trajectory of one table row."""
    content = json.loads(row["content_json"])
    trajectory = TrajectoryItem(
        content["trajectory"],
        group_id=content["group_id"],
        source_agent_record_ids=tuple(content["source_agent_record_ids"]),
    )
    return StoredTrajectory(sequence=int(row["sequence"]), trajectory_id=row["trajectory_id"], trajectory=trajectory)
