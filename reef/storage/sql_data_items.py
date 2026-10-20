"""The data item store in the record database of a scenario, for each SQL backend.

The data item table is in the same database as the records. Each database
operation uses one transaction of the record store. The files of a task are in
a ``TaskFileStore``. The adapters build the table with :func:`data_item_table`
and supply their namespace.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

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

from reef.core.batches import TaskItem, TrainDataItem, TrajectoryItem
from reef.storage.data_items import DataItemConflict, DataItemStore, StoredDataItem
from reef.storage.sql_records import SQLRecordStore
from reef.storage.tasks import TaskFileStore, files_digest


@dataclass(frozen=True)
class DataItemTable:
    """The data item table of one database, and the namespace of one store in it.

    Values: ``table`` is a table that ``data_item_table`` defines. ``scope`` maps
    column names of ``table`` to values. It is empty for a table without a
    namespace, and it names ``storage_id`` for a table that has that column.

    Purpose: a SQL adapter makes it. ``SQLDataItemStore`` restricts each
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


def data_item_table(metadata: MetaData, *, foreign_key: str | None = None) -> Table:
    """Define the ``train_data_item`` table in ``metadata``.

    If ``foreign_key`` names a column, the table also has the column
    ``storage_id``, which refers to it. The adapter then gives ``storage_id``
    in ``DataItemTable.scope``. Item IDs are unique in each namespace.
    """
    columns = (
        Column("item_id", Text, nullable=False),
        Column("kind", Text, nullable=False),
        Column("content_json", Text, nullable=False),
        Column("content_sha256", String(64), nullable=False),
    )
    if foreign_key is None:
        return Table(
            "train_data_item",
            metadata,
            Column("sequence", Integer, primary_key=True),
            *columns,
            UniqueConstraint("item_id"),
            sqlite_autoincrement=True,
        )
    return Table(
        "train_data_item",
        metadata,
        Column("sequence", BigInteger, Identity(), primary_key=True),
        Column("storage_id", String(32), ForeignKey(foreign_key), nullable=False),
        *columns,
        UniqueConstraint("storage_id", "item_id"),
        Index("train_data_item_storage_sequence", "storage_id", "sequence"),
    )


class SQLDataItemStore(DataItemStore):
    """The data items of one scenario in its record database, and the files of its tasks."""

    def __init__(self, records: SQLRecordStore, table: DataItemTable, files: TaskFileStore, scenario: str) -> None:
        self.records = records
        self.table = table
        self.files = files
        self.scenario = scenario

    def add(self, items: tuple[tuple[str, TrainDataItem], ...]) -> bool:
        rows = []
        for item_id, item in items:
            if not isinstance(item_id, str) or not item_id.strip():
                raise ValueError("a data item ID must be a non-empty string")
            kind, content = item_content(item)
            content_json = json.dumps(content, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            content_sha256 = hashlib.sha256(f"{kind}\n{content_json}".encode()).hexdigest()
            rows.append((item_id, item, kind, content_json, content_sha256))
        table = self.table.table
        inserted = False
        with self.records._transaction(write=True) as connection:
            for item_id, item, kind, content_json, content_sha256 in rows:
                stored = connection.execute(
                    select(table.c.content_sha256).where(self.table.condition(), table.c.item_id == item_id)
                ).scalar_one_or_none()
                if stored is not None:
                    if stored != content_sha256:
                        raise DataItemConflict(f"data item {item_id!r} already has different content")
                    continue
                if isinstance(item, TaskItem):
                    self.files.put(self.scenario, item.task_path)
                connection.execute(
                    table.insert().values(
                        **self.table.scope,
                        item_id=item_id,
                        kind=kind,
                        content_json=content_json,
                        content_sha256=content_sha256,
                    )
                )
                inserted = True
        return inserted

    def page(self, *, after_sequence: int = 0, limit: int = 256) -> tuple[StoredDataItem, ...]:
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
        return tuple(self.stored_item(row) for row in rows)

    def count(self, *, after_sequence: int = 0) -> int:
        table = self.table.table
        with self.records._transaction(write=False) as connection:
            statement = select(func.count()).select_from(table)
            return int(
                connection.execute(
                    statement.where(self.table.condition(), table.c.sequence > after_sequence)
                ).scalar_one()
            )

    def stored_item(self, row: RowMapping) -> StoredDataItem:
        content = json.loads(row["content_json"])
        sources = tuple(content["source_agent_record_ids"])
        item: TrainDataItem
        if row["kind"] == "trajectory":
            item = TrajectoryItem(content["trajectory"], group_id=content["group_id"], source_agent_record_ids=sources)
        else:
            item = TaskItem(
                self.files.path(self.scenario, content["digest"]),
                source_agent_record_ids=sources,
                metadata=content["metadata"],
            )
        return StoredDataItem(sequence=int(row["sequence"]), item_id=row["item_id"], item=item)


def item_content(item: TrainDataItem) -> tuple[str, dict[str, Any]]:
    """Return the kind of ``item`` and the JSON content that the table keeps for it.

    A task keeps the name and the digest of its directory instead of its local path.
    """
    if isinstance(item, TrajectoryItem):
        return "trajectory", {
            "trajectory": dict(item.trajectory),
            "group_id": item.group_id,
            "source_agent_record_ids": list(item.source_agent_record_ids),
        }
    return "task", {
        "name": item.task_path.name,
        "digest": files_digest(item.task_path),
        "source_agent_record_ids": list(item.source_agent_record_ids),
        "metadata": dict(item.metadata),
    }
