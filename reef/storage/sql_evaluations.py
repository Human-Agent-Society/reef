"""The evaluation store in the record database of a scenario, for each SQL backend.

The artifact evaluation table is in the same database as the records. Each
database operation uses one transaction of the record store. The adapters build
the table with :func:`evaluation_table` and supply their namespace.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from sqlalchemy import (
    BigInteger,
    Column,
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    select,
)
from sqlalchemy.engine import RowMapping

from reef.core.evaluation import ArtifactEvaluationRequest, PolicyEvaluation
from reef.storage.evaluations import EvaluationNotRunning, EvaluationStore, StoredEvaluation
from reef.storage.sql_records import ScopedTable, SQLRecordStore


def evaluation_table(metadata: MetaData, *, foreign_key: str | None = None) -> Table:
    """Define the ``artifact_evaluation`` table in ``metadata``.

    If ``foreign_key`` names a column, the table also has the column
    ``storage_id``, which refers to it. The adapter then gives ``storage_id``
    in ``ScopedTable.scope``. Evaluation IDs are unique in each namespace.
    """
    columns = (
        Column("evaluation_id", Text, nullable=False),
        Column("owner", Text, nullable=False),
        Column("request_json", Text, nullable=False),
        Column("state", String(16), nullable=False),
        Column("created_at", Float(53), nullable=False),
        Column("finished_at", Float(53)),
        Column("result_json", Text),
        Column("error", Text),
    )
    if foreign_key is None:
        return Table(
            "artifact_evaluation",
            metadata,
            Column("sequence", Integer, primary_key=True),
            *columns,
            UniqueConstraint("evaluation_id"),
            sqlite_autoincrement=True,
        )
    return Table(
        "artifact_evaluation",
        metadata,
        Column("sequence", BigInteger, Identity(), primary_key=True),
        Column("storage_id", String(32), ForeignKey(foreign_key), nullable=False),
        *columns,
        UniqueConstraint("storage_id", "evaluation_id"),
        Index("artifact_evaluation_storage_created", "storage_id", "created_at"),
    )


class SQLEvaluationStore(EvaluationStore):
    """The artifact evaluations of one scenario in its record database."""

    def __init__(self, records: SQLRecordStore, table: ScopedTable) -> None:
        self.records = records
        self.table = table

    def create(
        self, *, evaluation_id: str, owner: str, request: ArtifactEvaluationRequest, created_at: float
    ) -> StoredEvaluation:
        stored = StoredEvaluation(evaluation_id, owner, request, "running", created_at)
        table = self.table.table
        with self.records._transaction(write=True) as connection:
            held = connection.execute(
                select(table.c.sequence).where(self.table.condition(), table.c.evaluation_id == evaluation_id)
            ).scalar_one_or_none()
            if held is not None:
                raise ValueError(f"evaluation {evaluation_id!r} is stored already")
            connection.execute(
                table.insert().values(
                    **self.table.scope,
                    evaluation_id=evaluation_id,
                    owner=owner,
                    request_json=json.dumps(request.to_json(), ensure_ascii=False, sort_keys=True),
                    state="running",
                    created_at=created_at,
                )
            )
        return stored

    def finish(self, evaluation_id: str, *, result: PolicyEvaluation, finished_at: float) -> StoredEvaluation:
        if not isinstance(result, PolicyEvaluation):
            raise TypeError("an evaluation finishes with a PolicyEvaluation")
        return self.end(
            evaluation_id,
            state="finished",
            finished_at=finished_at,
            result_json=json.dumps(result.to_json(), ensure_ascii=False, sort_keys=True),
            error=None,
        )

    def fail(self, evaluation_id: str, *, error: str, finished_at: float) -> StoredEvaluation:
        if not isinstance(error, str) or not error:
            raise ValueError("an evaluation fails with a non-empty error")
        return self.end(evaluation_id, state="failed", finished_at=finished_at, result_json=None, error=error)

    def end(
        self, evaluation_id: str, *, state: str, finished_at: float, result_json: str | None, error: str | None
    ) -> StoredEvaluation:
        """Move a running evaluation to ``state`` in one transaction, and return it as stored."""
        table = self.table.table
        with self.records._transaction(write=True) as connection:
            row = (
                connection.execute(select(table).where(self.table.condition(), table.c.evaluation_id == evaluation_id))
                .mappings()
                .one_or_none()
            )
            if row is None or row["state"] != "running":
                raise EvaluationNotRunning(f"evaluation {evaluation_id!r} is not running")
            ended = {"state": state, "finished_at": finished_at, "result_json": result_json, "error": error}
            # Check the ended row before it is written, so the table never holds an invalid evaluation.
            stored = stored_evaluation({**row, **ended})
            connection.execute(
                table.update().where(self.table.condition(), table.c.evaluation_id == evaluation_id).values(**ended)
            )
        return stored

    def get(self, evaluation_id: str) -> StoredEvaluation | None:
        table = self.table.table
        with self.records._transaction(write=False) as connection:
            row = (
                connection.execute(select(table).where(self.table.condition(), table.c.evaluation_id == evaluation_id))
                .mappings()
                .one_or_none()
            )
        return None if row is None else stored_evaluation(row)

    def newest(self, *, limit: int = 50) -> tuple[StoredEvaluation, ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        table = self.table.table
        with self.records._transaction(write=False) as connection:
            rows = (
                connection.execute(
                    select(table)
                    .where(self.table.condition())
                    .order_by(table.c.created_at.desc(), table.c.sequence.desc())
                    .limit(limit)
                )
                .mappings()
                .all()
            )
        return tuple(stored_evaluation(row) for row in rows)


def stored_evaluation(row: RowMapping | Mapping[str, Any]) -> StoredEvaluation:
    """Return the stored evaluation of one table row."""
    result_json = row["result_json"]
    return StoredEvaluation(
        evaluation_id=str(row["evaluation_id"]),
        owner=str(row["owner"]),
        request=ArtifactEvaluationRequest.from_json(json.loads(str(row["request_json"]))),
        state=row["state"],
        created_at=float(row["created_at"]),
        finished_at=None if row["finished_at"] is None else float(row["finished_at"]),
        result=None if result_json is None else PolicyEvaluation.from_json(json.loads(str(result_json))),
        error=None if row["error"] is None else str(row["error"]),
    )


__all__ = ["SQLEvaluationStore", "evaluation_table", "stored_evaluation"]
