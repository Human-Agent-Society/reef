"""The evaluation store in the record database of a scenario, for each SQL backend.

The artifact evaluation table is in the same database as the commits. Each
database operation uses one transaction of the scenario database. The adapters build
the table with :func:`evaluation_table` and supply their namespace.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Literal

from sqlalchemy import (
    BigInteger,
    Column,
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    MetaData,
    Select,
    String,
    Table,
    Text,
    UniqueConstraint,
    select,
)
from sqlalchemy.engine import RowMapping

from reef.core.evaluation import ArtifactEvaluationRequest, PolicyEvaluation
from reef.storage.scenario.evaluations import (
    EvaluationStart,
    EvaluationStartConflictError,
    EvaluationStateConflictError,
    KeyedEvaluationStore,
    StoredEvaluation,
)
from reef.storage.sql.records import ScopedTable, SQLScenarioDatabase


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


def evaluation_digest_table(metadata: MetaData) -> Table:
    """The optional task digests of an evaluation, kept outside its list row."""
    return Table(
        "artifact_evaluation_digest",
        metadata,
        Column(
            "sequence", BigInteger, ForeignKey("artifact_evaluation.sequence", ondelete="CASCADE"), primary_key=True
        ),
        Column("task_digests_json", Text, nullable=False),
    )


def evaluation_start_table(metadata: MetaData, *, foreign_key: str | None = None) -> Table:
    """Define the start IDs of one scenario or of the stores that share the database."""
    scope = (
        () if foreign_key is None else (Column("storage_id", String(32), ForeignKey(foreign_key), primary_key=True),)
    )
    return Table(
        "artifact_evaluation_start",
        metadata,
        *scope,
        Column("start_id", Text, primary_key=True),
        Column("evaluation_id", Text, nullable=False),
        Column("fingerprint", String(64), nullable=False),
        Column("first_work_json", Text),
        Index("artifact_evaluation_start_evaluation", *(column.name for column in scope), "evaluation_id"),
    )


class SQLEvaluationStore(KeyedEvaluationStore):
    """The artifact evaluations of one scenario in its record database."""

    def __init__(self, database: SQLScenarioDatabase, table: ScopedTable, starts: ScopedTable, digests: Table) -> None:
        self.database = database
        self.table = table
        self.starts = starts
        self.digests = digests

    def selection(self, *, include_task_digests: bool = True) -> Select[Any]:
        """Select evaluations and, when requested, the digests keyed by their unique sequence."""
        table = self.table.table
        if include_task_digests:
            return select(table, self.digests.c.task_digests_json).outerjoin(
                self.digests, self.digests.c.sequence == table.c.sequence
            )
        return select(table)

    def create(
        self,
        *,
        evaluation_id: str,
        owner: str,
        request: ArtifactEvaluationRequest,
        created_at: float,
        state: Literal["queued", "running"] = "running",
        start: EvaluationStart | None = None,
    ) -> StoredEvaluation:
        if state not in ("queued", "running"):
            raise ValueError(f"an evaluation is created queued or running, not {state!r}")
        stored = StoredEvaluation(evaluation_id, owner, request, state, created_at)
        table = self.table.table
        with self.database._transaction(write=True) as connection:
            held = connection.execute(
                select(table.c.sequence).where(self.table.condition(), table.c.evaluation_id == evaluation_id)
            ).scalar_one_or_none()
            if held is not None:
                raise ValueError(f"evaluation {evaluation_id!r} is stored already")
            if start is not None and not self.database._write_if_absent(
                connection,
                self.starts.table,
                {
                    **self.starts.scope,
                    "start_id": start.start_id,
                    "evaluation_id": evaluation_id,
                    "fingerprint": start.fingerprint,
                    "first_work_json": start.first_work_json,
                },
            ):
                raise EvaluationStartConflictError("start_id is already stored")
            sequence = connection.execute(
                table.insert()
                .values(
                    **self.table.scope,
                    evaluation_id=evaluation_id,
                    owner=owner,
                    request_json=json.dumps(
                        request.to_json(include_task_digests=False), ensure_ascii=False, sort_keys=True
                    ),
                    state=state,
                    created_at=created_at,
                )
                .returning(table.c.sequence)
            ).scalar_one()
            if request.task_digests is not None:
                connection.execute(
                    self.digests.insert().values(sequence=sequence, task_digests_json=json.dumps(request.task_digests))
                )
        return stored

    def read_started(self, start_id: str, fingerprint: str) -> tuple[StoredEvaluation, str | None] | None:
        table, starts = self.table.table, self.starts.table
        with self.database._transaction(write=False) as connection:
            row = (
                connection.execute(
                    self.selection()
                    .add_columns(starts.c.fingerprint, starts.c.first_work_json)
                    .join_from(table, starts, table.c.evaluation_id == starts.c.evaluation_id)
                    .where(self.table.condition(), self.starts.condition(), starts.c.start_id == start_id)
                )
                .mappings()
                .one_or_none()
            )
        if row is None:
            return None
        if row["fingerprint"] != fingerprint:
            raise EvaluationStartConflictError("start_id already names a different evaluation request")
        return stored_evaluation(row), row["first_work_json"]

    def pending_starts(self) -> tuple[tuple[str, str], ...]:
        starts = self.starts.table
        with self.database._transaction(write=False) as connection:
            rows = connection.execute(
                select(starts.c.evaluation_id, starts.c.first_work_json).where(
                    self.starts.condition(), starts.c.first_work_json.is_not(None)
                )
            ).all()
        return tuple((evaluation_id, first_work_json) for evaluation_id, first_work_json in rows)

    def pending_start(self, evaluation_id: str) -> str | None:
        starts = self.starts.table
        with self.database._transaction(write=False) as connection:
            return connection.execute(
                select(starts.c.first_work_json).where(
                    self.starts.condition(), starts.c.evaluation_id == evaluation_id
                )
            ).scalar_one_or_none()

    def complete_start(self, evaluation_id: str) -> None:
        starts = self.starts.table
        with self.database._transaction(write=True) as connection:
            connection.execute(
                starts.update()
                .where(self.starts.condition(), starts.c.evaluation_id == evaluation_id)
                .values(first_work_json=None)
            )

    def begin(self, evaluation_id: str) -> StoredEvaluation:
        return self.move(evaluation_id, ("queued",), state="running", finished_at=None, result_json=None, error=None)

    def finish(self, evaluation_id: str, *, result: PolicyEvaluation, finished_at: float) -> StoredEvaluation:
        if not isinstance(result, PolicyEvaluation):
            raise TypeError("an evaluation finishes with a PolicyEvaluation")
        return self.move(
            evaluation_id,
            ("running",),
            state="finished",
            finished_at=finished_at,
            result_json=json.dumps(result.to_json(), ensure_ascii=False, sort_keys=True),
            error=None,
        )

    def fail(self, evaluation_id: str, *, error: str, finished_at: float) -> StoredEvaluation:
        if not isinstance(error, str) or not error:
            raise ValueError("an evaluation fails with a non-empty error")
        return self.move(
            evaluation_id,
            ("queued", "running"),
            state="failed",
            finished_at=finished_at,
            result_json=None,
            error=error,
        )

    def reopen(self, evaluation_id: str) -> StoredEvaluation:
        return self.move(
            evaluation_id, ("finished", "failed"), state="running", finished_at=None, result_json=None, error=None
        )

    def limit_rollouts(self, evaluation_id: str, max_concurrent_rollouts: int | None) -> StoredEvaluation:
        table = self.table.table
        with self.database._transaction(write=True) as connection:
            row = (
                connection.execute(
                    self.selection().where(self.table.condition(), table.c.evaluation_id == evaluation_id)
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                raise EvaluationStateConflictError(f"no evaluation {evaluation_id!r}")
            stored = stored_evaluation(row)
            # ``replace`` checks the new limit as a request checks it.
            request = replace(stored.request, max_concurrent_rollouts=max_concurrent_rollouts)
            request_json = json.dumps(request.to_json(include_task_digests=False), ensure_ascii=False, sort_keys=True)
            connection.execute(
                table.update()
                .where(self.table.condition(), table.c.evaluation_id == evaluation_id)
                .values(request_json=request_json)
            )
        return replace(stored, request=request)

    def move(
        self,
        evaluation_id: str,
        sources: tuple[str, ...],
        *,
        state: str,
        finished_at: float | None,
        result_json: str | None,
        error: str | None,
    ) -> StoredEvaluation:
        """Move an evaluation in one of ``sources`` to ``state`` in one transaction, and return it as stored."""
        table = self.table.table
        with self.database._transaction(write=True) as connection:
            row = (
                connection.execute(
                    self.selection().where(self.table.condition(), table.c.evaluation_id == evaluation_id)
                )
                .mappings()
                .one_or_none()
            )
            if row is None or row["state"] not in sources:
                raise EvaluationStateConflictError(f"evaluation {evaluation_id!r} is not {' or '.join(sources)}")
            moved = {"state": state, "finished_at": finished_at, "result_json": result_json, "error": error}
            # Check the moved row before it is written, so the table never holds an invalid evaluation.
            stored = stored_evaluation({**row, **moved})
            connection.execute(
                table.update().where(self.table.condition(), table.c.evaluation_id == evaluation_id).values(**moved)
            )
        return stored

    def read(self, evaluation_id: str) -> StoredEvaluation | None:
        table = self.table.table
        with self.database._transaction(write=False) as connection:
            row = (
                connection.execute(
                    self.selection().where(self.table.condition(), table.c.evaluation_id == evaluation_id)
                )
                .mappings()
                .one_or_none()
            )
        return None if row is None else stored_evaluation(row)

    def newest(self, *, limit: int = 50) -> tuple[StoredEvaluation, ...]:
        return self.select_newest(limit, include_task_digests=True)

    def newest_for_list(self, *, limit: int = 50) -> tuple[StoredEvaluation, ...]:
        return self.select_newest(limit, include_task_digests=False)

    def select_newest(self, limit: int, *, include_task_digests: bool) -> tuple[StoredEvaluation, ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        table = self.table.table
        with self.database._transaction(write=False) as connection:
            rows = (
                connection.execute(
                    self.selection(include_task_digests=include_task_digests)
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
    request = json.loads(str(row["request_json"]))
    if (task_digests_json := row.get("task_digests_json")) is not None:
        request["task_digests"] = json.loads(task_digests_json)
    return StoredEvaluation(
        evaluation_id=str(row["evaluation_id"]),
        owner=str(row["owner"]),
        request=ArtifactEvaluationRequest.from_json(request),
        state=row["state"],
        created_at=float(row["created_at"]),
        finished_at=None if row["finished_at"] is None else float(row["finished_at"]),
        result=None if result_json is None else PolicyEvaluation.from_json(json.loads(str(result_json))),
        error=None if row["error"] is None else str(row["error"]),
    )


__all__ = [
    "SQLEvaluationStore",
    "evaluation_digest_table",
    "evaluation_start_table",
    "evaluation_table",
    "stored_evaluation",
]
