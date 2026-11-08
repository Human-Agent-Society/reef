"""The evaluation store in the record database of a scenario, for each SQL backend.

The artifact evaluation table and the evaluation call table are in the same
database as the commits. Each database operation uses one transaction of the
scenario database. The adapters build the tables with :func:`evaluation_table`
and :func:`evaluation_call_table`, and supply their namespace.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
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
    String,
    Table,
    Text,
    UniqueConstraint,
    case,
    func,
    select,
)
from sqlalchemy.engine import RowMapping

from reef.core.evaluation import ArtifactEvaluationRequest, PolicyEvaluation
from reef.storage.evaluations import (
    EvaluationCall,
    EvaluationCallSummary,
    EvaluationCallTotals,
    EvaluationStateConflict,
    EvaluationStore,
    StoredEvaluation,
)
from reef.storage.sql_records import ScopedTable, SQLScenarioDatabase


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


def evaluation_call_table(metadata: MetaData, *, foreign_key: str | None = None) -> Table:
    """Define the ``evaluation_call`` table in ``metadata``: one row for each kept call of an evaluation rollout.

    ``foreign_key`` adds the column ``storage_id``, as in :func:`evaluation_table`. The key of a call
    (``evaluation_id``, ``task_index``, ``repeat``, ``attempt``, ``call_index``) is unique in each namespace. The
    table refers to no evaluation row: a worker can keep a call before control stores its evaluation.
    """
    key = ("evaluation_id", "task_index", "repeat", "attempt", "call_index")
    columns = (
        Column("evaluation_id", Text, nullable=False),
        Column("task_index", Integer, nullable=False),
        Column("repeat", Integer, nullable=False),
        Column("attempt", Integer, nullable=False),
        Column("call_index", Integer, nullable=False),
        Column("path", Text, nullable=False),
        Column("status", Integer, nullable=False),
        Column("started_at", Float(53), nullable=False),
        Column("duration_s", Float(53), nullable=False),
        Column("prompt_tokens", BigInteger),
        Column("completion_tokens", BigInteger),
        Column("finish_reason", Text),
        Column("tool_calls", Integer, nullable=False),
        Column("request_json", Text),
        Column("response_json", Text),
    )
    if foreign_key is None:
        return Table(
            "evaluation_call",
            metadata,
            Column("sequence", Integer, primary_key=True),
            *columns,
            UniqueConstraint(*key),
            sqlite_autoincrement=True,
        )
    return Table(
        "evaluation_call",
        metadata,
        Column("sequence", BigInteger, Identity(), primary_key=True),
        Column("storage_id", String(32), ForeignKey(foreign_key), nullable=False),
        *columns,
        UniqueConstraint("storage_id", *key),
    )


class SQLEvaluationStore(EvaluationStore):
    """The artifact evaluations of one scenario and their calls in its record database."""

    def __init__(self, database: SQLScenarioDatabase, table: ScopedTable, call_table: ScopedTable) -> None:
        self.database = database
        self.table = table
        self.call_table = call_table

    def create(
        self,
        *,
        evaluation_id: str,
        owner: str,
        request: ArtifactEvaluationRequest,
        created_at: float,
        state: Literal["queued", "running"] = "running",
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
            connection.execute(
                table.insert().values(
                    **self.table.scope,
                    evaluation_id=evaluation_id,
                    owner=owner,
                    request_json=json.dumps(request.to_json(), ensure_ascii=False, sort_keys=True),
                    state=state,
                    created_at=created_at,
                )
            )
        return stored

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
                connection.execute(select(table).where(self.table.condition(), table.c.evaluation_id == evaluation_id))
                .mappings()
                .one_or_none()
            )
            if row is None or row["state"] not in sources:
                raise EvaluationStateConflict(f"evaluation {evaluation_id!r} is not {' or '.join(sources)}")
            moved = {"state": state, "finished_at": finished_at, "result_json": result_json, "error": error}
            # Check the moved row before it is written, so the table never holds an invalid evaluation.
            stored = stored_evaluation({**row, **moved})
            connection.execute(
                table.update().where(self.table.condition(), table.c.evaluation_id == evaluation_id).values(**moved)
            )
        return stored

    def get(self, evaluation_id: str) -> StoredEvaluation | None:
        table = self.table.table
        with self.database._transaction(write=False) as connection:
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
        with self.database._transaction(write=False) as connection:
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

    def add_call(self, evaluation_id: str, call: EvaluationCall) -> None:
        summary = call.summary
        values = {
            **self.call_table.scope,
            "evaluation_id": evaluation_id,
            "task_index": summary.task_index,
            "repeat": summary.repeat,
            "attempt": summary.attempt,
            "call_index": summary.index,
            "path": summary.path,
            "status": summary.status,
            "started_at": summary.started_at,
            "duration_s": summary.duration_s,
            "prompt_tokens": summary.prompt_tokens,
            "completion_tokens": summary.completion_tokens,
            "finish_reason": summary.finish_reason,
            "tool_calls": summary.tool_calls,
            "request_json": None if call.request is None else json.dumps(call.request, ensure_ascii=False),
            "response_json": None if call.response is None else json.dumps(call.response, ensure_ascii=False),
        }
        with self.database._transaction(write=True) as connection:
            if not self.database._insert_if_absent(connection, self.call_table.table, values):
                raise ValueError(
                    f"call {summary.attempt}/{summary.index} of rollout {summary.task_index}/{summary.repeat} "
                    f"of evaluation {evaluation_id!r} is stored already"
                )

    def call_totals(self, evaluation_id: str) -> dict[tuple[int, int], EvaluationCallTotals]:
        table = self.call_table.table
        statement = (
            select(
                table.c.task_index,
                table.c.repeat,
                func.count(),
                func.sum(case((table.c.status != 200, 1), else_=0)),
                func.coalesce(func.sum(table.c.prompt_tokens), 0),
                func.coalesce(func.sum(table.c.completion_tokens), 0),
            )
            .where(self.call_table.condition(), table.c.evaluation_id == evaluation_id)
            .group_by(table.c.task_index, table.c.repeat)
        )
        with self.database._transaction(write=False) as connection:
            rows = connection.execute(statement).all()
        return {
            (int(task_index), int(repeat)): EvaluationCallTotals(int(count), int(failed), int(prompt), int(completion))
            for task_index, repeat, count, failed, prompt, completion in rows
        }

    def call_summaries(self, evaluation_id: str, task_index: int, repeat: int) -> tuple[EvaluationCallSummary, ...]:
        table = self.call_table.table
        bodies = {table.c.request_json, table.c.response_json}
        statement = (
            select(*(column for column in table.c if column not in bodies))
            .where(
                self.call_table.condition(),
                table.c.evaluation_id == evaluation_id,
                table.c.task_index == task_index,
                table.c.repeat == repeat,
            )
            .order_by(table.c.attempt, table.c.call_index)
        )
        with self.database._transaction(write=False) as connection:
            rows = connection.execute(statement).mappings().all()
        return tuple(call_summary(row) for row in rows)

    def call(
        self, evaluation_id: str, task_index: int, repeat: int, attempt: int, index: int
    ) -> EvaluationCall | None:
        table = self.call_table.table
        statement = select(table).where(
            self.call_table.condition(),
            table.c.evaluation_id == evaluation_id,
            table.c.task_index == task_index,
            table.c.repeat == repeat,
            table.c.attempt == attempt,
            table.c.call_index == index,
        )
        with self.database._transaction(write=False) as connection:
            row = connection.execute(statement).mappings().one_or_none()
        if row is None:
            return None
        request, response = row["request_json"], row["response_json"]
        return EvaluationCall(
            call_summary(row),
            None if request is None else json.loads(str(request)),
            None if response is None else json.loads(str(response)),
        )


def call_summary(row: RowMapping) -> EvaluationCallSummary:
    """Return the call summary of one ``evaluation_call`` row."""
    return EvaluationCallSummary(
        task_index=int(row["task_index"]),
        repeat=int(row["repeat"]),
        attempt=int(row["attempt"]),
        index=int(row["call_index"]),
        path=str(row["path"]),
        status=int(row["status"]),
        started_at=float(row["started_at"]),
        duration_s=float(row["duration_s"]),
        prompt_tokens=None if row["prompt_tokens"] is None else int(row["prompt_tokens"]),
        completion_tokens=None if row["completion_tokens"] is None else int(row["completion_tokens"]),
        finish_reason=None if row["finish_reason"] is None else str(row["finish_reason"]),
        tool_calls=int(row["tool_calls"]),
    )


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


__all__ = ["SQLEvaluationStore", "evaluation_call_table", "evaluation_table", "stored_evaluation"]
