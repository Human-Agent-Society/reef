"""The coordination tables of a deployment in a SQL database, for SQLite and PostgreSQL.

The services of a deployment hand work to each other through these tables:

- ``served_model`` and ``scheduled_stop`` hold ``served_models`` (``reef.storage.served_models``);
- ``control_lease`` holds the control lease (``reef.storage.control_lease``);
- ``work_item`` holds the work queues (``reef.storage.work_queue``);
- ``coordination_mark`` holds the marks (``reef.storage.marks``);
- ``coordination_sequence`` holds two counters: ``WORK_SEQUENCE`` numbers the changes of the work queues, and
  ``SERVED_SEQUENCE`` the changes of ``served_models`` and its stops.

A write that numbers its change takes its counter's row first. So the writes of one counter are serialized, and
the sequence order of their changes is the order in which they commit: a reader that reads the changes after a
sequence never misses a later commit. A write that numbers nothing (a lease extension, a mark) takes no counter, so
it never waits behind the writes of a counter.
"""

from __future__ import annotations

import hashlib
import json
from abc import abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import cast

from sqlalchemy import (
    URL,
    BigInteger,
    Column,
    ColumnElement,
    Float,
    Integer,
    MetaData,
    PrimaryKeyConstraint,
    String,
    Table,
    Text,
    and_,
    create_engine,
    delete,
    exists,
    func,
    insert,
    not_,
    select,
    update,
)
from sqlalchemy.engine import Connection, Engine, RowMapping
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import NullPool
from sqlalchemy.schema import CreateSchema

from reef.core.errors import ReefError
from reef.core.served_models import ScenarioModel, ScheduledStop, ServedArtifact, require_text
from reef.storage.control_lease import ControlLease, ControlLeaseState
from reef.storage.coordination import Coordination, CoordinationResource
from reef.storage.marks import Marks, check_mark_name
from reef.storage.postgres import postgres_url, validate_postgres_schema
from reef.storage.served_models import ControlLeaseConflict, ServedModelsWriter
from reef.storage.sqlite import enable_wal
from reef.storage.work_queue import (
    FINAL_STATES,
    HELD_STATES,
    JsonValue,
    NewWork,
    WorkConflict,
    WorkItem,
    WorkLeaseLost,
    WorkQueue,
    clock,
)


@dataclass(frozen=True)
class CoordinationTables:
    """The tables of one coordination database.

    Values: six ``Table`` objects of one ``MetaData``, defined by ``coordination_tables``.

    Purpose: a ``CoordinationDatabase`` makes it; the coordination stores query it.
    """

    sequence: Table
    served_model: Table
    scheduled_stop: Table
    control_lease: Table
    work_item: Table
    mark: Table


def coordination_tables(metadata: MetaData) -> CoordinationTables:
    """Define the coordination tables in ``metadata``. Times are seconds since the epoch; JSON is text."""
    sequence = Table(
        "coordination_sequence",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("value", BigInteger, nullable=False),
    )
    served_model = Table(
        "served_model",
        metadata,
        Column("scenario", Text, nullable=False),
        Column("name", Text, nullable=False),
        Column("artifact_id", Text, nullable=False),
        Column("url", Text, nullable=False),
        Column("model_name", Text, nullable=False),
        PrimaryKeyConstraint("scenario", "name"),
    )
    scheduled_stop = Table(
        "scheduled_stop",
        metadata,
        Column("artifact_id", Text, nullable=False),
        Column("url", Text, nullable=False),
        Column("model_name", Text, nullable=False),
        Column("stop_after", Float, nullable=False),
        PrimaryKeyConstraint("artifact_id", "url", "model_name"),
    )
    control_lease = Table(
        "control_lease",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("instance_id", Text, nullable=False),
        Column("number", BigInteger, nullable=False),
        Column("expires_at", Float, nullable=False),
    )
    work_item = Table(
        "work_item",
        metadata,
        Column("queue", Text, nullable=False),
        Column("item_id", Text, nullable=False),
        Column("scenario", Text, nullable=False),
        Column("kind", Text, nullable=False),
        Column("payload_json", Text, nullable=False),
        Column("state", String(16), nullable=False),
        Column("serial_key", Text, nullable=True, index=True),
        Column("available_at", Float, nullable=False),
        Column("created_at", Float, nullable=False),
        Column("attempt_count", Integer, nullable=False),
        Column("worker_id", Text, nullable=True),
        Column("lease_expires_at", Float, nullable=True),
        Column("progress_json", Text, nullable=False),
        Column("result_json", Text, nullable=True),
        Column("error", Text, nullable=True),
        Column("created_sequence", BigInteger, nullable=False, index=True),
        Column("changed_sequence", BigInteger, nullable=False, index=True),
        PrimaryKeyConstraint("queue", "item_id"),
    )
    mark = Table(
        "coordination_mark",
        metadata,
        Column("name", Text, primary_key=True),
        Column("value", BigInteger, nullable=False),
    )
    return CoordinationTables(sequence, served_model, scheduled_stop, control_lease, work_item, mark)


#: The counter row that numbers the changes of the work queues.
WORK_SEQUENCE = 1
#: The counter row that numbers the changes of ``served_models`` and its scheduled stops.
SERVED_SEQUENCE = 2


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class CoordinationDatabase(CoordinationResource):
    """One database that holds the coordination tables, and its transactions.

    Values: ``tables`` are the coordination tables in this database; the one row of the ``coordination_sequence``
    table holds the last change number. A subclass gives transactions (``connection``) and whether a read must
    lock rows to serialize writers (``locks_rows``).

    Purpose: ``SQLServedModels``, ``SQLWorkQueue`` and ``SQLControlLease`` share one database object, so a
    deployment opens one engine for all three.
    """

    def __init__(self, tables: CoordinationTables) -> None:
        self.tables = tables

    @abstractmethod
    @contextmanager
    def connection(self, *, write: bool = False) -> Iterator[Connection]:
        """One transaction: commit when the block ends, roll back when it raises. ``write`` says that the
        transaction writes, so a database without row locks takes its write lock at the start."""

    @property
    @abstractmethod
    def locks_rows(self) -> bool:
        """Whether ``SELECT ... FOR UPDATE`` is needed to serialize transactions (it is for PostgreSQL)."""

    @contextmanager
    def write(self, sequence: int) -> Iterator[tuple[Connection, SequenceCounter]]:
        """A write transaction that holds the counter ``sequence`` (``WORK_SEQUENCE`` or ``SERVED_SEQUENCE``) until
        it ends."""
        with self.connection(write=True) as connection:
            query = select(self.tables.sequence.c.value).where(self.tables.sequence.c.id == sequence)
            if self.locks_rows:
                query = query.with_for_update()
            value = connection.execute(query).scalar_one()
            counter = SequenceCounter(int(value))
            yield connection, counter
            if counter.value != value:
                connection.execute(
                    update(self.tables.sequence)
                    .where(self.tables.sequence.c.id == sequence)
                    .values(value=counter.value)
                )

    @contextmanager
    def read(self) -> Iterator[Connection]:
        with self.connection() as connection:
            yield connection

    def create(self, connection: Connection, metadata: MetaData) -> None:
        metadata.create_all(connection)
        present = set(connection.execute(select(self.tables.sequence.c.id)).scalars())
        for sequence in (WORK_SEQUENCE, SERVED_SEQUENCE):
            if sequence not in present:
                connection.execute(insert(self.tables.sequence).values(id=sequence, value=0))

    @abstractmethod
    def close(self) -> None:
        """Release the connections."""


class SequenceCounter:
    """The change counter during one write transaction.

    Values: ``value`` is a non-negative integer, the last change number that the transaction has given out; it
    starts at the stored counter and increases by one at each ``next``.

    Purpose: ``CoordinationDatabase.write`` makes one per transaction and stores ``value`` when the transaction
    ends; the stores number each change with ``next``, so the change order is the commit order.
    """

    def __init__(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("a change counter is a non-negative integer")
        self.value = value

    def next(self) -> int:
        self.value += 1
        return self.value


class SQLiteCoordinationDatabase(CoordinationDatabase):
    """The coordination tables in a SQLite file, ``<directory>/coordination.sqlite3``, or in memory without a
    directory. It holds one connection and serializes its use with a lock, as the SQLite data store does."""

    def __init__(self, directory: Path | None) -> None:
        metadata = MetaData()
        super().__init__(coordination_tables(metadata))
        if directory is None:
            database = ":memory:"
        else:
            directory.mkdir(parents=True, exist_ok=True)
            database = str(directory / "coordination.sqlite3")
        self.engine = create_engine(
            URL.create("sqlite+pysqlite", database=database),
            connect_args={"timeout": 30, "check_same_thread": False},
            poolclass=NullPool,
            hide_parameters=True,
        )
        self.lock = RLock()
        self.open: Connection | None = self.engine.connect()
        try:
            if database != ":memory:":
                # Several processes of one host open the same file at once; the first switches it to WAL.
                enable_wal(self.open, database, timeout_seconds=30.0, retry_seconds=0.01)
                self.open.commit()
            with self.open.begin():
                # Serialize the table creation of concurrent openers.
                self.open.exec_driver_sql("BEGIN IMMEDIATE")
                self.create(self.open, metadata)
        except BaseException:
            self.close()
            raise

    @property
    def locks_rows(self) -> bool:
        return False

    @contextmanager
    def connection(self, *, write: bool = False) -> Iterator[Connection]:
        with self.lock:
            if self.open is None:
                raise ReefError("the coordination database is closed")
            with self.open.begin():
                if write:
                    # Another process can hold the file: take the write lock now, within the busy timeout, so a
                    # read never has to upgrade to a write, which SQLite refuses at once under contention.
                    self.open.exec_driver_sql("BEGIN IMMEDIATE")
                yield self.open

    def close(self) -> None:
        with self.lock:
            if self.open is not None:
                self.open.close()
                self.open = None
                self.engine.dispose()


class PostgresCoordinationDatabase(CoordinationDatabase):
    """The coordination tables in a PostgreSQL schema. All services of a deployment name the same schema."""

    def __init__(self, database_url: str, *, schema: str = "reef_coordination") -> None:
        validate_postgres_schema(schema)
        metadata = MetaData(schema=schema)
        super().__init__(coordination_tables(metadata))
        try:
            self.engine: Engine | None = create_engine(
                postgres_url(database_url), pool_pre_ping=True, hide_parameters=True
            )
        except ModuleNotFoundError:
            raise ReefError("the PostgreSQL coordination database requires the reef-infra[postgres] extra") from None
        key = int.from_bytes(
            hashlib.sha256(f"reef-coordination-schema:{schema}".encode()).digest()[:8], "big", signed=True
        )
        try:
            with self.engine.begin() as connection:
                # Serialize the check-then-create of the schema across services that start together.
                connection.execute(select(func.pg_advisory_xact_lock(key)))
                connection.execute(CreateSchema(schema, if_not_exists=True))
                self.create(connection, metadata)
        except BaseException:
            self.close()
            raise

    @property
    def locks_rows(self) -> bool:
        return True

    @contextmanager
    def connection(self, *, write: bool = False) -> Iterator[Connection]:
        if self.engine is None:
            raise ReefError("the coordination database is closed")
        with self.engine.begin() as connection:
            yield connection

    def close(self) -> None:
        if self.engine is not None:
            self.engine.dispose()
            self.engine = None


def served_from_row(row: RowMapping) -> ServedArtifact:
    return ServedArtifact(row["artifact_id"], row["url"], row["model_name"])


class SQLServedModels(ServedModelsWriter):
    """``served_models`` in a coordination database. Writes check the number of ``lease`` in their transaction."""

    def __init__(self, database: CoordinationDatabase, lease: SQLControlLease) -> None:
        self.database = database
        self.tables = database.tables
        self.lease = lease

    def checked(self, connection: Connection, control_lease_number: int) -> None:
        current = self.lease.number_in(connection)
        if current != control_lease_number:
            raise ControlLeaseConflict(
                f"control lease number {control_lease_number} is not the current number {current}"
            )

    def read(self, key: ScenarioModel) -> ServedArtifact | None:
        table = self.tables.served_model
        with self.database.read() as connection:
            row = (
                connection.execute(select(table).where(table.c.scenario == key.scenario, table.c.name == key.name))
                .mappings()
                .first()
            )
        return None if row is None else served_from_row(row)

    def read_scenario(self, scenario: str) -> Mapping[str, ServedArtifact]:
        table = self.tables.served_model
        with self.database.read() as connection:
            rows = connection.execute(select(table).where(table.c.scenario == scenario)).mappings().all()
        return {row["name"]: served_from_row(row) for row in rows}

    def changed_sequence(self) -> int:
        with self.database.read() as connection:
            return int(
                connection.execute(
                    select(self.tables.sequence.c.value).where(self.tables.sequence.c.id == SERVED_SEQUENCE)
                ).scalar_one()
            )

    def write(self, key: ScenarioModel, served: ServedArtifact, *, control_lease_number: int) -> None:
        table = self.tables.served_model
        with self.database.write(SERVED_SEQUENCE) as (connection, counter):
            self.checked(connection, control_lease_number)
            connection.execute(delete(table).where(table.c.scenario == key.scenario, table.c.name == key.name))
            connection.execute(insert(table).values(scenario=key.scenario, name=key.name, **served.to_json()))
            counter.next()

    def remove(self, key: ScenarioModel, *, control_lease_number: int) -> ServedArtifact | None:
        table = self.tables.served_model
        condition = and_(table.c.scenario == key.scenario, table.c.name == key.name)
        with self.database.write(SERVED_SEQUENCE) as (connection, counter):
            self.checked(connection, control_lease_number)
            row = connection.execute(select(table).where(condition)).mappings().first()
            if row is None:
                return None
            connection.execute(delete(table).where(condition))
            counter.next()
            return served_from_row(row)

    def schedule_stop(self, stop: ScheduledStop, *, control_lease_number: int) -> None:
        table = self.tables.scheduled_stop
        served = stop.served
        with self.database.write(SERVED_SEQUENCE) as (connection, counter):
            self.checked(connection, control_lease_number)
            connection.execute(delete(table).where(*self.stop_condition(served)))
            connection.execute(insert(table).values(stop_after=stop.stop_after, **served.to_json()))
            counter.next()

    def stop_condition(self, served: ServedArtifact) -> tuple[ColumnElement[bool], ...]:
        table = self.tables.scheduled_stop
        return (
            table.c.artifact_id == served.artifact_id,
            table.c.url == served.url,
            table.c.model_name == served.model_name,
        )

    def due_stops(self, now: float) -> tuple[ScheduledStop, ...]:
        table = self.tables.scheduled_stop
        with self.database.read() as connection:
            rows = (
                connection.execute(select(table).where(table.c.stop_after <= now).order_by(table.c.stop_after))
                .mappings()
                .all()
            )
        return tuple(ScheduledStop(served_from_row(row), float(row["stop_after"])) for row in rows)

    def delete_stop(self, served: ServedArtifact, *, control_lease_number: int) -> None:
        with self.database.write(SERVED_SEQUENCE) as (connection, counter):
            self.checked(connection, control_lease_number)
            if connection.execute(delete(self.tables.scheduled_stop).where(*self.stop_condition(served))).rowcount:
                counter.next()

    def all_served(self) -> frozenset[ServedArtifact]:
        with self.database.read() as connection:
            rows = connection.execute(select(self.tables.served_model)).mappings().all()
        return frozenset(served_from_row(row) for row in rows)


class SQLControlLease(ControlLease):
    """The control lease as the one row of ``control_lease``."""

    def __init__(self, database: CoordinationDatabase, *, lease_seconds: float = 30.0) -> None:
        super().__init__(lease_seconds=lease_seconds)
        self.database = database
        self.table = database.tables.control_lease

    def held(self, connection: Connection) -> ControlLeaseState | None:
        query = select(self.table).where(self.table.c.id == 1)
        if self.database.locks_rows:
            query = query.with_for_update()
        row = connection.execute(query).mappings().first()
        if row is None:
            return None
        return ControlLeaseState(row["instance_id"], int(row["number"]), float(row["expires_at"]))

    def store(self, connection: Connection, state: ControlLeaseState, *, existed: bool) -> ControlLeaseState:
        values = {"instance_id": state.instance_id, "number": state.number, "expires_at": state.expires_at}
        if existed:
            connection.execute(update(self.table).where(self.table.c.id == 1).values(**values))
        else:
            connection.execute(insert(self.table).values(id=1, **values))
        return state

    def acquire(self, instance_id: str, *, now: float | None = None) -> ControlLeaseState | None:
        require_text(instance_id, "instance_id")
        moment = clock(now)
        try:
            with self.database.connection(write=True) as connection:
                held = self.held(connection)
                if held is not None and held.instance_id == instance_id and held.expires_at >= moment:
                    number = held.number
                elif held is None or held.expires_at < moment:
                    number = 1 if held is None else held.number + 1
                else:
                    return None
                state = ControlLeaseState(instance_id, number, moment + self.lease_seconds)
                return self.store(connection, state, existed=held is not None)
        except IntegrityError:
            # Another instance took the first lease in the same moment: its row is there now, so this take lost.
            return None

    def extend(self, instance_id: str, *, now: float | None = None) -> ControlLeaseState | None:
        moment = clock(now)
        with self.database.connection(write=True) as connection:
            held = self.held(connection)
            if held is None or held.instance_id != instance_id or held.expires_at < moment:
                return None
            return self.store(
                connection, ControlLeaseState(instance_id, held.number, moment + self.lease_seconds), existed=True
            )

    def release(self, instance_id: str, *, now: float | None = None) -> bool:
        moment = clock(now)
        with self.database.connection(write=True) as connection:
            held = self.held(connection)
            if held is None or held.instance_id != instance_id or held.expires_at < moment:
                return False
            self.store(connection, ControlLeaseState(instance_id, held.number, moment), existed=True)
            return True

    def number_in(self, connection: Connection) -> int:
        """The number of the lease that holds now, read in ``connection``'s transaction; 0 before the first take."""
        number = connection.execute(select(self.table.c.number).where(self.table.c.id == 1)).scalar()
        return 0 if number is None else int(number)

    def current(self) -> ControlLeaseState | None:
        with self.database.read() as connection:
            row = connection.execute(select(self.table).where(self.table.c.id == 1)).mappings().first()
        if row is None:
            return None
        return ControlLeaseState(row["instance_id"], int(row["number"]), float(row["expires_at"]))


class SQLMarks(Marks):
    """The marks as rows of ``coordination_mark``. A mark changes no work, so it takes no sequence value."""

    def __init__(self, database: CoordinationDatabase) -> None:
        self.database = database
        self.table = database.tables.mark

    def read(self, name: str) -> int:
        with self.database.read() as connection:
            value = connection.execute(
                select(self.table.c.value).where(self.table.c.name == check_mark_name(name))
            ).scalar_one_or_none()
        return 0 if value is None else int(value)

    def advance(self, name: str, value: int) -> None:
        check_mark_name(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("a mark is a non-negative int")
        with self.database.connection(write=True) as connection:
            query = select(self.table.c.value).where(self.table.c.name == name)
            if self.database.locks_rows:
                query = query.with_for_update()
            current = connection.execute(query).scalar_one_or_none()
            if current is None:
                connection.execute(insert(self.table).values(name=name, value=value))
            elif value > int(current):
                connection.execute(update(self.table).where(self.table.c.name == name).values(value=value))


def item_from_row(row: RowMapping) -> WorkItem:
    return WorkItem(
        queue=row["queue"],
        id=row["item_id"],
        scenario=row["scenario"],
        kind=row["kind"],
        payload=json.loads(row["payload_json"]),
        state=row["state"],
        serial_key=row["serial_key"],
        available_at=float(row["available_at"]),
        created_at=float(row["created_at"]),
        attempt_count=int(row["attempt_count"]),
        worker_id=row["worker_id"],
        lease_expires_at=None if row["lease_expires_at"] is None else float(row["lease_expires_at"]),
        progress=tuple(json.loads(row["progress_json"])),
        result=None if row["result_json"] is None else json.loads(row["result_json"]),
        error=row["error"],
        created_sequence=int(row["created_sequence"]),
        changed_sequence=int(row["changed_sequence"]),
    )


class SQLWorkQueue(WorkQueue):
    """Work queues in a coordination database.

    Every change holds the ``WORK_SEQUENCE`` counter row, so the changes of all queues are serialized: two workers
    that take at once never get the same item. ``take`` selects the oldest item that the worker can take.
    """

    def __init__(
        self,
        database: CoordinationDatabase,
        *,
        lease_seconds: float = 60.0,
        max_attempts: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__(lease_seconds=lease_seconds, max_attempts=max_attempts)
        self.database = database
        self.table = database.tables.work_item

    def key(self, queue: str, item_id: str) -> tuple[ColumnElement[bool], ...]:
        return (self.table.c.queue == queue, self.table.c.item_id == item_id)

    def row(self, connection: Connection, queue: str, item_id: str, *, lock: bool = False) -> WorkItem | None:
        query = select(self.table).where(*self.key(queue, item_id))
        if lock and self.database.locks_rows:
            query = query.with_for_update()
        found = connection.execute(query).mappings().first()
        return None if found is None else item_from_row(found)

    def save(self, connection: Connection, counter: SequenceCounter, item: WorkItem, **changes: object) -> WorkItem:
        values: dict[str, object] = {"changed_sequence": counter.next()}
        for name, value in changes.items():
            if name == "progress":
                values["progress_json"] = canonical_json(list(cast(Sequence[object], value)))
            elif name == "result":
                values["result_json"] = None if value is None else canonical_json(value)
            else:
                values[name] = value
        connection.execute(update(self.table).where(*self.key(item.queue, item.id)).values(**values))
        saved = self.row(connection, item.queue, item.id)
        if saved is None:
            raise ReefError(f"work item {item.queue}/{item.id} disappeared")
        return saved

    def held(self, connection: Connection, queue: str, item_id: str, worker_id: str, now: float) -> WorkItem:
        item = self.row(connection, queue, item_id, lock=True)
        if (
            item is None
            or item.state not in HELD_STATES
            or item.worker_id != worker_id
            or (item.lease_expires_at is not None and item.lease_expires_at < now)
        ):
            raise WorkLeaseLost(f"worker {worker_id!r} does not hold {queue}/{item_id}")
        return item

    def add(self, work: NewWork, *, now: float | None = None) -> WorkItem:
        moment = clock(now)
        payload = canonical_json(dict(work.payload))
        with self.database.write(WORK_SEQUENCE) as (connection, counter):
            existing = self.row(connection, work.queue, work.id)
            if existing is not None:
                if canonical_json(dict(existing.payload)) != payload:
                    raise WorkConflict(f"{work.queue}/{work.id} exists with a different payload")
                return existing
            sequence = counter.next()
            connection.execute(
                insert(self.table).values(
                    queue=require_text(work.queue, "queue"),
                    item_id=require_text(work.id, "id"),
                    scenario=require_text(work.scenario, "scenario"),
                    kind=require_text(work.kind, "kind"),
                    payload_json=payload,
                    state="queued",
                    serial_key=work.serial_key,
                    available_at=moment if work.available_at is None else work.available_at,
                    created_at=moment,
                    attempt_count=0,
                    progress_json="[]",
                    created_sequence=sequence,
                    changed_sequence=sequence,
                )
            )
            added = self.row(connection, work.queue, work.id)
            if added is None:
                raise ReefError(f"work item {work.queue}/{work.id} was not added")
            return added

    def take(self, queue: str, worker_id: str, kinds: Sequence[str], *, now: float | None = None) -> WorkItem | None:
        require_text(worker_id, "worker_id")
        if not kinds:
            return None
        moment = clock(now)
        item = self.table
        earlier = self.table.alias("earlier")
        blocked = exists(
            select(earlier.c.item_id).where(
                earlier.c.queue == item.c.queue,
                earlier.c.serial_key == item.c.serial_key,
                earlier.c.created_sequence < item.c.created_sequence,
                earlier.c.state.not_in(sorted(FINAL_STATES)),
            )
        )
        query = (
            select(item)
            .where(
                item.c.queue == queue,
                item.c.state == "queued",
                item.c.kind.in_(list(kinds)),
                item.c.available_at <= moment,
                not_(and_(item.c.serial_key.is_not(None), blocked)),
            )
            .order_by(item.c.created_sequence)
            .limit(1)
        )
        if self.database.locks_rows:
            query = query.with_for_update(skip_locked=True, of=item)
        with self.database.write(WORK_SEQUENCE) as (connection, counter):
            found = connection.execute(query).mappings().first()
            if found is None:
                return None
            chosen = item_from_row(found)
            return self.save(
                connection,
                counter,
                chosen,
                state="assigned",
                worker_id=worker_id,
                lease_expires_at=moment + self.lease_seconds,
                attempt_count=chosen.attempt_count + 1,
                error=None,
            )

    def extend_lease(self, queue: str, item_id: str, worker_id: str, *, now: float | None = None) -> WorkItem:
        moment = clock(now)
        with self.database.write(WORK_SEQUENCE) as (connection, counter):
            item = self.held(connection, queue, item_id, worker_id, moment)
            return self.save(connection, counter, item, lease_expires_at=moment + self.lease_seconds)

    def start(self, queue: str, item_id: str, worker_id: str, *, now: float | None = None) -> WorkItem:
        with self.database.write(WORK_SEQUENCE) as (connection, counter):
            item = self.held(connection, queue, item_id, worker_id, clock(now))
            return item if item.state == "running" else self.save(connection, counter, item, state="running")

    def add_progress(
        self,
        queue: str,
        item_id: str,
        worker_id: str,
        progress: Mapping[str, JsonValue],
        *,
        number: int,
        now: float | None = None,
    ) -> WorkItem:
        with self.database.write(WORK_SEQUENCE) as (connection, counter):
            item = self.held(connection, queue, item_id, worker_id, clock(now))
            if number < len(item.progress):
                return item
            if number != len(item.progress):
                raise ValueError(f"progress number {number} skips {len(item.progress)}")
            return self.save(connection, counter, item, progress=(*item.progress, dict(progress)))

    def finish(
        self, queue: str, item_id: str, worker_id: str, result: Mapping[str, JsonValue], *, now: float | None = None
    ) -> WorkItem:
        with self.database.write(WORK_SEQUENCE) as (connection, counter):
            item = self.held(connection, queue, item_id, worker_id, clock(now))
            return self.save(
                connection, counter, item, state="finished", result=dict(result), worker_id=None, lease_expires_at=None
            )

    def fail(
        self,
        queue: str,
        item_id: str,
        worker_id: str,
        error: str,
        *,
        retryable: bool,
        now: float | None = None,
    ) -> WorkItem:
        require_text(error, "error")
        with self.database.write(WORK_SEQUENCE) as (connection, counter):
            item = self.held(connection, queue, item_id, worker_id, clock(now))
            again = retryable and item.attempt_count < self.attempts_allowed(queue)
            return self.save(
                connection,
                counter,
                item,
                state="queued" if again else "failed",
                error=error,
                worker_id=None,
                lease_expires_at=None,
            )

    def cancel(self, queue: str, item_id: str, error: str) -> WorkItem | None:
        require_text(error, "error")
        with self.database.write(WORK_SEQUENCE) as (connection, counter):
            item = self.row(connection, queue, item_id, lock=True)
            if item is None or item.final:
                return item
            return self.save(
                connection, counter, item, state="failed", error=error, worker_id=None, lease_expires_at=None
            )

    def requeue_expired(self, *, now: float | None = None) -> tuple[WorkItem, ...]:
        moment = clock(now)
        query = select(self.table).where(
            self.table.c.state.in_(sorted(HELD_STATES)), self.table.c.lease_expires_at < moment
        )
        if self.database.locks_rows:
            query = query.with_for_update(skip_locked=True)
        with self.database.write(WORK_SEQUENCE) as (connection, counter):
            expired = [item_from_row(row) for row in connection.execute(query).mappings().all()]
            return tuple(
                self.save(
                    connection,
                    counter,
                    item,
                    state="queued" if item.attempt_count < self.attempts_allowed(item.queue) else "failed",
                    error="the worker's lease expired",
                    worker_id=None,
                    lease_expires_at=None,
                )
                for item in expired
            )

    def get(self, queue: str, item_id: str) -> WorkItem | None:
        with self.database.read() as connection:
            return self.row(connection, queue, item_id)

    def read_changes(self, queue: str, after_sequence: int, *, limit: int = 1000) -> tuple[WorkItem, ...]:
        query = (
            select(self.table)
            .where(self.table.c.queue == queue, self.table.c.changed_sequence > after_sequence)
            .order_by(self.table.c.changed_sequence)
            .limit(limit)
        )
        with self.database.read() as connection:
            return tuple(item_from_row(row) for row in connection.execute(query).mappings().all())

    def items(self, queue: str, *, scenario: str | None = None, kind: str | None = None) -> tuple[WorkItem, ...]:
        conditions = [self.table.c.queue == queue]
        if scenario is not None:
            conditions.append(self.table.c.scenario == scenario)
        if kind is not None:
            conditions.append(self.table.c.kind == kind)
        query = select(self.table).where(*conditions).order_by(self.table.c.created_sequence)
        with self.database.read() as connection:
            return tuple(item_from_row(row) for row in connection.execute(query).mappings().all())


def open_coordination_database(
    backend: str, *, directory: Path | None, database_url: str | None = None, schema: str | None = None
) -> CoordinationDatabase:
    """The coordination database of a deployment: ``sqlite`` in ``directory`` (in memory when it is None), or
    ``postgres`` at ``database_url`` in ``schema``."""
    if backend == "sqlite":
        return SQLiteCoordinationDatabase(directory)
    if backend == "postgres":
        if not database_url:
            raise ValueError("the postgres coordination database needs a database URL")
        return PostgresCoordinationDatabase(database_url, schema=schema or "reef_coordination")
    raise ValueError(f"coordination backend must be sqlite or postgres, not {backend!r}")


def open_coordination(
    backend: str,
    *,
    agent_record_dir: str,
    record_database_url: str | None,
    record_database_schema: str = "reef_records",
    lease_seconds: float = 60.0,
) -> Coordination:
    """The coordination store of a deployment on ``backend``: ``postgres`` uses ``record_database_url`` with the
    schema ``<record_database_schema>_coordination``, so that deployments with their own record schemas keep their
    own coordination tables; ``sqlite`` uses a SQLite file in ``<agent_record_dir>/coordination``. Raise
    ``ValueError`` for another backend."""
    if backend not in ("sqlite", "postgres"):
        raise ValueError(f"the coordination backend is sqlite or postgres, not {backend!r}")
    if backend == "postgres":
        database = open_coordination_database(
            "postgres",
            directory=None,
            database_url=record_database_url,
            schema=f"{record_database_schema}_coordination",
        )
    else:
        database = open_coordination_database("sqlite", directory=Path(agent_record_dir) / "coordination")
    return coordination_on(database, lease_seconds=lease_seconds)


def coordination_on(database: CoordinationDatabase, *, lease_seconds: float = 60.0) -> Coordination:
    """The coordination store whose tables ``database`` holds; ``lease_seconds`` is the lease of a work item."""
    lease = SQLControlLease(database)
    return Coordination(
        database,
        SQLWorkQueue(database, lease_seconds=lease_seconds),
        SQLServedModels(database, lease),
        lease,
        SQLMarks(database),
    )


__all__ = [
    "CoordinationDatabase",
    "CoordinationTables",
    "PostgresCoordinationDatabase",
    "SQLControlLease",
    "SQLMarks",
    "SQLServedModels",
    "SQLWorkQueue",
    "SQLiteCoordinationDatabase",
    "coordination_on",
    "coordination_tables",
    "open_coordination",
    "open_coordination_database",
]
