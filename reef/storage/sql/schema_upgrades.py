"""The version of a store's SQL tables, and the steps that upgrade them.

A store that changes the layout of its tables adds a ``SchemaStep``. ``upgrade_schema`` runs when the store opens its
database: it makes the tables of a new database, and it brings the tables of an earlier version to the current one,
one step after the other. So an upgrade of Reef needs no hand-written SQL, and a database of a later Reef is refused
with a reason instead of being read wrongly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from sqlalchemy import Column, Integer, MetaData, Table, delete, insert, inspect, select
from sqlalchemy.engine import Connection

from reef.core.errors import ReefError


class SchemaStep(ABC):
    """One upgrade of a store's tables, from one version to the next.

    Operation: ``apply(connection, schema)`` changes the tables of the version before the step into the tables of
    the version after it, in the caller's transaction. ``schema`` is the SQL schema of the tables, or None for a
    database without schemas. The tables of the current layout exist when it runs: the step moves rows into them
    and drops what only the earlier layout had. A step reads no table that a later step makes or drops, so the
    steps of several versions run in order in one transaction.

    Purpose: a store lists its steps in the order of their versions and gives them to ``upgrade_schema``. A step
    never changes after a release: a database at its version exists.
    """

    @abstractmethod
    def apply(self, connection: Connection, schema: str | None) -> None: ...


def upgrade_schema(
    connection: Connection, metadata: MetaData, steps: Sequence[SchemaStep], *, version_table: str, first_table: str
) -> int:
    """Make the tables of ``metadata`` the current version, and return it: 1 plus the number of ``steps``.

    The caller's transaction holds a lock that serializes the stores that open the database together.

    * A database without ``first_table``, a table that every version has, is new. It gets the tables, and no step
      runs.
    * A database with ``first_table`` and without ``version_table`` is at version 1, the layout before the store
      recorded its version.
    * A database at version ``v`` gets the missing tables of the current layout, then the steps ``v``, ``v + 1``
      and so on (step ``n`` upgrades version ``n``).
    * ``ReefError`` for a version above the current one: a later Reef wrote the tables.

    ``version_table`` holds the one row of the version afterwards.
    """
    current = 1 + len(steps)
    schema = metadata.schema
    versions = Table(version_table, MetaData(schema=schema), Column("version", Integer, primary_key=True))
    inspector = inspect(connection)
    if not inspector.has_table(first_table, schema=schema):
        stored = current
    elif not inspector.has_table(version_table, schema=schema):
        stored = 1
    else:
        stored = int(connection.execute(select(versions.c.version)).scalar_one())
    if stored > current:
        raise ReefError(
            f"the tables of {first_table!r} are at version {stored}, and this Reef reads version {current}: "
            "a later Reef wrote them"
        )
    metadata.create_all(connection)
    versions.create(connection, checkfirst=True)
    for step in steps[stored - 1 :]:
        step.apply(connection, schema)
    connection.execute(delete(versions))
    connection.execute(insert(versions).values(version=current))
    return current


__all__ = ["SchemaStep", "upgrade_schema"]
