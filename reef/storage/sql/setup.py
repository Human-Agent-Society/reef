"""Connection setup shared by the SQL stores: SQLite's WAL mode, PostgreSQL URLs and schema names, and the range of
the strings that start with a prefix."""

from __future__ import annotations

import re
import time

from sqlalchemy import ColumnElement, and_
from sqlalchemy.engine import URL, Connection, make_url
from sqlalchemy.exc import ArgumentError, OperationalError

from reef.core.errors import ReefError


def prefix_upper_bound(prefix: str) -> str | None:
    """The least string that is greater than every string that starts with ``prefix``, in code point order; None when
    no string is (``prefix`` is empty or has only U+10FFFF, the largest code point).

    Code point order is the byte order of UTF-8: the order of SQLite's BINARY collation and of PostgreSQL's "C"
    collation.
    """
    kept = prefix.rstrip(chr(0x10FFFF))
    if not kept:
        return None
    following = ord(kept[-1]) + 1
    # The surrogates U+D800 to U+DFFF are not characters: the code point after U+D7FF is U+E000.
    return kept[:-1] + chr(0xE000 if following == 0xD800 else following)


def prefix_range(column: ColumnElement[str], prefix: str) -> ColumnElement[bool]:
    """The condition that ``column``, a string in code point order (SQLite's BINARY collation, PostgreSQL's "C"
    collation), starts with ``prefix``: it is from ``prefix`` to ``prefix_upper_bound(prefix)``. An index of the
    column in that order finds the rows of the range."""
    upper = prefix_upper_bound(prefix)
    if upper is None:
        return column >= prefix
    return and_(column >= prefix, column < upper)


def enable_wal(connection: Connection, database: str, *, timeout_seconds: float, retry_seconds: float) -> None:
    """Switch the SQLite ``database`` of ``connection`` to WAL, waiting out a concurrent opener.

    SQLite takes an exclusive lock to change the journal mode and answers
    SQLITE_BUSY immediately instead of running the busy handler, so the
    connection timeout does not cover this statement. Openers that race to
    upgrade a database still in rollback mode therefore have to retry: the
    first one converts it and the rest read back ``wal`` on a later attempt.
    A database already in WAL answers on the first try, so the steady state
    costs nothing. Raise ``ReefError`` when the switch does not happen within ``timeout_seconds``.
    """
    deadline = time.monotonic() + timeout_seconds
    observed = None
    while True:
        # A contended switch shows up either way: as SQLITE_BUSY, or as the
        # unchanged mode read back, which SQLite returns instead of raising.
        try:
            row = connection.exec_driver_sql("PRAGMA journal_mode = WAL").fetchone()
        except OperationalError as exc:
            if "locked" not in str(exc.orig) and "busy" not in str(exc.orig):
                raise
        else:
            observed = None if row is None else str(row[0]).lower()
            if observed == "wal":
                return
        if time.monotonic() >= deadline:
            raise ReefError(
                f"could not switch {database} to WAL within {timeout_seconds:g}s; "
                f"another connection held the database (journal mode is {observed or 'unknown'})"
            )
        time.sleep(retry_seconds)


#: The libpq parameters that end a connection to a server that stopped answering, as when the server restarts. Without
#: them, a call waits for its answer forever and holds what its thread holds. A connect waits up to 10 s. A connection
#: that waits for an answer sends a keepalive after 30 s idle, and fails after 3 probes 10 s apart. Data that the server
#: does not acknowledge in 60 s fails the connection. A URL that sets a parameter keeps its value.
DEAD_SERVER_PARAMETERS = {
    "connect_timeout": "10",
    "keepalives": "1",
    "keepalives_idle": "30",
    "keepalives_interval": "10",
    "keepalives_count": "3",
    "tcp_user_timeout": "60000",
}


def postgres_url(database_url: str) -> URL:
    """Validate a PostgreSQL URL and select psycopg 3 without exposing credentials. The URL gets each
    ``DEAD_SERVER_PARAMETERS`` value that it does not set."""
    try:
        url = make_url(database_url)
        if url.drivername not in {"postgres", "postgresql", "postgresql+psycopg"}:
            raise ValueError
        # Force validation here; malformed ports otherwise fail during engine setup.
        if url.port is not None and not 0 < url.port < 65536:
            raise ValueError
    except (TypeError, ValueError, AttributeError, ArgumentError):
        raise ValueError("record_database_url must be a PostgreSQL URL using psycopg 3") from None
    defaults = {name: value for name, value in DEAD_SERVER_PARAMETERS.items() if name not in url.query}
    return url.set(drivername="postgresql+psycopg").update_query_dict(defaults)


def validate_postgres_schema(schema: str) -> None:
    """Require a dedicated, portable SQL schema name for deployment isolation."""
    if not isinstance(schema, str) or re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", schema) is None:
        raise ValueError("record_database_schema must be a lowercase SQL identifier of at most 63 characters")
    if schema == "public" or schema == "information_schema" or schema.startswith("pg_"):
        raise ValueError("record_database_schema must be a dedicated Reef schema")
