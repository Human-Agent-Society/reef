"""Connection setup shared by the SQL stores: SQLite's WAL mode and PostgreSQL URLs and schema names."""

from __future__ import annotations

import re
import time

from sqlalchemy.engine import URL, Connection, make_url
from sqlalchemy.exc import ArgumentError, OperationalError

from reef.core.errors import ReefError


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


def postgres_url(database_url: str) -> URL:
    """Validate a PostgreSQL URL and select psycopg 3 without exposing credentials."""
    try:
        url = make_url(database_url)
        if url.drivername not in {"postgres", "postgresql", "postgresql+psycopg"}:
            raise ValueError
        # Force validation here; malformed ports otherwise fail during engine setup.
        if url.port is not None and not 0 < url.port < 65536:
            raise ValueError
    except (TypeError, ValueError, AttributeError, ArgumentError):
        raise ValueError("record_database_url must be a PostgreSQL URL using psycopg 3") from None
    return url.set(drivername="postgresql+psycopg")


def validate_postgres_schema(schema: str) -> None:
    """Require a dedicated, portable SQL schema name for deployment isolation."""
    if not isinstance(schema, str) or re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", schema) is None:
        raise ValueError("record_database_schema must be a lowercase SQL identifier of at most 63 characters")
    if schema == "public" or schema == "information_schema" or schema.startswith("pg_"):
        raise ValueError("record_database_schema must be a dedicated Reef schema")
