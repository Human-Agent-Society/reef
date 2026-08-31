"""The results store: one append-only SQLite table, viewed as a DataFrame.

Design rules (see docs/design.md):

- Dimensions are tags, not schema. Rows carry a free-form JSON tag dict and
  ``df()`` expands tags into columns. Metrics never require a fixed schema:
  each metric function documents the tag names it expects.
- Raw scores only. Normalization happens at query time, never at write time.
- Re-running resumes. Each row's key is a stable ID for the call that
  produced it; ``get()`` lets callers skip work that already has a row.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import pandas as pd

from reef_eval.types import Row

_SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    key        TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,
    task       TEXT NOT NULL,
    tags       TEXT NOT NULL,
    rewards    TEXT NOT NULL,
    uri        TEXT,
    created_at REAL NOT NULL
);
-- reef-eval's own reads go through the primary key or load the whole table into
-- pandas, so these indexes are for querying results.sqlite directly, which
-- the data model treats as a supported way to audit a run.
CREATE INDEX IF NOT EXISTS idx_results_kind ON results (kind);
CREATE INDEX IF NOT EXISTS idx_results_task ON results (task);
"""


class Store:
    """Append-only SQLite store, safe for use from one asyncio process."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def put(self, row: Row) -> None:
        """Insert a row. Raises on a duplicate key: callers are expected to
        check ``get()`` first, and a violated assumption should be loud."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO results (key, kind, task, tags, rewards, uri, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    row.key,
                    row.kind,
                    row.task,
                    json.dumps(row.tags, sort_keys=True),
                    json.dumps(row.rewards, sort_keys=True),
                    row.uri,
                    row.created_at,
                ),
            )
            self._conn.commit()

    def get(self, key: str) -> Row | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT key, kind, task, tags, rewards, uri, created_at"
                " FROM results WHERE key = ?",
                (key,),
            )
            hit = cur.fetchone()
        return self._to_row(hit) if hit else None

    def delete_prefix(self, key_prefix: str) -> int:
        """Remove rows whose key starts with *key_prefix*.

        The one sanctioned mutation: clearing a partially-written episode
        (its trace rows) before a retry.
        """
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM results WHERE key GLOB ?", (key_prefix + "*",)
            )
            self._conn.commit()
            return cur.rowcount

    def df(self, kind: str | None = None) -> pd.DataFrame:
        """All rows as a DataFrame with tags and rewards expanded to columns.

        Reward keys become columns verbatim (a plain Harbor task yields a
        ``reward`` column); tag keys likewise. Collisions are resolved with
        prefixes so every column stays 1-dimensional:

        - a tag named like a base column (``key``, ``kind``, ``task``,
          ``uri``, ``created_at``) becomes ``tag_<name>``;
        - a reward named like a base or tag column becomes ``reward_<name>``.
        """
        with self._lock:
            raw = pd.read_sql_query("SELECT * FROM results", self._conn)
        if kind is not None:
            raw = raw[raw["kind"] == kind].reset_index(drop=True)
        if raw.empty:
            return raw

        base = raw.drop(columns=["tags", "rewards"]).reset_index(drop=True)
        tags = pd.json_normalize(raw["tags"].map(json.loads))
        tags.columns = [f"tag_{c}" if c in base.columns else c for c in tags.columns]
        rewards = pd.json_normalize(raw["rewards"].map(json.loads))
        taken = set(base.columns) | set(tags.columns)
        rewards.columns = [f"reward_{c}" if c in taken else c for c in rewards.columns]
        return pd.concat([base, tags, rewards], axis=1)

    @staticmethod
    def _to_row(hit: tuple) -> Row:
        key, kind, task, tags, rewards, uri, created_at = hit
        return Row(
            key=key,
            kind=kind,
            task=task,
            tags=json.loads(tags),
            rewards=json.loads(rewards),
            uri=uri,
            created_at=created_at,
        )
