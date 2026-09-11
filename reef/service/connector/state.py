"""Private connector configuration and a durable record of dispatched commands."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class ConnectorState:
    """Keep credentials on this machine and never re-execute a recorded command."""

    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(directory, 0o700)
        self.config_path = directory / "connection.json"
        self.db = sqlite3.connect(directory / "commands.sqlite3")
        os.chmod(directory / "commands.sqlite3", 0o600)
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS commands (id TEXT PRIMARY KEY, result TEXT, acknowledged INTEGER DEFAULT 0)"
        )
        self.db.commit()

    def load(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return {}
        return json.loads(self.config_path.read_text())

    def save(self, config: dict[str, Any]) -> None:
        temporary = self.directory / "connection.tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(config, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.config_path)

    @contextmanager
    def lock(self):
        """OS locks release on crashes; a PID file alone cannot prove ownership."""
        stream = (self.directory / "connector.lock").open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                stream.write(b"0")
                stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            stream.close()
            raise RuntimeError("A connector is already running for this instance") from exc
        try:
            yield
        finally:
            stream.close()

    def recover(self) -> None:
        self.db.execute(
            "UPDATE commands SET result=? WHERE result IS NULL",
            (
                json.dumps(
                    {"state": "unknown", "error": "Connector restarted after dispatch. Check Reef before retrying."}
                ),
            ),
        )
        self.db.commit()

    def start(self, command_id: str) -> bool:
        cursor = self.db.execute("INSERT OR IGNORE INTO commands (id) VALUES (?)", (command_id,))
        self.db.commit()
        return cursor.rowcount == 1

    def finish(self, command_id: str, result: dict[str, Any]) -> None:
        self.db.execute("UPDATE commands SET result=?, acknowledged=0 WHERE id=?", (json.dumps(result), command_id))
        self.db.commit()

    def pending(self) -> list[tuple[str, dict[str, Any]]]:
        return [
            (row[0], json.loads(row[1]))
            for row in self.db.execute(
                "SELECT id,result FROM commands WHERE acknowledged=0 AND result IS NOT NULL LIMIT 20"
            )
        ]

    def acknowledge(self, command_id: str) -> None:
        self.db.execute("UPDATE commands SET acknowledged=1 WHERE id=?", (command_id,))
        self.db.commit()

    def close(self) -> None:
        self.db.close()
