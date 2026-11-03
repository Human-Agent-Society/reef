"""Training samples for tests (RFC #1140): scenario storage whose sessions read their samples from a data store,
and task samples with a directory."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from reef.core.job_data import TaskSample
from reef.service.task_import import DataScenarioStorage
from reef.storage.sql_data import SQLiteDataStore
from reef.storage.sqlite import SQLiteScenarioStorage

#: The directories of ``made_task``; they last as long as the test process.
TASKS = tempfile.TemporaryDirectory(prefix="reef-test-tasks-")


def data_storage(directory: Path | None = None) -> DataScenarioStorage:
    """SQLite scenario storage in ``directory`` (in memory without one) whose sessions read their training samples
    from a SQLite data store beside it, with the same task files."""
    storage = SQLiteScenarioStorage(directory)
    return DataScenarioStorage(
        storage, SQLiteDataStore(None if directory is None else directory / "data", storage.task_files)
    )


def made_task(name: str, **fields: Any) -> TaskSample:
    """A task sample whose directory, named ``name``, holds one instruction; ``fields`` are the sample's other
    fields, such as ``metadata``."""
    path = Path(TASKS.name) / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "instruction.md").write_text(f"Solve {name}.\n")
    return TaskSample(path, **fields)


__all__ = ["data_storage", "made_task"]
