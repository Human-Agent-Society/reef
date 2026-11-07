"""Training samples for tests (RFC #1140): scenario storage beside a data store, the sample store of an open
scenario, and task samples with a directory."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from reef.core.job_data import TaskSample
from reef.service.task_import import DataScenarioStorage
from reef.storage.samples import SampleStore
from reef.storage.scenario import ScenarioStore
from reef.storage.sqlite import SQLiteScenarioStorage

#: The directories of ``made_task``; they last as long as the test process.
TASKS = tempfile.TemporaryDirectory(prefix="reef-test-tasks-")


def data_storage(directory: Path | None = None) -> DataScenarioStorage:
    """SQLite scenario storage in ``directory`` (in memory without one) with its SQLite data store, which keeps its
    records and its samples."""
    storage = SQLiteScenarioStorage(directory)
    return DataScenarioStorage(storage, storage.data)


def sample_store(storage: DataScenarioStorage, scenario: str, session: ScenarioStore) -> SampleStore:
    """The sample store of ``scenario``, whose session ``storage`` opened."""
    items = storage.sample_store(scenario, session)
    assert items is not None
    return items


def open_sample_store(directory: Path | None, scenario: str) -> tuple[ScenarioStore, SampleStore]:
    """Open ``scenario`` in ``data_storage(directory)``; return its session and its sample store."""
    storage = data_storage(directory)
    session = storage.open(scenario)
    return session, sample_store(storage, scenario, session)


def made_task(name: str, **fields: Any) -> TaskSample:
    """A task sample whose directory, named ``name``, holds one instruction; ``fields`` are the sample's other
    fields, such as ``metadata``."""
    path = Path(TASKS.name) / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "instruction.md").write_text(f"Solve {name}.\n")
    return TaskSample(path, **fields)


__all__ = ["data_storage", "made_task", "open_sample_store", "sample_store"]
