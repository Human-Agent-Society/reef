"""Task store contract on every bundled scenario storage."""

from __future__ import annotations

import tempfile
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import pytest

from reef.core.config import config_option
from reef.core.plugins import PluginSettings
from reef.core.tasks import HarborTask, read_harbor_task, write_harbor_task
from reef.service.assembly import build_dispatcher
from reef.service.deploy.service_config import ServiceConfig
from reef.storage.postgres import PostgresScenarioStorage
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.storage.tasks import (
    GenerationState,
    LocalTaskFileStore,
    StoredTask,
    TaskCandidate,
    TaskConflict,
    TaskStoreError,
    files_digest,
)


def generated(name: str, *, instruction: str = "", sources: tuple[str, ...] = ("record-1",)) -> HarborTask:
    return HarborTask(
        name=name,
        instruction=instruction or f"Solve {name}.",
        tests={"test.sh": "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n"},
        environment={"Dockerfile": "FROM python:3.12-slim\n"},
        config={
            "verifier": {"timeout_sec": 30},
            "environment": {"network_mode": "allowlist", "allowed_hosts": ["pypi.org"]},
        },
        metadata={"designer_record_id": "designer-1"},
        solution={"hint.txt": "look closer"},
        source_agent_record_ids=sources,
    )


def candidates(root: Path, *tasks: HarborTask) -> tuple[Path, ...]:
    """Write each task in a directory of its own, as the record-to-task stage does."""
    staging = Path(tempfile.mkdtemp(dir=root))
    return tuple(write_harbor_task(task, staging / str(index)) for index, task in enumerate(tasks))


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def storage(request, tmp_path):
    if request.param == "postgres":
        url, schema = request.getfixturevalue("postgres_config")
        opened = PostgresScenarioStorage(url, tmp_path, schema=schema)
    else:
        opened = SQLiteScenarioStorage(None if request.param == "memory" else tmp_path)
    with closing(opened):
        yield opened


@pytest.fixture
def session(storage):
    with closing(storage.open("math")) as session:
        yield session


def test_a_generation_records_candidates_and_outcomes(session, tmp_path):
    store = session.tasks.open("tasks")
    tasks = (generated("alpha"), generated("beta", sources=("record-2",)), generated("gamma"))
    directories = candidates(tmp_path, *tasks)
    digests = [files_digest(directory) for directory in directories]

    assert store.begin_generation(generation=0, source_record_ids=("record-1", "record-2")) == GenerationState(
        generation=0, source_record_ids=("record-1", "record-2")
    )
    assert store.save_candidates(generation=0, directories=directories).candidates == tuple(
        TaskCandidate(task.name, digest) for task, digest in zip(tasks, digests, strict=True)
    )
    alpha = store.publish_task(generation=0, index=0, split="train")
    store.publish_task(generation=0, index=1, split="eval")
    store.reject_task(generation=0, index=2, reason="the oracle failed")
    completed = store.complete_generation(generation=0)

    assert alpha == StoredTask(name="alpha", split="train", files_digest=digests[0])
    directory = store.path(alpha.files_digest)
    assert directory.name == "alpha"
    assert read_harbor_task(directory) == tasks[0]
    assert completed == GenerationState(
        generation=0,
        source_record_ids=("record-1", "record-2"),
        candidates=(
            TaskCandidate("alpha", digests[0], split="train"),
            TaskCandidate("beta", digests[1], split="eval"),
            TaskCandidate("gamma", digests[2], error="the oracle failed"),
        ),
        is_complete=True,
    )
    assert read_harbor_task(store.path(completed.candidates[2].files_digest)) == tasks[2]
    assert store.generation(0) == completed
    assert store.generation(1) is None
    assert store.generations() == (completed,)
    assert [task.name for task in store.tasks()] == ["alpha", "beta"]
    assert store.tasks(split="eval") == (StoredTask("beta", "eval", digests[1]),)


def test_identical_retries_change_nothing(session, tmp_path):
    store = session.tasks.open("tasks")
    tasks = (generated("alpha"), generated("beta"))
    for _ in range(2):
        store.begin_generation(generation=3, source_record_ids=("record-1",))
        store.save_candidates(generation=3, directories=candidates(tmp_path, *tasks))
        store.publish_task(generation=3, index=0, split="train")
        store.reject_task(generation=3, index=1, reason="invalid")
        store.complete_generation(generation=3)

    state = store.generation(3)
    assert state is not None and state.is_complete
    assert [(candidate.name, candidate.split, candidate.error) for candidate in state.candidates or ()] == [
        ("alpha", "train", ""),
        ("beta", None, "invalid"),
    ]
    assert len(store.tasks()) == 1


def test_conflicting_generation_writes_are_refused(session, tmp_path):
    store = session.tasks.open("tasks")
    with pytest.raises(TaskStoreError, match="does not exist"):
        store.save_candidates(generation=0, directories=candidates(tmp_path, generated("alpha")))
    store.begin_generation(generation=0, source_record_ids=("record-1",))
    with pytest.raises(TaskStoreError, match="other source records"):
        store.begin_generation(generation=0, source_record_ids=("record-2",))
    store.save_candidates(generation=0, directories=candidates(tmp_path, generated("alpha"), generated("beta")))
    with pytest.raises(TaskStoreError, match="other candidates"):
        store.save_candidates(generation=0, directories=candidates(tmp_path, generated("alpha")))
    with pytest.raises(TaskStoreError, match="no candidate 2"):
        store.publish_task(generation=0, index=2, split="train")
    with pytest.raises(TaskStoreError, match="candidates without outcome"):
        store.complete_generation(generation=0)

    store.publish_task(generation=0, index=0, split="train")
    store.reject_task(generation=0, index=1, reason="invalid")
    with pytest.raises(TaskConflict, match="published to train, not eval"):
        store.publish_task(generation=0, index=0, split="eval")
    with pytest.raises(TaskStoreError, match="is published"):
        store.reject_task(generation=0, index=0, reason="invalid")
    with pytest.raises(TaskStoreError, match="is rejected"):
        store.publish_task(generation=0, index=1, split="train")
    with pytest.raises(TaskStoreError, match="another reason"):
        store.reject_task(generation=0, index=1, reason="duplicate")


def test_a_task_name_is_published_once(session, tmp_path):
    store = session.tasks.open("tasks")
    store.begin_generation(generation=0, source_record_ids=("record-1",))
    store.save_candidates(generation=0, directories=candidates(tmp_path, generated("alpha")))
    first = store.publish_task(generation=0, index=0, split="eval")
    store.begin_generation(generation=1, source_record_ids=("record-1",))
    store.save_candidates(
        generation=1,
        directories=candidates(
            tmp_path, generated("alpha"), generated("alpha"), generated("alpha", instruction="Other.")
        ),
    )

    with pytest.raises(TaskConflict, match="published to eval, not train"):
        store.publish_task(generation=1, index=0, split="train")
    assert store.publish_task(generation=1, index=1, split="eval") == first
    with pytest.raises(TaskConflict, match="published with other files"):
        store.publish_task(generation=1, index=2, split="eval")
    assert store.tasks() == (first,)


def binary_task(root: Path, name: str, content: bytes = bytes(range(256))) -> Path:
    directory = root / name
    (directory / "environment").mkdir(parents=True)
    (directory / "environment" / "image.png").write_bytes(content)
    (directory / "instruction.md").write_text("Describe the image.\n")
    return directory


def test_an_added_task_keeps_its_binary_files_and_publication_order(session, tmp_path):
    store = session.tasks.open("tasks")
    store.begin_generation(generation=0, source_record_ids=("record-1",))
    store.save_candidates(generation=0, directories=candidates(tmp_path, generated("alpha"), generated("beta")))
    store.publish_task(generation=0, index=1, split="train")
    source = binary_task(tmp_path, "image-task")

    added = store.add_task(directory=source, split="eval")
    store.publish_task(generation=0, index=0, split="train")

    assert added == StoredTask(name="image-task", split="eval", files_digest=files_digest(source))
    assert (store.path(added.files_digest) / "environment" / "image.png").read_bytes() == bytes(range(256))
    assert store.add_task(directory=source, split="eval") == added
    assert [task.name for task in store.tasks()] == ["beta", "image-task", "alpha"]
    assert store.tasks(split="eval") == (added,)


def test_an_added_task_conflicts_with_a_published_name(session, tmp_path):
    store = session.tasks.open("tasks")
    store.add_task(directory=binary_task(tmp_path / "first", "image-task"), split="eval")
    with pytest.raises(TaskConflict, match="published to eval, not train"):
        store.add_task(directory=binary_task(tmp_path / "second", "image-task"), split="train")
    with pytest.raises(TaskConflict, match="other files"):
        store.add_task(directory=binary_task(tmp_path / "third", "image-task", b"other"), split="eval")
    store.begin_generation(generation=0, source_record_ids=("record-1",))
    store.save_candidates(generation=0, directories=candidates(tmp_path, generated("image-task")))
    with pytest.raises(TaskConflict, match="other files"):
        store.publish_task(generation=0, index=0, split="eval")
    with pytest.raises(ValueError, match="split"):
        store.add_task(directory=binary_task(tmp_path / "fourth", "other-task"), split="test")
    assert [task.name for task in store.tasks()] == ["image-task"]


def test_store_names_are_separate(session, tmp_path):
    alpha = session.tasks.open("alpha")
    alpha.begin_generation(generation=0, source_record_ids=("record-1",))
    alpha.save_candidates(generation=0, directories=candidates(tmp_path, generated("one")))
    alpha.publish_task(generation=0, index=0, split="train")

    beta = session.tasks.open("beta")
    assert beta.tasks() == ()
    assert beta.generations() == ()
    assert [task.name for task in session.tasks.open("alpha").tasks()] == ["one"]


def test_a_closed_session_opens_no_task_store(storage):
    session = storage.open("math")
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.tasks.open("tasks")


@pytest.fixture(params=["sqlite", "postgres"])
def open_durable_storage(request, tmp_path):
    if request.param == "postgres":
        url, schema = request.getfixturevalue("postgres_config")
        return lambda: PostgresScenarioStorage(url, tmp_path, schema=schema)
    return lambda: SQLiteScenarioStorage(tmp_path)


def test_tasks_survive_a_restart_and_archive_retires_them(open_durable_storage, tmp_path):
    sources = tmp_path / "source"
    sources.mkdir()
    with closing(open_durable_storage()) as storage, closing(storage.open("math")) as session:
        store = session.tasks.open("tasks")
        store.begin_generation(generation=0, source_record_ids=("record-1",))
        store.save_candidates(generation=0, directories=candidates(sources, generated("alpha"), generated("beta")))
        store.publish_task(generation=0, index=0, split="train")
        added = store.add_task(directory=binary_task(sources, "image-task"), split="eval")

    with closing(open_durable_storage()) as storage:
        with closing(storage.open("math")) as session:
            store = session.tasks.open("tasks")
            state = store.generation(0)
            assert state is not None and state.pending_indices == (1,)
            assert read_harbor_task(store.path(state.candidates[1].files_digest)) == generated("beta")
            published, restored = store.tasks()
            assert read_harbor_task(store.path(published.files_digest)) == generated("alpha")
            assert restored == added
            assert (store.path(restored.files_digest) / "environment" / "image.png").read_bytes() == bytes(range(256))
        storage.archive("math")
        with closing(storage.open("math")) as session:
            store = session.tasks.open("tasks")
            assert store.generations() == ()
            with pytest.raises(TaskStoreError, match="no files"):
                store.path(published.files_digest)


def test_an_archived_postgres_session_writes_no_task(postgres_config, tmp_path):
    url, schema = postgres_config
    with (
        closing(PostgresScenarioStorage(url, tmp_path, schema=schema)) as storage,
        closing(storage.open("math")) as session,
    ):
        store = session.tasks.open("tasks")
        storage.archive("math")
        with pytest.raises(RuntimeError, match="archived"):
            store.begin_generation(generation=0, source_record_ids=("record-1",))


def test_the_local_file_store_keeps_binary_files_by_digest(tmp_path):
    source = tmp_path / "source" / "image-task"
    (source / "environment").mkdir(parents=True)
    (source / "environment" / "data.bin").write_bytes(bytes(range(256)))
    (source / "instruction.md").write_text("Describe the image.\n")

    with closing(LocalTaskFileStore(tmp_path / "files")) as files:
        digest = files.put("math", source)
        assert files.put("math", source) == digest == files_digest(source)
        copy = files.path("math", digest)
        assert copy.name == "image-task"
        assert (copy / "environment" / "data.bin").read_bytes() == bytes(range(256))
        with pytest.raises(TaskStoreError, match="no files"):
            files.path("code", digest)

        (source / "link").symlink_to(source / "instruction.md")
        with pytest.raises(TaskStoreError, match="not a file"):
            files.put("math", source)

        (archived,) = files.archive("math")
        assert (Path(archived) / digest / "image-task" / "environment" / "data.bin").exists()
        with pytest.raises(TaskStoreError, match="no files"):
            files.path("math", digest)


@dataclass(frozen=True)
class SubdirectorySettings(PluginSettings):
    subdirectory: str = config_option("", help="Subdirectory of the task files.")


class SubdirectoryTaskFileStore(LocalTaskFileStore):
    """A plugin store that keeps its files in a configured subdirectory of ``storage.task_file_dir``."""

    @classmethod
    def from_config(cls, config: dict[str, Any], *, directory: Path) -> Self:
        settings = SubdirectorySettings.parse(config, plugin_class=cls)
        return cls(directory / settings.subdirectory)


@pytest.mark.parametrize(
    ("backend", "files_root"),
    [
        ("local", "files"),
        ({"module": f"{__name__}:SubdirectoryTaskFileStore", "config": {"subdirectory": "plugin"}}, "files/plugin"),
    ],
)
def test_the_deployment_selects_the_task_file_store(tmp_path, backend, files_root):
    settings = ServiceConfig(
        recipe="recipe",
        agent_record_dir=str(tmp_path / "records"),
        artifact_repository=str(tmp_path / "artifacts.git"),
        artifact_work_dir=str(tmp_path / "work"),
        artifact_cache_dir=str(tmp_path / "cache"),
        task_file_backend=backend,
        task_file_dir=str(tmp_path / "files"),
    )
    dispatcher = build_dispatcher(settings)
    try:
        store = dispatcher.get_or_create_scenario("math").store.tasks.open("tasks")
        store.begin_generation(generation=0, source_record_ids=("record-1",))
        store.save_candidates(generation=0, directories=candidates(tmp_path, generated("alpha")))
        directory = store.path(store.publish_task(generation=0, index=0, split="train").files_digest)
        assert directory.is_relative_to(tmp_path / files_root)
    finally:
        dispatcher.close()


@pytest.mark.parametrize(
    "backend",
    ["unknown", "reef.storage.tasks:TaskFileStore", {"module": "reef.storage.sqlite:SQLiteScenarioStorage"}],
)
def test_the_deployment_rejects_an_invalid_task_file_backend(backend):
    with pytest.raises(ValueError, match="task_file_backend"):
        ServiceConfig(recipe="recipe", task_file_backend=backend)
