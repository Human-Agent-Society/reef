"""Generated task store contract on every bundled scenario storage."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import pytest

from reef.core.config import config_option
from reef.core.plugins import PluginSettings
from reef.core.tasks import GeneratedTask, HarborTask, PlayArm, read_harbor_task
from reef.service.assembly import build_dispatcher
from reef.service.deploy.service_config import ServiceConfig
from reef.storage.postgres import PostgresScenarioStorage
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.storage.tasks import (
    GenerationState,
    LocalTaskFileStore,
    SplitSettings,
    StoredTask,
    TaskCandidate,
    TaskConflict,
    TaskStoreError,
    files_digest,
)

SPLIT = SplitSettings(eval_fraction=0.25, seed=7)


def generated(name: str, *, instruction: str = "", sources: tuple[str, ...] = ("record-1",)) -> GeneratedTask:
    return GeneratedTask(
        task=HarborTask(
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
        ),
        measurement_arms=(PlayArm("hinted", 2, ("solution/hint.txt",)),),
    )


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


def test_a_generation_records_candidates_and_outcomes(session):
    store = session.tasks.open("tasks", split=SPLIT)
    candidates = (generated("alpha"), generated("beta", sources=("record-2",)), generated("gamma"))

    assert store.begin_generation(generation=0, source_record_ids=("record-1", "record-2")) == GenerationState(
        generation=0, source_record_ids=("record-1", "record-2")
    )
    assert store.save_candidates(generation=0, candidates=candidates).candidates == tuple(
        TaskCandidate(item) for item in candidates
    )
    alpha = store.publish_task(generation=0, index=0, split="train")
    store.publish_task(generation=0, index=1, split="eval")
    store.reject_task(generation=0, index=2, reason="the oracle failed")
    completed = store.complete_generation(generation=0)

    assert alpha == StoredTask(
        generated=candidates[0], split="train", generation=0, index=0, files_digest=alpha.files_digest
    )
    directory = store.path(alpha)
    assert directory.name == "alpha"
    assert read_harbor_task(directory) == candidates[0].task
    assert completed == GenerationState(
        generation=0,
        source_record_ids=("record-1", "record-2"),
        candidates=(
            TaskCandidate(candidates[0], split="train"),
            TaskCandidate(candidates[1], split="eval"),
            TaskCandidate(candidates[2], error="the oracle failed"),
        ),
        is_complete=True,
    )
    assert store.generation(0) == completed
    assert store.generation(1) is None
    assert store.generations() == (completed,)
    assert [task.generated.task.name for task in store.tasks()] == ["alpha", "beta"]
    assert [task.generated for task in store.tasks(split="eval")] == [candidates[1]]


def test_identical_retries_change_nothing(session):
    store = session.tasks.open("tasks", split=SPLIT)
    candidates = (generated("alpha"), generated("beta"))
    for _ in range(2):
        store.begin_generation(generation=3, source_record_ids=("record-1",))
        store.save_candidates(generation=3, candidates=candidates)
        store.publish_task(generation=3, index=0, split="train")
        store.reject_task(generation=3, index=1, reason="invalid")
        store.complete_generation(generation=3)

    state = store.generation(3)
    assert state is not None and state.is_complete
    assert state.candidates == (
        TaskCandidate(candidates[0], split="train"),
        TaskCandidate(candidates[1], error="invalid"),
    )
    assert len(store.tasks()) == 1


def test_conflicting_generation_writes_are_refused(session):
    store = session.tasks.open("tasks", split=SPLIT)
    with pytest.raises(TaskStoreError, match="does not exist"):
        store.save_candidates(generation=0, candidates=(generated("alpha"),))
    store.begin_generation(generation=0, source_record_ids=("record-1",))
    with pytest.raises(TaskStoreError, match="other source records"):
        store.begin_generation(generation=0, source_record_ids=("record-2",))
    store.save_candidates(generation=0, candidates=(generated("alpha"), generated("beta")))
    with pytest.raises(TaskStoreError, match="other candidates"):
        store.save_candidates(generation=0, candidates=(generated("alpha"),))
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


def test_a_task_name_is_published_once(session):
    store = session.tasks.open("tasks", split=SPLIT)
    store.begin_generation(generation=0, source_record_ids=("record-1",))
    store.save_candidates(generation=0, candidates=(generated("alpha"),))
    first = store.publish_task(generation=0, index=0, split="eval")
    store.begin_generation(generation=1, source_record_ids=("record-1",))
    store.save_candidates(
        generation=1, candidates=(generated("alpha"), generated("alpha"), generated("alpha", instruction="Other."))
    )

    with pytest.raises(TaskConflict, match="published to eval, not train"):
        store.publish_task(generation=1, index=0, split="train")
    assert store.publish_task(generation=1, index=1, split="eval") == first
    with pytest.raises(TaskConflict, match="published with other files"):
        store.publish_task(generation=1, index=2, split="eval")
    assert store.tasks() == (first,)


def test_split_settings_are_fixed_and_names_are_separate(session):
    alpha = session.tasks.open("alpha", split=SPLIT)
    alpha.begin_generation(generation=0, source_record_ids=("record-1",))
    alpha.save_candidates(generation=0, candidates=(generated("one"),))
    alpha.publish_task(generation=0, index=0, split="train")

    with pytest.raises(TaskStoreError, match=r"eval_fraction 0\.25 and seed 7"):
        session.tasks.open("alpha", split=SplitSettings(eval_fraction=0.5, seed=7))
    beta = session.tasks.open("beta", split=SplitSettings(eval_fraction=0.0, seed=0))
    assert beta.tasks() == ()
    assert beta.generations() == ()
    assert [task.generated.task.name for task in session.tasks.open("alpha", split=SPLIT).tasks()] == ["one"]


def test_a_closed_session_opens_no_task_store(storage):
    session = storage.open("math")
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.tasks.open("tasks", split=SPLIT)


@pytest.fixture(params=["sqlite", "postgres"])
def open_durable_storage(request, tmp_path):
    if request.param == "postgres":
        url, schema = request.getfixturevalue("postgres_config")
        return lambda: PostgresScenarioStorage(url, tmp_path, schema=schema)
    return lambda: SQLiteScenarioStorage(tmp_path)


def test_tasks_survive_a_restart_and_archive_retires_them(open_durable_storage):
    with closing(open_durable_storage()) as storage, closing(storage.open("math")) as session:
        store = session.tasks.open("tasks", split=SPLIT)
        store.begin_generation(generation=0, source_record_ids=("record-1",))
        store.save_candidates(generation=0, candidates=(generated("alpha"), generated("beta")))
        store.publish_task(generation=0, index=0, split="train")

    with closing(open_durable_storage()) as storage:
        with closing(storage.open("math")) as session:
            store = session.tasks.open("tasks", split=SPLIT)
            state = store.generation(0)
            assert state is not None and state.pending_indices == (1,)
            (published,) = store.tasks()
            assert published.generated == generated("alpha")
            assert read_harbor_task(store.path(published)) == generated("alpha").task
        storage.archive("math")
        with closing(storage.open("math")) as session:
            store = session.tasks.open("tasks", split=SPLIT)
            assert store.generations() == ()
            with pytest.raises(TaskStoreError, match="no files"):
                store.path(published)


def test_an_archived_postgres_session_writes_no_task(postgres_config, tmp_path):
    url, schema = postgres_config
    with (
        closing(PostgresScenarioStorage(url, tmp_path, schema=schema)) as storage,
        closing(storage.open("math")) as session,
    ):
        store = session.tasks.open("tasks", split=SPLIT)
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
        store = dispatcher.get_or_create_scenario("math").store.tasks.open("tasks", split=SPLIT)
        store.begin_generation(generation=0, source_record_ids=("record-1",))
        store.save_candidates(generation=0, candidates=(generated("alpha"),))
        directory = store.path(store.publish_task(generation=0, index=0, split="train"))
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
