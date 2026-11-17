"""Task store contract on every bundled scenario storage."""

from __future__ import annotations

import hashlib
import tempfile
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import pytest

from reef.core.config import config_option
from reef.core.errors import ScenarioDeletedError
from reef.core.plugins import PluginSettings
from reef.core.tasks import HarborTask, read_harbor_task, write_harbor_task
from reef.service.assembly import build_dispatcher
from reef.service.deploy.service_config import ServiceSettings
from reef.storage.postgres import PostgresScenarioStorage
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.storage.tasks import (
    MAX_TASK_DIRECTORY_ENTRIES,
    MAX_TASK_FILE_BYTES,
    MAX_TASK_FILES,
    FilesDigest,
    GenerationState,
    LocalTaskFileStore,
    StoredTask,
    TaskCandidate,
    TaskConflictError,
    TaskLimitError,
    TaskStoreError,
    files_digest,
    task_file_order,
    task_limit_errors,
)


def generated(name: str, *, instruction: str = "", sources: tuple[str, ...] = ("record-1",)) -> HarborTask:
    return HarborTask(
        name=name,
        files={
            "instruction.md": instruction or f"Solve {name}.",
            "tests/test.sh": "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n",
            "environment/Dockerfile": "FROM python:3.12-slim\n",
            "solution/hint.txt": "look closer",
        },
        config={
            "verifier": {"timeout_sec": 30},
            "environment": {"network_mode": "allowlist", "allowed_hosts": ["pypi.org"]},
        },
        metadata={"designer_record_id": "designer-1"},
        source_agent_record_ids=sources,
    )


def candidates(root: Path, *tasks: HarborTask) -> tuple[Path, ...]:
    """Write each task in a directory of its own, as the record-to-sample stage does."""
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
    assert store.write_candidates(generation=0, directories=directories).candidates == tuple(
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
        store.write_candidates(generation=3, directories=candidates(tmp_path, *tasks))
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
        store.write_candidates(generation=0, directories=candidates(tmp_path, generated("alpha")))
    store.begin_generation(generation=0, source_record_ids=("record-1",))
    with pytest.raises(TaskStoreError, match="other source records"):
        store.begin_generation(generation=0, source_record_ids=("record-2",))
    store.write_candidates(generation=0, directories=candidates(tmp_path, generated("alpha"), generated("beta")))
    with pytest.raises(TaskStoreError, match="other candidates"):
        store.write_candidates(generation=0, directories=candidates(tmp_path, generated("alpha")))
    with pytest.raises(TaskStoreError, match="no candidate 2"):
        store.publish_task(generation=0, index=2, split="train")
    with pytest.raises(TaskStoreError, match="candidates without outcome"):
        store.complete_generation(generation=0)

    store.publish_task(generation=0, index=0, split="train")
    store.reject_task(generation=0, index=1, reason="invalid")
    with pytest.raises(TaskConflictError, match="published to train, not eval"):
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
    store.write_candidates(generation=0, directories=candidates(tmp_path, generated("alpha")))
    first = store.publish_task(generation=0, index=0, split="eval")
    store.begin_generation(generation=1, source_record_ids=("record-1",))
    store.write_candidates(
        generation=1,
        directories=candidates(
            tmp_path, generated("alpha"), generated("alpha"), generated("alpha", instruction="Other.")
        ),
    )

    with pytest.raises(TaskConflictError, match="published to eval, not train"):
        store.publish_task(generation=1, index=0, split="train")
    assert store.publish_task(generation=1, index=1, split="eval") == first
    with pytest.raises(TaskConflictError, match="published with other files"):
        store.publish_task(generation=1, index=2, split="eval")
    assert store.tasks() == (first,)


def test_documents_are_written_once_and_trained_marks_name_published_train_tasks(session, tmp_path):
    store = session.tasks.open("tasks")
    store.begin_generation(generation=0, source_record_ids=("record-1",))
    store.write_candidates(generation=0, directories=candidates(tmp_path, generated("alpha"), generated("beta")))
    store.publish_task(generation=0, index=0, split="train")
    store.publish_task(generation=0, index=1, split="eval")

    store.write_document(key="rollout/alpha", document='{"a":1}')
    store.write_document(key="rollout/alpha", document='{"a":1}')
    store.write_document(key="delivered/0", document="")
    assert store.documents() == {"rollout/alpha": '{"a":1}', "delivered/0": ""}, "an identical retry changes nothing"
    with pytest.raises(TaskStoreError, match="other text"):
        store.write_document(key="rollout/alpha", document='{"a":2}')
    with pytest.raises(TaskStoreError, match="non-empty key"):
        store.write_document(key="", document="x")
    assert session.tasks.open("other").documents() == {}


def test_a_state_document_is_replaced_and_removed_apart_from_the_documents(session):
    store = session.tasks.open("tasks")
    assert store.state_document("cursor") is None and store.state_document_keys() == ()
    store.write_document(key="cursor", document="written once")
    store.write_state_document(key="cursor", document='{"next":1}')
    store.write_state_document(key="cursor", document='{"next":2}')
    store.write_state_document(key="open/00010", document="ten")
    store.write_state_document(key="open/00002", document="")
    store.write_state_document(key="OPEN/00001", document="other prefix")
    store.write_state_document(key="open%", document="no pattern")
    assert store.state_document("cursor") == '{"next":2}', "a write replaces the document of the key"
    assert store.state_document("open/00002") == ""
    assert store.state_document_keys() == ("OPEN/00001", "cursor", "open%", "open/00002", "open/00010")
    assert store.state_document_keys(prefix="open/") == ("open/00002", "open/00010"), "the prefix is exact"
    assert store.state_document_keys(prefix="open%") == ("open%",)

    store.delete_state_document(key="open/00002")
    store.delete_state_document(key="open/00002")
    store.delete_state_document(key="unknown")
    assert store.state_document("open/00002") is None
    assert store.state_document_keys(prefix="open/") == ("open/00010",), "an unknown key changes nothing"
    with pytest.raises(TaskStoreError, match="non-empty key"):
        store.write_state_document(key="", document="x")
    assert store.documents() == {"cursor": "written once"}, "the two kinds of document have separate keys"
    assert session.tasks.open("other").state_document_keys() == ()


def test_saved_files_are_read_by_their_digest_and_change_no_generation(session, tmp_path):
    store = session.tasks.open("tasks")
    (directory,) = candidates(tmp_path, generated("parent"))
    digest = store.write_files(directory)
    assert digest == files_digest(directory) == store.write_files(directory), "a repeated save changes nothing"
    assert store.path(digest).name == "parent" and read_harbor_task(store.path(digest)) == generated("parent")
    assert store.generations() == () and store.tasks() == ()
    (directory / "link").symlink_to(directory / "instruction.md")
    with pytest.raises(TaskStoreError, match="not a file"):
        store.write_files(directory)


def test_store_names_are_separate(session, tmp_path):
    alpha = session.tasks.open("alpha")
    alpha.begin_generation(generation=0, source_record_ids=("record-1",))
    alpha.write_candidates(generation=0, directories=candidates(tmp_path, generated("one")))
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
        store.write_candidates(generation=0, directories=candidates(sources, generated("alpha"), generated("beta")))
        store.publish_task(generation=0, index=0, split="train")
        store.write_document(key="rollout/alpha", document="saved")
        store.write_state_document(key="cursor", document="first")
        store.write_state_document(key="cursor", document="second")
        store.write_state_document(key="open/00000", document="removed")
        store.delete_state_document(key="open/00000")
        saved_digest = store.write_files(candidates(sources, generated("parent"))[0])

    with closing(open_durable_storage()) as storage:
        with closing(storage.open("math")) as session:
            store = session.tasks.open("tasks")
            state = store.generation(0)
            assert state is not None and state.pending_indices == (1,)
            assert read_harbor_task(store.path(state.candidates[1].files_digest)) == generated("beta")
            (published,) = store.tasks()
            assert read_harbor_task(store.path(published.files_digest)) == generated("alpha")
            assert store.documents() == {"rollout/alpha": "saved"}
            assert store.state_document_keys() == ("cursor",) and store.state_document("cursor") == "second"
            assert read_harbor_task(store.path(saved_digest)) == generated("parent")
        storage.archive("math")
        with pytest.raises(ScenarioDeletedError):
            storage.open("math")


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
        digest = files.write("math", source)
        assert files.write("math", source) == digest == files_digest(source)
        copy = files.path("math", digest)
        assert copy.name == "image-task"
        assert (copy / "environment" / "data.bin").read_bytes() == bytes(range(256))
        with pytest.raises(TaskStoreError, match="no files"):
            files.path("code", digest)

        (source / "link").symlink_to(source / "instruction.md")
        with pytest.raises(TaskStoreError, match="not a file"):
            files.write("math", source)

        (archived,) = files.archive("math")
        assert (Path(archived) / digest / "image-task" / "environment" / "data.bin").exists()
        with pytest.raises(TaskStoreError, match="no files"):
            files.path("math", digest)


def tricky_task(root: Path) -> Path:
    """A task directory whose names sort differently as strings and as path parts, with empty files and directories."""
    source = root / "tricky-task"
    for name, content in {
        "a-b": b"dash",
        "a.b": b"dot",
        "a b": b"space",
        "_x": b"",
        "a/b": b"nested",
        "a/b-c/d": b"deeper",
        "a/b.c": b"",
        "a0/z": b"zero",
        "\u00e9t\u00e9.md": b"accents",
        "z/empty.txt": b"",
        "big.bin": bytes(range(256)) * 13,
    }.items():
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        (source / name).write_bytes(content)
    (source / "empty-dir").mkdir()
    (source / "a" / "empty" / "deeper").mkdir(parents=True)
    return source


def test_files_digest_keeps_its_value_and_a_stream_of_files_gives_the_same_digest(tmp_path, monkeypatch):
    source = tricky_task(tmp_path)
    # The digest of this tree before ``FilesDigest``: stored digests stay valid.
    expected = "3ddb08ecfba184b01aaa63d827e7f315bcac4236af0eabc84774ef597b1b099a"
    assert files_digest(source) == expected
    monkeypatch.setattr("reef.storage.tasks.DIGEST_CHUNK_BYTES", 5)
    assert files_digest(source) == expected
    # The order of rglob is the order of the path parts: "a/b" comes before "a-b".
    files = sorted(
        (path.relative_to(source).as_posix() for path in source.rglob("*") if path.is_file()), key=task_file_order
    )
    assert files[:3] == ["_x", "a/b", "a/b-c/d"] and files.index("a/b.c") < files.index("a b") < files.index("a-b")
    digest = FilesDigest("tricky-task")
    for name in files:
        content = (source / name).read_bytes()
        digest.add_file(name, len(content))
        for start in range(0, len(content), 1000):
            digest.update(content[start : start + 1000])
    assert digest.hexdigest() == expected
    # Empty directories add nothing; the directory name is part of the digest.
    assert FilesDigest("empty").hexdigest() == hashlib.sha256(b"empty").hexdigest()


def test_files_digest_refuses_files_out_of_order_or_of_another_size():
    digest = FilesDigest("task")
    digest.add_file("b", 2)
    digest.update(b"x")
    with pytest.raises(TaskStoreError, match="1 bytes before its size"):
        digest.add_file("c", 0)
    with pytest.raises(TaskStoreError, match="past the size"):
        digest.update(b"yz")
    with pytest.raises(TaskStoreError, match="1 bytes before its size"):
        digest.hexdigest()
    digest.update(b"y")
    for name in ("b", "a", "a-b/c"):
        with pytest.raises(TaskStoreError, match="does not come after b"):
            digest.add_file(name, 0)
    for name in ("", "/abs", "c//d", "c/../d", "./c"):
        with pytest.raises(TaskStoreError, match="not a relative path"):
            digest.add_file(name, 0)
    with pytest.raises(TaskStoreError, match="non-negative"):
        digest.add_file("c", -1)


def test_task_limits_count_files_sizes_and_the_entries_of_each_directory():
    assert task_limit_errors([("instruction.md", 10), ("tests", None), ("tests/test.sh", MAX_TASK_FILE_BYTES)]) == []
    many = [(f"environment/assets/{index}.png", 1) for index in range(MAX_TASK_DIRECTORY_ENTRIES + 1)]
    top = [(f"file-{index}", 1) for index in range(MAX_TASK_DIRECTORY_ENTRIES)]
    nested = [(f"data/{index // 9000}/{index}", 0) for index in range(MAX_TASK_FILES - len(many) - len(top))]
    # The top directory holds 10,000 files, "huge", "environment" and "data", which no entry gives. A path given
    # twice counts once.
    huge = ("huge", MAX_TASK_FILE_BYTES + 1)
    errors = task_limit_errors([huge, *many, *top, *nested, huge])
    assert [(error.code, error.path) for error in errors] == [
        ("file_too_large", "huge"),
        ("too_many_entries", ""),
        ("too_many_entries", "environment/assets"),
        ("too_many_files", ""),
    ]
    assert [error.message for error in errors] == [
        "huge has 500,000,000,001 bytes. The limit is 500,000,000,000 bytes.",
        "The task directory has 10,003 entries. The limit is 10,000.",
        "environment/assets has 10,001 entries. The limit is 10,000.",
        "The task has 100,001 files. The limit is 100,000.",
    ]
    for entries, reason in (([("../x", 1)], "not a relative path"), ([("x", -1)], "non-negative")):
        with pytest.raises(ValueError, match=reason):
            task_limit_errors(entries)
    with pytest.raises(ValueError, match="code"):
        TaskLimitError("too_big", "x", "x is too big.")  # type: ignore[arg-type]


def test_the_local_file_store_contains_a_complete_directory_by_partition_digest_and_name(tmp_path):
    source = tricky_task(tmp_path)
    with closing(LocalTaskFileStore(tmp_path / "files")) as files:
        digest = files_digest(source)
        assert not files.contains("data/acme", digest, "tricky-task")
        assert files.write("data/acme", source) == digest
        assert files.contains("data/acme", digest, "tricky-task")
        assert not files.contains("data/acme", digest, "other-task")
        assert not files.contains("data/other", digest, "tricky-task")
        assert not files.contains("data/acme", "0" * 64, "tricky-task")
        with pytest.raises(ValueError, match="not a files digest"):
            files.contains("data/acme", "../" + digest, "tricky-task")


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
    settings = ServiceSettings(
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
        store.write_candidates(generation=0, directories=candidates(tmp_path, generated("alpha")))
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
        ServiceSettings(recipe="recipe", task_file_backend=backend)
