"""A task written as a Harbor directory: whole or not at all, read back, and refused once edited."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import tomllib
import warnings
from pathlib import Path

import pytest

from reef.core.tasks import (
    TASK_CONFIG_VERSION,
    HarborTask,
    HarborTaskConflict,
    HarborTaskError,
    import_harbor_task,
    read_harbor_task,
    write_harbor_task,
)

VERIFIER = (
    "#!/bin/sh\nset -eu\n"
    '[ "$(cat /workspace/answer.txt)" = 391 ] && echo 1 > /logs/verifier/reward.txt || echo 0 > /logs/verifier/reward.txt\n'
)


def task(**overrides: object) -> HarborTask:
    fields: dict[str, object] = {
        "name": "sum-391",
        "instruction": "Add 137 and 254 and write the sum to /workspace/answer.txt.",
        "tests": {"test.sh": VERIFIER},
        "environment": {"Dockerfile": "FROM python:3.12-slim\nWORKDIR /workspace\n"},
        "config": {
            "verifier": {"timeout_sec": 30},
            "agent": {"timeout_sec": 300},
            "environment": {"cpus": 1, "memory_mb": 512, "storage_mb": 1024, "gpus": 0},
        },
        "metadata": {"author_name": "Reef", "tags": ["reef", "arithmetic"]},
        "source_agent_record_ids": ("rec-inference-1", "rec-report-1"),
    }
    fields.update(overrides)
    return HarborTask(**fields)  # type: ignore[arg-type]


def files_of(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


# ----------------------------------------------------------------------------------------------- writing


def test_write_lays_out_the_directory_harbor_loads(tmp_path: Path) -> None:
    root = write_harbor_task(task(), tmp_path / "tasks")
    assert root == tmp_path / "tasks" / "sum-391"
    assert files_of(root) == ["environment/Dockerfile", "instruction.md", "task.toml", "tests/test.sh"]
    assert (root / "instruction.md").read_text() == task().instruction
    assert (root / "environment" / "Dockerfile").read_text().startswith("FROM python")
    test_sh = root / "tests" / "test.sh"
    assert test_sh.read_text() == VERIFIER
    assert os.access(test_sh, os.X_OK)
    document = tomllib.loads((root / "task.toml").read_text())
    assert document["version"] == TASK_CONFIG_VERSION
    assert document["verifier"] == {"timeout_sec": 30.0}
    assert document["agent"] == {"timeout_sec": 300.0}
    assert document["environment"] == {"cpus": 1, "memory_mb": 512, "storage_mb": 1024, "gpus": 0}
    assert document["metadata"]["author_name"] == "Reef"
    assert document["metadata"]["tags"] == ["reef", "arithmetic"]
    assert document["metadata"]["reef"] == {
        "digest": task().digest,
        "source_agent_record_ids": ["rec-inference-1", "rec-report-1"],
    }


def test_the_published_directory_has_the_umask_mode_not_a_private_one(tmp_path: Path) -> None:
    mask = os.umask(0o022)
    os.umask(mask)
    root = write_harbor_task(task(), tmp_path)
    assert root.stat().st_mode & 0o777 == 0o777 & ~mask
    assert sorted(p.name for p in tmp_path.iterdir()) == [".staging", root.name]


def test_helpers_solution_and_nested_environment_files_are_written(tmp_path: Path) -> None:
    root = write_harbor_task(
        task(
            tests={"test.sh": "#!/bin/sh\npython3 /tests/grade.py\n", "grade.py": "print('graded')\n"},
            environment={"Dockerfile": "FROM python:3.12-slim\n", "data/input.csv": "id,value\n1,2\n"},
            solution={"solve.sh": "#!/bin/sh\necho 391 > /workspace/answer.txt\n"},
        ),
        tmp_path,
    )
    assert (root / "tests" / "grade.py").read_text() == "print('graded')\n"
    assert (root / "environment" / "data" / "input.csv").read_text() == "id,value\n1,2\n"
    assert (root / "solution" / "solve.sh").read_text().startswith("#!/bin/sh")


def test_carriage_returns_survive_the_round_trip(tmp_path: Path) -> None:
    spec = task(
        instruction="line one\r\nline two\r\n",
        tests={"test.sh": "#!/bin/sh\r\necho 1 > /logs/verifier/reward.txt\r\n"},
        environment={"Dockerfile": "FROM x\r\n", "data.csv": "a,b\rc,d\n"},
    )
    root = write_harbor_task(spec, tmp_path)
    assert (root / "instruction.md").read_bytes() == b"line one\r\nline two\r\n"
    assert read_harbor_task(root) == spec
    assert write_harbor_task(spec, tmp_path) == root


def test_the_same_task_written_again_is_a_no_op(tmp_path: Path) -> None:
    first = write_harbor_task(task(), tmp_path)
    before = sorted((p.relative_to(first).as_posix(), p.stat().st_mtime_ns) for p in first.rglob("*") if p.is_file())
    again = write_harbor_task(task(), tmp_path)
    after = sorted((p.relative_to(again).as_posix(), p.stat().st_mtime_ns) for p in again.rglob("*") if p.is_file())
    assert again == first
    assert after == before


def test_a_different_task_under_the_same_name_is_a_conflict(tmp_path: Path) -> None:
    write_harbor_task(task(), tmp_path)
    with pytest.raises(HarborTaskConflict, match="different task with the same name"):
        write_harbor_task(task(instruction="Add 1 and 1."), tmp_path)
    with pytest.raises(HarborTaskConflict, match="different task with the same name"):
        write_harbor_task(task(source_agent_record_ids=("rec-other",)), tmp_path)


def test_a_foreign_directory_under_the_name_is_a_conflict(tmp_path: Path) -> None:
    (tmp_path / "sum-391").mkdir()
    (tmp_path / "sum-391" / "task.toml").write_text('version = "1.0"\n')
    with pytest.raises(HarborTaskConflict, match="not a task reef wrote"):
        write_harbor_task(task(), tmp_path)


def test_an_unreadable_top_level_file_is_a_conflict_not_a_crash(tmp_path: Path) -> None:
    root = write_harbor_task(task(), tmp_path)
    (root / "instruction.md").write_bytes(b"\xff\xfe not utf-8")
    with pytest.raises(HarborTaskConflict, match="not UTF-8 text"):
        write_harbor_task(task(), tmp_path)


def test_a_failed_write_leaves_nothing_behind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import reef.core.tasks.harbor as module

    def explode(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(module, "write_task_files", explode)
    with pytest.raises(OSError, match="disk full"):
        write_harbor_task(task(), tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [".staging"]
    assert list((tmp_path / ".staging").iterdir()) == []


def test_a_rename_that_fails_for_its_own_reason_is_not_reported_as_a_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import reef.core.tasks.harbor as module

    def refuse(*args: object, **kwargs: object) -> None:
        raise PermissionError("read-only volume")

    monkeypatch.setattr(module.os, "rename", refuse)
    with pytest.raises(PermissionError, match="read-only volume"):
        write_harbor_task(task(), tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [".staging"]
    assert list((tmp_path / ".staging").iterdir()) == []


def test_a_writer_that_lost_the_race_accepts_the_same_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import reef.core.tasks.harbor as module

    real_rename = module.os.rename

    def race(source: object, destination: object) -> None:
        # The other writer lands the same task first, so this rename finds the target taken.
        winner = tmp_path / "winner-staging"
        winner.mkdir()
        module.write_task_files(task(), winner)
        real_rename(winner, destination)
        real_rename(source, destination)

    monkeypatch.setattr(module.os, "rename", race)
    assert write_harbor_task(task(), tmp_path) == tmp_path / "sum-391"
    assert read_harbor_task(tmp_path / "sum-391") == task()


def test_root_is_created_when_missing(tmp_path: Path) -> None:
    root = write_harbor_task(task(), tmp_path / "a" / "b")
    assert root.is_dir()


# ----------------------------------------------------------------------------------------------- reading


def test_read_returns_the_task_that_was_written(tmp_path: Path) -> None:
    written = task(solution={"solve.sh": "#!/bin/sh\necho 391 > /workspace/answer.txt\n"})
    root = write_harbor_task(written, tmp_path)
    assert read_harbor_task(root) == written
    assert read_harbor_task(root).digest == written.digest


@pytest.mark.parametrize(
    "edit",
    [
        ("instruction.md", "Add 1 and 1."),
        ("tests/test.sh", "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n"),
        ("environment/Dockerfile", "FROM ubuntu\n"),
    ],
)
def test_an_edited_file_is_refused(tmp_path: Path, edit: tuple[str, str]) -> None:
    root = write_harbor_task(task(), tmp_path)
    (root / edit[0]).write_text(edit[1])
    with pytest.raises(HarborTaskError, match="does not match its digest"):
        read_harbor_task(root)


def test_an_edited_task_toml_table_is_refused(tmp_path: Path) -> None:
    root = write_harbor_task(task(), tmp_path)
    text = (root / "task.toml").read_text().replace("timeout_sec = 300.0", "timeout_sec = 3.0")
    (root / "task.toml").write_text(text)
    with pytest.raises(HarborTaskError, match="does not match its digest"):
        read_harbor_task(root)


@pytest.mark.parametrize("extra", ["README.md", "tests/helper.sh", "trajectory.json", "environment/notes.txt"])
def test_a_file_reef_did_not_write_is_refused(tmp_path: Path, extra: str) -> None:
    root = write_harbor_task(task(), tmp_path)
    (root / extra).parent.mkdir(parents=True, exist_ok=True)
    (root / extra).write_text("x")
    with pytest.raises(HarborTaskError, match=r"did not write|does not match its digest"):
        read_harbor_task(root)


@pytest.mark.parametrize(
    "table", ['[solution]\nenv = { LEAK = "1" }\n', '[[steps]]\nname = "hand"\n', 'artifacts = ["/etc/passwd"]\n']
)
def test_a_task_toml_table_reef_did_not_write_is_refused(tmp_path: Path, table: str) -> None:
    root = write_harbor_task(task(), tmp_path)
    text = (root / "task.toml").read_text()
    (root / "task.toml").write_text(table + text if table.startswith("artifacts") else text + "\n" + table)
    with pytest.raises(HarborTaskError, match="tables reef did not write"):
        read_harbor_task(root)


def test_a_directory_without_the_reef_table_is_refused(tmp_path: Path) -> None:
    root = write_harbor_task(task(), tmp_path)
    (root / "task.toml").write_text('version = "1.0"\n[metadata]\nauthor_name = "Reef"\n')
    with pytest.raises(HarborTaskError, match=r"no \[metadata\.reef\] table"):
        read_harbor_task(root)


def test_a_toml_date_in_metadata_is_refused_not_crashed(tmp_path: Path) -> None:
    root = write_harbor_task(task(), tmp_path)
    text = (root / "task.toml").read_text().replace('author_name = "Reef"', "author_name = 2026-01-01")
    (root / "task.toml").write_text(text)
    with pytest.raises(HarborTaskError, match="strings, numbers, booleans"):
        read_harbor_task(root)


@pytest.mark.parametrize("missing", ["task.toml", "instruction.md", "tests/test.sh", "environment"])
def test_a_missing_part_is_named(tmp_path: Path, missing: str) -> None:
    root = write_harbor_task(task(), tmp_path)
    path = root / missing
    if path.is_dir():
        for child in path.iterdir():
            child.unlink()
        path.rmdir()
    else:
        path.unlink()
    with pytest.raises(HarborTaskError, match=r"is missing|must be non-empty"):
        read_harbor_task(root)


def test_read_refuses_a_path_that_is_not_a_directory(tmp_path: Path) -> None:
    with pytest.raises(HarborTaskError, match="not a task directory"):
        read_harbor_task(tmp_path / "nope")


# ----------------------------------------------------------------------------------------------- the spec


def test_digest_ignores_mapping_order_and_type() -> None:
    a = task(environment={"Dockerfile": "FROM x\n", "b.txt": "b", "a.txt": "a"}, metadata={"z": 1, "y": "2"})
    b = task(environment={"a.txt": "a", "b.txt": "b", "Dockerfile": "FROM x\n"}, metadata={"y": "2", "z": 1})
    assert a.digest == b.digest
    assert a.digest != task().digest


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"name": "Sum"}, "task name"),
        ({"name": "a/b"}, "task name"),
        ({"name": "a..b"}, "task name"),
        ({"name": "-lead"}, "task name"),
        ({"instruction": "  "}, "instruction must be non-empty"),
        ({"instruction": "bad \ud800 surrogate"}, "not valid Unicode"),
        ({"tests": {}}, "tests/test.sh must be non-empty"),
        ({"tests": {"helper.sh": "x"}}, "tests/test.sh must be non-empty"),
        ({"tests": {"test.sh": "   "}}, "tests/test.sh must be non-empty"),
        ({"environment": {}}, "needs a Dockerfile"),
        ({"environment": {"../Dockerfile": "x"}}, "stay inside"),
        ({"environment": {"/etc/passwd": "x"}}, "stay inside"),
        ({"environment": {".": "x", "Dockerfile": "FROM x\n"}}, "stay inside"),
        ({"environment": {"./Dockerfile": "x", "Dockerfile": "FROM x\n"}}, "written plainly"),
        ({"environment": {"a//b": "x", "Dockerfile": "FROM x\n"}}, "written plainly"),
        ({"environment": {"a\x00b": "x", "Dockerfile": "FROM x\n"}}, "control characters"),
        ({"environment": {"Dockerfile": "FROM x\n", "dockerfile": "x"}}, "fold together"),
        ({"environment": {"Dockerfile": "FROM x\n", "caf\u00e9.txt": "a", "cafe\u0301.txt": "b"}}, "fold together"),
        ({"environment": {"Dockerfile": "FROM x\n", "a": "x", "a/b": "y"}}, "fold together"),
        ({"environment": {"Dockerfile": b"bytes"}}, "must be text"),
        ({"environment": {"Dockerfile": "FROM x\n", "note.txt": "\udfff"}}, "not valid Unicode"),
        ({"solution": {"dir/../solve.sh": "x"}}, "stay inside"),
        ({"config": {"steps": {}}}, "not one reef writes"),
        ({"config": {"agent": {"workdir": "/x"}}}, "not a key Harbor reads"),
        ({"config": {"agent": {"timeout_sec": 0}}}, "positive number"),
        ({"config": {"agent": {"timeout_sec": True}}}, "positive number"),
        ({"config": {"agent": {"timeout_sec": float("nan")}}}, "positive number"),
        ({"config": {"verifier": {"timeout_sec": float("inf")}}}, "positive number"),
        ({"config": {"environment": {"cpus": -1}}}, "non-negative integer"),
        ({"config": {"environment": {"cpus": 1.5}}}, "non-negative integer"),
        ({"config": {"environment": {"network_mode": "lan"}}}, "one of"),
        ({"config": {"environment": {"allowed_hosts": ["a.example"]}}}, "needs network_mode"),
        ({"config": {"environment": {"allowed_hosts": "a.example", "network_mode": "allowlist"}}}, "list of host"),
        ({"config": {"verifier": {"env": {"K": 1}}}}, "variable names to strings"),
        ({"config": {"environment": {"docker_image": ""}}}, "non-empty string"),
        ({"metadata": {"reef": {}}}, "written by reef"),
        ({"metadata": {"when": object()}}, "strings, numbers, booleans"),
        ({"metadata": {"when": datetime.date(2026, 1, 1)}}, "strings, numbers, booleans"),
        ({"source_agent_record_ids": ("a", "a")}, "distinct"),
        ({"source_agent_record_ids": ["a"]}, "tuple"),
    ],
)
def test_a_bad_spec_is_refused_at_construction(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(HarborTaskError, match=message):
        task(**overrides)


@pytest.mark.parametrize(
    ("host", "normalized"),
    [
        ("pypi.org", "pypi.org"),
        ("  Files.PythonHosted.org.  ", "files.pythonhosted.org"),
        ("*.example.com", "*.example.com"),
        ("192.0.2.1", "192.0.2.1"),
        ("192.0.2.0/24", "192.0.2.0/24"),
        ("2001:db8::1", "2001:db8::1"),
        ("2001:db8::/32", "2001:db8::/32"),
    ],
)
def test_allowed_hosts_are_normalized_the_way_harbor_accepts_them(host: str, normalized: str) -> None:
    spec = task(config={"environment": {"network_mode": "allowlist", "allowed_hosts": [host]}})
    assert spec.config["environment"]["allowed_hosts"] == [normalized]


@pytest.mark.parametrize(
    "host",
    [
        "https://pypi.org",
        "pypi.org:443",
        "pypi.org/simple",
        "*",
        "*host.com",
        "a_b.example",
        "[2001:db8::1]",
        " ",
        "192.0.2.1/33",
        "*.192.0.2.1",
    ],
)
def test_allowed_hosts_harbor_rejects_are_refused_here(host: str) -> None:
    with pytest.raises(HarborTaskError, match=r"must be a host name|non-empty host"):
        task(config={"environment": {"network_mode": "allowlist", "allowed_hosts": [host]}})


def test_a_prebuilt_image_needs_no_dockerfile() -> None:
    spec = task(environment={"notes.txt": "x"}, config={"environment": {"docker_image": "python:3.12-slim"}})
    assert spec.config["environment"]["docker_image"] == "python:3.12-slim"


def test_a_verifier_may_write_reward_json_through_a_helper() -> None:
    spec = task(tests={"test.sh": "#!/bin/sh\npython3 /tests/grade.py\n", "grade.py": "..."})
    assert "grade.py" in spec.tests


def test_the_allowlist_form_round_trips(tmp_path: Path) -> None:
    spec = task(config={"environment": {"network_mode": "allowlist", "allowed_hosts": ["pypi.org"]}})
    root = write_harbor_task(spec, tmp_path)
    document = tomllib.loads((root / "task.toml").read_text())
    assert document["environment"] == {"network_mode": "allowlist", "allowed_hosts": ["pypi.org"]}
    assert read_harbor_task(root) == spec


# ----------------------------------------------------------------------------------------------- the consumer


def test_harbor_itself_loads_what_reef_wrote(tmp_path: Path) -> None:
    """The directory is read by Harbor's own task model, not only by reef's reader."""
    config_module = pytest.importorskip("harbor.models.task.config")
    paths_module = pytest.importorskip("harbor.models.task.paths")
    spec = task(
        config={
            "verifier": {"timeout_sec": 30},
            "agent": {"timeout_sec": 300},
            "environment": {"network_mode": "allowlist", "allowed_hosts": ["*.pythonhosted.org", "2001:db8::/32"]},
        }
    )
    root = write_harbor_task(spec, tmp_path)
    config = config_module.TaskConfig.model_validate_toml((root / "task.toml").read_text())
    assert config.schema_version == TASK_CONFIG_VERSION
    assert config.verifier.timeout_sec == 30.0
    assert config.agent.timeout_sec == 300.0
    assert config.environment.allowed_hosts == ["*.pythonhosted.org", "2001:db8::/32"]
    assert config.metadata["reef"]["digest"] == spec.digest
    paths = paths_module.TaskPaths(root)
    assert paths.instruction_path.is_file()
    assert paths.config_path.is_file()
    assert paths.environment_dir.is_dir()
    assert paths.test_path.is_file()


# ----------------------------------------------------------------------------------------------- round two


def test_a_prebuilt_image_task_with_no_environment_files_is_written_with_the_directory(tmp_path: Path) -> None:
    spec = task(environment={}, config={"environment": {"docker_image": "python:3.12-slim"}})
    root = write_harbor_task(spec, tmp_path)
    assert (root / "environment").is_dir() and files_of(root / "environment") == []
    assert read_harbor_task(root) == spec


@pytest.mark.parametrize("extra", ["junk/task.toml", "junk/deeper/instruction.md", "steps/one/instruction.md"])
def test_a_root_file_name_at_depth_is_still_an_extra_entry(tmp_path: Path, extra: str) -> None:
    root = write_harbor_task(task(), tmp_path)
    (root / extra).parent.mkdir(parents=True)
    (root / extra).write_text("x")
    with pytest.raises(HarborTaskError, match="entries reef did not write"):
        read_harbor_task(root)


def test_a_plain_file_named_solution_is_refused(tmp_path: Path) -> None:
    root = write_harbor_task(task(), tmp_path)
    (root / "solution").write_text("x")
    with pytest.raises(HarborTaskError, match="entries reef did not write"):
        read_harbor_task(root)


def test_entries_that_are_not_regular_files_are_refused(tmp_path: Path) -> None:
    root = write_harbor_task(task(), tmp_path)
    cases = {
        root / "gone": lambda p: p.symlink_to(tmp_path / "nowhere"),
        root / "tests" / "loop": lambda p: p.symlink_to(root),
        root / "tests" / "copy.sh": lambda p: p.symlink_to(root / "tests" / "test.sh"),
        root / "tests" / "pipe": lambda p: os.mkfifo(p),
    }
    for path, make in cases.items():
        make(path)
        with pytest.raises(HarborTaskError, match="not a regular file or directory"):
            read_harbor_task(root)
        path.unlink()
    assert read_harbor_task(root) == task()


def test_an_empty_foreign_directory_is_refused(tmp_path: Path) -> None:
    root = write_harbor_task(task(), tmp_path)
    (root / "tests" / "empty").mkdir()
    with pytest.raises(HarborTaskError, match="entries reef did not write: tests/empty"):
        read_harbor_task(root)


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads every directory")
def test_an_unreadable_directory_is_an_error_not_a_crash(tmp_path: Path) -> None:
    root = write_harbor_task(task(), tmp_path)
    (root / "tests").chmod(0)
    try:
        with pytest.raises(HarborTaskError, match="cannot be read"):
            read_harbor_task(root)
        with pytest.raises(HarborTaskConflict, match=r"cannot be read|not a task reef wrote"):
            write_harbor_task(task(), tmp_path)
    finally:
        (root / "tests").chmod(0o755)


@pytest.mark.parametrize(
    "overrides",
    [
        {"metadata": {"note": "bad \ud800"}},
        {"metadata": {"bad \udc00": 1}},
        {"metadata": {"items": ["ok", "bad \ud800"]}},
        {"config": {"verifier": {"env": {"K": "bad \ud800"}}}},
        {"config": {"verifier": {"env": {"bad \ud800": "v"}}}},
        {"config": {"environment": {"docker_image": "img \ud800"}}},
        {"source_agent_record_ids": ("rec \ud800",)},
        {"environment": {"Dockerfile": "FROM x\n", "name \ud800.txt": "x"}},
    ],
)
def test_a_lone_surrogate_anywhere_is_refused_at_construction(overrides: dict[str, object]) -> None:
    with pytest.raises(HarborTaskError, match="not valid Unicode"):
        task(**overrides)


@pytest.mark.parametrize(
    "environment",
    [
        {"Dockerfile": "FROM x\n", "A/b": "x", "a": "y"},
        {"Dockerfile": "FROM x\n", "a": "y", "A/b": "x"},
        {"Dockerfile": "FROM x\n", "caf\u00e9/b": "x", "cafe\u0301": "y"},
    ],
)
def test_a_file_that_folds_onto_a_directory_is_refused(environment: dict[str, str]) -> None:
    with pytest.raises(HarborTaskError, match="fold together"):
        task(environment=environment)


@pytest.mark.parametrize("name", ["x" * 256, "a/" + "y" * 256, "/".join(["d"] * 600)])
def test_a_name_longer_than_a_filesystem_allows_is_refused(name: str) -> None:
    with pytest.raises(HarborTaskError, match="longer than a filesystem allows"):
        task(environment={"Dockerfile": "FROM x\n", name: "x"})


def test_staging_lives_under_a_hidden_directory_that_never_looks_like_a_task(tmp_path: Path) -> None:
    import reef.core.tasks.harbor as module

    leftover = tmp_path / module.STAGING_DIRECTORY / "sum-391.deadbeef"
    leftover.mkdir(parents=True)
    module.write_task_files(task(), leftover)
    root = write_harbor_task(task(), tmp_path)
    # The leftover of a killed writer is left alone, out of the way of a root listing, and blocks nothing.
    assert sorted(p.name for p in tmp_path.iterdir()) == [module.STAGING_DIRECTORY, root.name]
    assert leftover.is_dir()
    assert read_harbor_task(root) == task()


def test_the_staging_directory_stays_so_writers_never_pull_it_from_under_each_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import reef.core.tasks.harbor as module

    write_harbor_task(task(), tmp_path)
    assert (tmp_path / module.STAGING_DIRECTORY).is_dir()
    # Another process removing the parent between two of our steps is survived: the leaf is made with parents.
    real_uuid4 = module.uuid.uuid4

    def uuid4_after_a_sweep() -> object:
        (tmp_path / module.STAGING_DIRECTORY).rmdir()
        return real_uuid4()

    monkeypatch.setattr(module.uuid, "uuid4", uuid4_after_a_sweep)
    assert write_harbor_task(task(name="sum-392"), tmp_path) == tmp_path / "sum-392"


def test_a_replay_and_a_conflict_create_no_staging_directory(tmp_path: Path) -> None:
    root = write_harbor_task(task(), tmp_path)
    (tmp_path / ".staging").rmdir()
    write_harbor_task(task(), tmp_path)
    with pytest.raises(HarborTaskConflict):
        write_harbor_task(task(instruction="Add 1 and 1."), tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == [root.name]


def test_a_task_directory_reads_as_itself_from_inside(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = write_harbor_task(task(), tmp_path)
    monkeypatch.chdir(root)
    assert read_harbor_task(Path(".")) == task()
    assert read_harbor_task(Path("../sum-391")) == task()
    assert read_harbor_task(Path("tests/..")) == task()
    hop = tmp_path / "hop"
    hop.symlink_to(root / "tests")
    assert read_harbor_task(hop / "..") == task()


def test_a_numeric_record_id_written_by_hand_is_refused(tmp_path: Path) -> None:
    root = write_harbor_task(task(source_agent_record_ids=("1",)), tmp_path)
    text = (root / "task.toml").read_text()
    (root / "task.toml").write_text(text.replace('"1",', "1,"))
    with pytest.raises(HarborTaskError, match="must hold strings"):
        read_harbor_task(root)


def test_an_absurd_timeout_is_refused_not_overflowed() -> None:
    with pytest.raises(HarborTaskError, match="positive number of seconds"):
        task(config={"agent": {"timeout_sec": 10**400}})


def test_a_dangling_symlink_at_the_target_is_a_conflict(tmp_path: Path) -> None:
    (tmp_path / "sum-391").symlink_to(tmp_path / "nowhere")
    with pytest.raises(HarborTaskConflict, match="not a directory"):
        write_harbor_task(task(), tmp_path)


def test_extra_keys_under_the_reef_table_are_refused(tmp_path: Path) -> None:
    root = write_harbor_task(task(), tmp_path)
    text = (root / "task.toml").read_text().replace("[metadata.reef]", '[metadata.reef]\nsigned = "yes"')
    (root / "task.toml").write_text(text)
    with pytest.raises(HarborTaskError, match="must hold exactly"):
        read_harbor_task(root)


@pytest.mark.parametrize("host", ["fe80::1%eth0", "fe80::1%25eth0", "fe80::%eth0/64", "::1%0"])
def test_ipv6_zone_ids_are_refused_like_harbor_does(host: str) -> None:
    with pytest.raises(HarborTaskError, match="must be a host name"):
        task(config={"environment": {"network_mode": "allowlist", "allowed_hosts": [host]}})


def test_the_user_env_and_workdir_keys_harbor_reads_round_trip(tmp_path: Path) -> None:
    spec = task(
        config={
            "verifier": {"timeout_sec": 30, "user": "root"},
            "agent": {"timeout_sec": 300, "user": 1000},
            "environment": {"env": {"HOME": "/workspace"}, "workdir": "/workspace"},
        }
    )
    root = write_harbor_task(spec, tmp_path)
    document = tomllib.loads((root / "task.toml").read_text())
    assert document["agent"] == {"timeout_sec": 300.0, "user": 1000}
    assert document["verifier"] == {"timeout_sec": 30.0, "user": "root"}
    assert document["environment"] == {"env": {"HOME": "/workspace"}, "workdir": "/workspace"}
    assert read_harbor_task(root) == spec


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"config": {"agent": {"user": ""}}}, "user name or a non-negative uid"),
        ({"config": {"agent": {"user": -1}}}, "user name or a non-negative uid"),
        ({"config": {"agent": {"user": True}}}, "user name or a non-negative uid"),
        ({"config": {"environment": {"workdir": ""}}}, "non-empty string"),
        ({"config": {"environment": {"env": {"": "x"}}}}, "variable names to strings"),
        ({"config": {"environment": {"os": "linux"}}}, "not a key Harbor reads"),
    ],
)
def test_the_new_keys_are_checked_the_way_harbor_checks_them(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(HarborTaskError, match=message):
        task(**overrides)


@pytest.mark.parametrize(
    "environment",
    [
        {"Dockerfile": "FROM x\n", "A/x": "1", "a/y": "2"},
        {"Dockerfile": "FROM x\n", "caf\u00e9/x": "1", "cafe\u0301/y": "2"},
        {"Dockerfile": "FROM x\n", "lib/a/x": "1", "LIB/b/y": "2"},
    ],
)
def test_directory_spellings_that_fold_together_are_refused(environment: dict[str, str]) -> None:
    with pytest.raises(HarborTaskError, match="fold together"):
        task(environment=environment)


def test_the_same_directory_spelled_the_same_way_twice_is_fine() -> None:
    spec = task(environment={"Dockerfile": "FROM x\n", "lib/a": "1", "lib/b": "2", "lib/sub/c": "3"})
    assert len(spec.environment) == 4


def test_a_volume_that_merges_names_is_reported_as_such(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import reef.core.tasks.harbor as module

    real_read = module.read_all_entries

    def merged(root: Path) -> tuple[dict[str, str], set[str]]:
        files, directories = real_read(root)
        files.pop("environment/Dockerfile", None)
        return files, directories

    monkeypatch.setattr(module, "read_all_entries", merged)
    with pytest.raises(HarborTaskError, match="is the volume case folding"):
        write_harbor_task(task(), tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == [".staging"]


def test_a_renamed_task_directory_is_refused_with_the_reason(tmp_path: Path) -> None:
    root = write_harbor_task(task(), tmp_path)
    moved = root.rename(tmp_path / "sum-392")
    with pytest.raises(HarborTaskError, match="edited, renamed or read through another name"):
        read_harbor_task(moved)


# ----------------------------------------------------------------------------------------------- imported tasks

#: Harbor's ProgramBench task template (harbor-framework/harbor#2295) filled in for one item.
PROGRAMBENCH_TASK_TOML = """schema_version = "1.3"
source = "ProgramBench"
artifacts = [
  "/logs/verifier/reward.json",
  "/logs/verifier/reward.txt",
  { source = "/workspace", exclude = ["executable"] },
]

[task]
name = "programbench/jq-1"
description = "ProgramBench cleanroom reconstruction task for jqlang/jq"
authors = [{ name = "Ada Example" }, { name = "Grace Example" }]
keywords = ["programbench", "cleanroom", "c"]

[metadata]
source = "ProgramBench"
instance_id = "jqlang__jq.1"
test_count = 12

[agent]
timeout_sec = 14400
network_mode = "allowlist"
allowed_hosts = ["api.anthropic.com", "api.openai.com"]

[verifier]
timeout_sec = 3600
user = "root"
environment_mode = "separate"

[verifier.env]
PROGRAMBENCH_SCRIPT_PTY = "1"

[verifier.environment]
build_timeout_sec = 1800
cpus = 2
memory_mb = 4096
storage_mb = 10240
gpus = 0
workdir = "/workspace"

[environment]
docker_image = "programbench/jqlang__jq.1:task_cleanroom_v6"
network_mode = "allowlist"
allowed_hosts = ["api.anthropic.com", "api.openai.com"]
build_timeout_sec = 1800
cpus = 2
memory_mb = 4096
storage_mb = 10240
gpus = 0
workdir = "/workspace"
"""


def programbench_task(root: Path, task_toml: str = PROGRAMBENCH_TASK_TOML) -> Path:
    source_path = root / "jq-1"
    files = {
        "task.toml": task_toml,
        "instruction.md": "Rebuild /workspace/executable from its behavior alone.\n",
        "environment/Dockerfile": "FROM programbench/jqlang__jq.1:task_cleanroom_v6\n",
        "tests/test.sh": "#!/bin/bash\npython3 /tests/programbench_evaluator.py\n",
        "tests/Dockerfile": "FROM programbench/jqlang__jq.1:task_cleanroom_v6\nRUN rm -rf /workspace\n",
        "tests/programbench_evaluator.py": "print('evaluated')\n",
        "solution/solve.sh": "#!/bin/bash\ncp /reference/executable /workspace/executable\n",
    }
    for relative, text in files.items():
        (source_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (source_path / relative).write_text(text)
    return source_path


def test_a_programbench_task_imports_with_every_file_and_every_key_harbor_reads(tmp_path: Path) -> None:
    source_path = programbench_task(tmp_path / "harbor")
    path = import_harbor_task(source_path, tmp_path / "tasks", parents=(task().digest,))
    assert path == tmp_path / "tasks" / "jq-1"
    assert files_of(path) == files_of(source_path)
    for relative in files_of(source_path):
        if relative != "task.toml":
            assert (path / relative).read_text() == (source_path / relative).read_text()
    imported = read_harbor_task(path)
    assert imported.is_imported and imported.parents == (task().digest,) and imported.source_agent_record_ids == ()
    original = tomllib.loads((source_path / "task.toml").read_text())
    copied = tomllib.loads((path / "task.toml").read_text())
    assert original.pop("schema_version") == "1.3" and copied.pop("version") == TASK_CONFIG_VERSION
    assert copied["metadata"].pop("reef") == {
        "digest": imported.digest,
        "source_agent_record_ids": [],
        "parents": [task().digest],
        "is_imported": True,
    }
    assert copied == original
    assert import_harbor_task(source_path, tmp_path / "tasks", parents=(task().digest,)) == path


def test_harbor_itself_loads_an_imported_task_with_its_separate_verifier(tmp_path: Path) -> None:
    config_module = pytest.importorskip("harbor.models.task.config")
    path = import_harbor_task(programbench_task(tmp_path / "harbor"), tmp_path / "tasks")
    config = config_module.TaskConfig.model_validate_toml((path / "task.toml").read_text())
    assert config.verifier.environment_mode.value == "separate"
    assert config.verifier.environment.workdir == "/workspace"
    assert config.verifier.env == {"PROGRAMBENCH_SCRIPT_PTY": "1"}
    assert config.agent.network_mode.value == "allowlist"
    assert config.agent.allowed_hosts == ["api.anthropic.com", "api.openai.com"]
    assert config.source == "ProgramBench" and config.task.name == "programbench/jq-1"
    assert config.artifacts[-1].source == "/workspace" and config.artifacts[-1].exclude == ["executable"]
    assert config.metadata["reef"]["digest"] == read_harbor_task(path).digest


@pytest.mark.parametrize(
    ("relative", "old", "new"),
    [
        ("tests/programbench_evaluator.py", "evaluated", "passed"),
        ("task.toml", 'environment_mode = "separate"', 'environment_mode = "shared"'),
        ("task.toml", '{ source = "/workspace", exclude = [', '{ source = "/workspace", exclude = [ "keep",'),
    ],
)
def test_an_edited_import_is_refused(tmp_path: Path, relative: str, old: str, new: str) -> None:
    path = import_harbor_task(programbench_task(tmp_path / "harbor"), tmp_path / "tasks")
    text = (path / relative).read_text()
    assert old in text
    (path / relative).write_text(text.replace(old, new))
    with pytest.raises(HarborTaskError, match="does not match its digest"):
        read_harbor_task(path)


TERMINAL_BENCH_TASK_TOML = """version = "1.0"

[task]
name = "terminal-bench/sampler"
description = ""

[metadata]
difficulty = "medium"

[agent]
timeout_sec = 900.0

[environment]
docker_image = "alexgshaw/adaptive-rejection-sampler:20251031"
cpus = 1
memory = "2G"
storage = "10G"
"""


def test_a_terminal_bench_task_with_the_older_size_keys_imports_as_harbor_reads_it(tmp_path: Path) -> None:
    source_path = programbench_task(tmp_path / "harbor", TERMINAL_BENCH_TASK_TOML)
    path = import_harbor_task(source_path, tmp_path / "tasks")
    copied = tomllib.loads((path / "task.toml").read_text())
    assert copied["environment"]["memory"] == "2G" and copied["environment"]["storage"] == "10G"
    assert copied["task"]["description"] == ""
    config_module = pytest.importorskip("harbor.models.task.config")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        config = config_module.TaskConfig.model_validate_toml((path / "task.toml").read_text())
    assert (config.environment.memory_mb, config.environment.storage_mb) == (2048, 10240)


@pytest.mark.parametrize("size", ['"2 gigs"', '"G"', '"nanG"', "2048", '"-1G"'])
def test_an_older_size_harbor_cannot_read_is_refused(tmp_path: Path, size: str) -> None:
    source_path = programbench_task(tmp_path / "harbor", TERMINAL_BENCH_TASK_TOML.replace('"2G"', size))
    with pytest.raises(HarborTaskError, match=r"environment\.memory must be a size Harbor reads"):
        import_harbor_task(source_path, tmp_path / "tasks")


def test_an_entry_outside_harbors_layout_is_refused_by_the_import(tmp_path: Path) -> None:
    source_path = programbench_task(tmp_path / "harbor")
    (source_path / "LICENSE").write_text("MIT\n")
    with pytest.raises(HarborTaskError, match="holds entries reef did not write: LICENSE"):
        import_harbor_task(source_path, tmp_path / "tasks")


def test_a_harbor_task_reef_did_not_write_is_refused_naming_the_importer(tmp_path: Path) -> None:
    with pytest.raises(HarborTaskError, match=r"no \[metadata\.reef\] table.*import_harbor_task"):
        read_harbor_task(programbench_task(tmp_path))


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            'schema_version = "1.3"\n',
            'schema_version = "1.3"\ndifficulty = 3\n',
            "Harbor 0.20.0 does not read: difficulty",
        ),
        ("[agent]\n", '[agent]\nmodel = "m"\n', r"config\.agent\.model is not a key Harbor reads"),
        (
            "[verifier.environment]\n",
            '[verifier.environment]\nshell = "sh"\n',
            r"verifier\.environment\.shell is not a key",
        ),
        ('environment_mode = "separate"', 'environment_mode = "apart"', "must be one of shared, separate"),
        ('schema_version = "1.3"\n', 'multi_step_reward_strategy = "max"\n', "must be one of mean, final"),
        ('source = "ProgramBench"\nartifacts', 'source = ""\nartifacts', "source must be a non-empty string"),
        ("[metadata]\n", '[metadata.reef]\ndigest = "x"\n\n[metadata]\n', "already a reef task"),
    ],
)
def test_an_import_harbor_would_not_read_is_refused(tmp_path: Path, old: str, new: str, message: str) -> None:
    assert old in PROGRAMBENCH_TASK_TOML
    source_path = programbench_task(tmp_path / "harbor", PROGRAMBENCH_TASK_TOML.replace(old, new, 1))
    with pytest.raises(HarborTaskError, match=message):
        import_harbor_task(source_path, tmp_path / "tasks")
    assert not (tmp_path / "tasks" / "jq-1").exists()


def test_parents_round_trip_and_a_task_without_them_keeps_the_digest_it_had(tmp_path: Path) -> None:
    parent = task()
    child = task(name="sum-392", parents=(parent.digest,))
    root = write_harbor_task(child, tmp_path)
    assert tomllib.loads((root / "task.toml").read_text())["metadata"]["reef"]["parents"] == [parent.digest]
    assert read_harbor_task(root) == child and child.digest != task(name="sum-392").digest
    assert "parents" not in tomllib.loads(write_harbor_task(parent, tmp_path).joinpath("task.toml").read_text())
    before = {
        "name": parent.name,
        "instruction": parent.instruction,
        "tests": dict(parent.tests),
        "environment": dict(parent.environment),
        "solution": dict(parent.solution),
        "config": {table: dict(values) for table, values in parent.config.items()},
        "metadata": dict(parent.metadata),
        "source_agent_record_ids": list(parent.source_agent_record_ids),
    }
    canonical = json.dumps(before, sort_keys=True, separators=(",", ":"))
    assert parent.digest == hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("is_imported = true", "does not match its digest"),
        ("is_imported = false", "must hold exactly"),
        ('parents = "x"', "must hold exactly"),
        ("parents = []\nsigned = true", "must hold exactly"),
    ],
)
def test_a_stamp_edited_into_a_reef_task_is_refused(tmp_path: Path, line: str, message: str) -> None:
    root = write_harbor_task(task(), tmp_path)
    text = (root / "task.toml").read_text().replace("[metadata.reef]\n", f"[metadata.reef]\n{line}\n")
    (root / "task.toml").write_text(text)
    with pytest.raises(HarborTaskError, match=message):
        read_harbor_task(root)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"parents": ("abc",)}, "parents must be a tuple of distinct task digests"),
        ({"parents": ("A" * 64,)}, "parents must be a tuple of distinct task digests"),
        ({"parents": ["a" * 64]}, "parents must be a tuple of distinct task digests"),
        ({"parents": ("a" * 64, "a" * 64)}, "parents must be a tuple of distinct task digests"),
        ({"is_imported": "yes"}, "is_imported must be a boolean"),
        ({"top_level_config": {"source": "x"}}, "imported task only"),
        ({"is_imported": True, "top_level_config": {"steps_dir": "x"}}, "not one Harbor reads"),
        ({"config": {"verifier": {"environment_mode": "separate"}}}, "not a key Harbor reads"),
        ({"is_imported": True, "config": {"verifier": {"environment": {"os": "mac"}}}}, "one of linux, windows"),
        ({"is_imported": True, "config": {"environment": {"allow_internet": "yes"}}}, "must be a boolean"),
        ({"is_imported": True, "config": {"agent": {"allowed_hosts": ["pypi.org"]}}}, "needs network_mode"),
        ({"is_imported": True, "config": {"task": {"keywords": [datetime.date(2026, 1, 1)]}}}, "strings, numbers"),
    ],
)
def test_the_stamp_and_the_harbor_keys_are_checked_at_construction(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(HarborTaskError, match=message):
        task(**overrides)
