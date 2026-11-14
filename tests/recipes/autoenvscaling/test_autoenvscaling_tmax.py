"""The TMax converter and verifier: extracted TMax tasks become seed tasks that reef reads back.

The tasks are small stand-ins laid out as TMax-15K lays its tasks out; no test downloads anything or builds an image.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from recipes.beta.autoenvscaling import admitted_task, tmax, workspace_reply
from recipes.beta.autoenvscaling.generator import seed_tasks
from reef.core.tasks import read_harbor_task

CONTAINER_DEF = """Bootstrap: docker
From: python:3.12-slim

%files
    /home/tmax/build/task-7/data/config.ini /app/config.ini

%environment
    export APP_MODE=strict
    export PATH=/opt/tools:$PATH

%post
    mkdir -p /app/out
    echo ready > /app/out/state.txt

%labels
    Author tmax
"""
DESCRIPTION = "Read /app/config.ini and write the value of its mode key, and nothing else, to /app/out/mode.txt."
FINAL_STATE = "def test_mode():\n    assert open('/app/out/mode.txt').read().strip() == 'strict'\n"


def tmax_task(
    root: Path, task_id: str, *, container_def: str = CONTAINER_DEF, fixture: bytes = b"mode = strict\n"
) -> Path:
    """One TMax task directory as TMax-15K's tasks.zip holds it."""
    task_dir = root / task_id
    (task_dir / "data").mkdir(parents=True)
    (task_dir / "container.def").write_text(container_def)
    (task_dir / "data" / "config.ini").write_bytes(fixture)
    (task_dir / "task.json").write_text(json.dumps({"description": DESCRIPTION, "domain": "files", "tier": "easy"}))
    (task_dir / "test_final_state.py").write_text(FINAL_STATE)
    (task_dir / "test_initial_state.py").write_text("def test_start():\n    pass\n")
    return task_dir


V11_TOML = """schema_version = "1.1"

[task]
name = "tmax/{task_id}-v1.1"

[metadata]
task_id = "{task_id}"
domain = "data_querying"
skill_type = "Data Comprehension"
rubric_label = "{label}"
validation_status = "{status}"
image_rebuilt = {rebuilt}
known_verifier_defect = {defect}

[agent]
timeout_sec = 600.0
user = "root"

[verifier]
timeout_sec = 300.0
user = "root"

[environment]
workdir = "/app"
cpus = 1
memory_mb = 2048
allow_internet = true
"""
V11_DOCKERFILE = "FROM --platform=linux/amd64 hamishivi/swerl-tmax-hardened:task_000007_ab12cd34-0c4e\nWORKDIR /app\n"
V11_TEST = "#!/bin/bash\nif python3 -m pytest /tmp/t.py -x; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi\n"


def harbor_task(
    root: Path,
    task_id: str,
    *,
    status: str = "validated",
    defect: bool = False,
    label: str = "OK",
    rebuilt: bool = True,
    truth: bytes = b"[1, 2]\n",
) -> Path:
    """One TMax v1.1 task directory as TMaxxx/tmax-13k-v1.1-harbor's tasks.zip holds it."""
    task_dir = root / task_id
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "task.toml").write_text(
        V11_TOML.format(
            task_id=task_id, status=status, defect=str(defect).lower(), label=label, rebuilt=str(rebuilt).lower()
        )
    )
    (task_dir / "instruction.md").write_text(DESCRIPTION)
    (task_dir / "setup.sh").write_text("#!/bin/bash\necho baked into the image\n")
    (task_dir / "environment" / "Dockerfile").write_text(V11_DOCKERFILE)
    (task_dir / "environment" / "docker-compose.yaml").write_text("services:\n  main:\n    platform: linux/amd64\n")
    (task_dir / "tests" / "test.sh").write_text(V11_TEST)
    (task_dir / "tests" / ".truth.json").write_bytes(truth)
    return task_dir


@pytest.mark.unit
def test_a_tmax_v11_harbor_task_becomes_a_seed_that_keeps_its_verifier_and_settings(tmp_path: Path) -> None:
    harbor_task(tmp_path / "tasks", "task_000007_ab12cd34")
    arguments = ["--harbor", "--tasks", str(tmp_path / "tasks"), "--out", str(tmp_path / "seeds")]
    assert tmax.main(arguments) == 0

    task = read_harbor_task(tmp_path / "seeds" / "task-000007-ab12cd34")
    assert task.instruction == DESCRIPTION
    # The binary verifier and its hidden truth stay as v1.1 ships them.
    assert task.tests == {"test.sh": V11_TEST, ".truth.json": "[1, 2]\n"}
    assert task.solution == {"solve.sh": tmax.SOLVE_SCRIPT}
    assert task.environment["Dockerfile"].splitlines() == [*V11_DOCKERFILE.splitlines(), *tmax.HARBOR_TOOL_STEPS]
    assert "docker-compose.yaml" in task.environment
    assert task.metadata == {
        "tmax_task": "task_000007_ab12cd34",
        "tmax_release": "v1.1",
        "domain": "data_querying",
        "skill_type": "Data Comprehension",
    }
    # The task keeps the settings the v1.1 audit validated, in the names Harbor 0.23 reads.
    assert task.config["agent"]["timeout_sec"] == 600 and task.config["verifier"]["timeout_sec"] == 300
    assert task.config["environment"]["network_mode"] == "public"
    assert task.config["environment"]["memory_mb"] == 2048
    assert [seed.name for seed in seed_tasks(tmp_path / "seeds")] == ["task-000007-ab12cd34"]
    assert json.loads((tmp_path / "seeds" / "conversion.json").read_text()) == {
        "converted": [{"task_id": "task_000007_ab12cd34", "name": "task-000007-ab12cd34", "release": "v1.1"}],
        "refused": [],
    }


@pytest.mark.unit
def test_a_v11_task_the_audit_did_not_clear_or_with_a_binary_file_is_refused(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks"
    harbor_task(tasks, "task_000001_aaaaaaaa", status="not_validated")
    harbor_task(tasks, "task_000002_bbbbbbbb", defect=True)
    harbor_task(tasks, "task_000003_cccccccc", label="ANSWER-LEAK", rebuilt=False)
    harbor_task(tasks, "task_000004_dddddddd", truth=b"\xff\xfe binary oracle")
    harbor_task(tasks, "task_000005_eeeeeeee", label="ANSWER-LEAK")
    assert tmax.main(["--harbor", "--tasks", str(tasks), "--out", str(tmp_path / "seeds")]) == 1

    result = json.loads((tmp_path / "seeds" / "conversion.json").read_text())
    reasons = {entry["task_id"]: entry["reason"] for entry in result["refused"]}
    assert reasons["task_000001_aaaaaaaa"].startswith("the v1.1 audit did not validate the task")
    assert reasons["task_000002_bbbbbbbb"] == "the v1.1 audit found a defect in its verifier"
    assert reasons["task_000003_cccccccc"] == "the task leaks its answer, and v1.1 kept its original image"
    assert "binary file" in reasons["task_000004_dddddddd"]
    # A leak that v1.1 repaired with a rebuilt image is a seed.
    assert [entry["task_id"] for entry in result["converted"]] == ["task_000005_eeeeeeee"]


@pytest.mark.unit
def test_a_tmax_task_becomes_a_reef_harbor_task_that_reads_back(tmp_path: Path) -> None:
    tmax_task(tmp_path / "tasks", "Task_007")
    assert tmax.main(["--tasks", str(tmp_path / "tasks"), "--out", str(tmp_path / "seeds")]) == 0

    task = read_harbor_task(tmp_path / "seeds" / "task-007")
    assert task.environment["Dockerfile"].splitlines() == [
        "FROM python:3.12-slim",
        "ENV APP_MODE=strict",
        "ENV PATH=/opt/tools:$PATH",
        "COPY fixtures/data/config.ini /app/config.ini",
        "COPY tmax_build.sh /tmp/tmax_build.sh",
        "RUN bash /tmp/tmax_build.sh && rm -f /tmp/tmax_build.sh",
        *tmax.TOOL_STEPS,
        "WORKDIR /app",
    ]
    assert task.environment["fixtures/data/config.ini"] == "mode = strict\n"
    # Neither the agent's container nor the verifier has network: the image installs tmux and pytest.
    assert "apt-get install -y tmux" in task.environment["Dockerfile"]
    assert "python3 -m pip install -q pytest" in task.environment["Dockerfile"]
    assert task.environment["tmax_build.sh"] == (
        "#!/bin/bash\nset -e\n    mkdir -p /app/out\n    echo ready > /app/out/state.txt\n\n"
    )
    assert task.instruction == DESCRIPTION
    assert task.tests == {"test.sh": tmax.VERIFIER.read_text(), "test_outputs.py": FINAL_STATE}
    assert task.solution == {"solve.sh": tmax.SOLVE_SCRIPT}
    assert task.metadata == {"tmax_task": "Task_007", "domain": "files", "tier": "easy"}
    assert task.config["environment"]["network_mode"] == "no-network"
    # The initial-state check is not used.
    assert not any("initial" in name for name in task.files)
    assert json.loads((tmp_path / "seeds" / "conversion.json").read_text()) == {
        "converted": [
            {
                "task_id": "Task_007",
                "name": "task-007",
                "image": "python:3.12-slim",
                "fixtures": 1,
                "unhandled_sections": ["labels"],
            }
        ],
        "refused": [],
    }
    # The generator reads the directory as its seeds: the conversion's file and the writer's staging are no seeds.
    assert [seed.name for seed in seed_tasks(tmp_path / "seeds")] == ["task-007"]
    # A second conversion keeps the task written before.
    assert tmax.main(["--tasks", str(tmp_path / "tasks"), "--out", str(tmp_path / "seeds")]) == 0


@pytest.mark.unit
def test_a_tmax_task_with_a_binary_fixture_or_no_from_image_is_reported_not_converted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks = tmp_path / "tasks"
    tmax_task(tasks, "good")
    tmax_task(tasks, "binary", fixture=b"\x89PNG\r\n\x1a\n\xff\x00")
    tmax_task(tasks, "imageless", container_def=CONTAINER_DEF.replace("From: python:3.12-slim\n", ""))
    ambiguous = tmax_task(tasks, "ambiguous", container_def=CONTAINER_DEF.replace("task-7/data/", ""))
    (ambiguous / "other").mkdir()
    (ambiguous / "other" / "config.ini").write_text("mode = loose\n")
    fixture_entry = "/home/tmax/build/task-7/data/config.ini /app/config.ini"
    entries = "/x/data/a+b.csv /app/1.csv\n    /x/data/a=b.csv /app/2.csv"
    collision = tmax_task(tasks, "collision", container_def=CONTAINER_DEF.replace(fixture_entry, entries))
    for name in ("a+b.csv", "a=b.csv"):
        (collision / "data" / name).write_text(f"{name}\n")
    # A file name longer than a reply's file name may be: the task writer takes it, a variant's reply would not.
    long_name = "f" * 116 + ".txt"
    long = tmax_task(tasks, "long", container_def=CONTAINER_DEF.replace("data/config.ini /", f"data/{long_name} /"))
    (long / "data" / long_name).write_text("mode = strict\n")
    task_ids = ["binary", "good", "imageless", "ambiguous", "collision", "long", "absent"]
    (tmp_path / "ids.json").write_text(json.dumps(task_ids))
    exit_code = tmax.main(
        ["--tasks", str(tasks), "--out", str(tmp_path / "seeds"), "--select", str(tmp_path / "ids.json")]
    )
    assert exit_code == 1
    result = json.loads((tmp_path / "seeds" / "conversion.json").read_text())
    assert [entry["task_id"] for entry in result["converted"]] == ["good"]
    reasons = {entry["task_id"]: entry["reason"] for entry in result["refused"]}
    assert reasons == {
        "binary": "config.ini is a binary file; a reef task holds text files only",
        "imageless": "container.def has no From: image",
        "ambiguous": (
            "the fixture /home/tmax/build/config.ini matches several files in the task directory: "
            "data/config.ini, other/config.ini"
        ),
        "collision": "the fixtures data/a+b.csv and data/a=b.csv share the name fixtures/data/a_b.csv",
        "long": (
            "a variant could not keep the environment: the reply's environment names a file the task cannot hold: "
            f"'fixtures/data/{long_name}'"
        ),
        "absent": "no task directory has this id",
    }
    assert sorted(path.name for path in (tmp_path / "seeds").iterdir() if not path.name.startswith(".")) == [
        "conversion.json",
        "good",
    ]
    assert "converted 1 of 7 tasks" in capsys.readouterr().out


SHARED_NAMES_DEF = """Bootstrap: docker
From: ubuntu:22.04

%files
    /x/a/config.json /etc/app/a.json
    /x/b/config.json /etc/app/b.json
    /x/sales+2024.csv /home/user/sales+2024.csv

%post
    true
"""


@pytest.mark.unit
def test_each_fixture_keeps_its_own_file_under_a_name_a_variant_can_hold(tmp_path: Path) -> None:
    task_dir = tmax_task(tmp_path / "tasks", "t1", container_def=SHARED_NAMES_DEF)
    for folder, text in (("a", '{"a": 1}\n'), ("b", '{"b": 2}\n')):
        (task_dir / folder).mkdir()
        (task_dir / folder / "config.json").write_text(text)
    (task_dir / "sales+2024.csv").write_text("month,total\n")
    # Upstream's selection file: objects with a task_id.
    (tmp_path / "selection.json").write_text(json.dumps([{"task_id": "t1", "domain": "data"}]))
    arguments = ["--tasks", str(tmp_path / "tasks"), "--out", str(tmp_path / "seeds")]
    assert tmax.main([*arguments, "--select", str(tmp_path / "selection.json")]) == 0

    task = read_harbor_task(tmp_path / "seeds" / "t1")
    assert [line for line in task.environment["Dockerfile"].splitlines() if line.startswith("COPY fixtures/")] == [
        "COPY fixtures/a/config.json /etc/app/a.json",
        "COPY fixtures/b/config.json /etc/app/b.json",
        "COPY fixtures/sales_2024.csv /home/user/sales+2024.csv",
    ]
    assert (task.environment["fixtures/a/config.json"], task.environment["fixtures/b/config.json"]) == (
        '{"a": 1}\n',
        '{"b": 2}\n',
    )
    # A variant's workspace starts with the seed's environment, and admission takes every file of it.
    admitted = admitted_task(
        workspace_reply(tmp_path / "seeds" / "t1"), generation=0, index=0, record_id="seed-0", checkout_path=None
    )
    assert admitted.environment == task.environment


def verifier_reward(
    tmp_path: Path, test_outputs: str | None, *, planted: bool = False, checks_config: bool = False
) -> str:
    """The reward the verifier writes for ``test_outputs`` (no file for None), under a ``python3`` with pytest."""
    tests = tmp_path / "tests"
    logs = tmp_path / "logs"
    tests.mkdir(parents=True)
    logs.mkdir()
    if test_outputs is not None:
        (tests / "test_outputs.py").write_text(test_outputs)
    # A config file and a conftest.py above the tests that would pass every test: the verifier never reads them.
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "conftest.py").write_text(
        "import pytest\n\n\n@pytest.hookimpl(hookwrapper=True)\ndef pytest_runtest_makereport(item, call):\n"
        "    outcome = yield\n    outcome.get_result().outcome = 'passed'\n"
    )
    if planted:
        (logs / "reward.txt").write_text("1\n")
        (logs / "reward.json").write_text('{"reward": 1}')
        (logs / "junit.xml").write_text('<testsuite tests="1" failures="0" errors="0" skipped="0"/>')
    # The test's own interpreter, which has pytest, as the verifier's python3.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # With checks_config, python3 refuses a pytest -c file without a supported extension, as newer pytest does.
    check = (
        'prev=""; for arg in "$@"; do if [ "$prev" = "-c" ]; then case "$arg" in '
        '*.ini|*.cfg|*.toml) ;; *) echo "unsupported config $arg" >&2; exit 4;; esac; fi; prev="$arg"; done\n'
        if checks_config
        else ""
    )
    (bin_dir / "python3").write_text(f'#!/bin/sh\n{check}exec "{sys.executable}" "$@"\n')
    (bin_dir / "python3").chmod(0o755)
    environ = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "TMAX_TESTS_DIR": str(tests),
        "TMAX_VERIFIER_DIR": str(logs),
    }
    subprocess.run(["bash", str(tmax.VERIFIER)], env=environ, check=True, capture_output=True, timeout=120)
    assert not (logs / "reward.json").exists(), "a planted reward.json is removed"
    return (logs / "reward.txt").read_text()


TEST_OUTPUTS = """import pytest


def test_one():
    pass


def test_two():
    pass


def test_three():
    assert False


@pytest.mark.skip(reason="never counts as passed")
def test_four():
    pass
"""


@pytest.mark.unit
@pytest.mark.parametrize(
    ("test_outputs", "reward"),
    [
        (TEST_OUTPUTS, "0.5000\n"),
        ("def test_one():\n    pass\n", "1.0000\n"),
        ("def test_one(:\n", "0.0000\n"),
        ("", "0.0000\n"),
        (None, "0.0000\n"),
    ],
    ids=("two-of-four", "all", "unimportable", "no-tests", "no-file"),
)
def test_the_tmax_verifier_scores_the_fraction_passed_and_fails_closed(
    tmp_path: Path, test_outputs: str | None, reward: str
) -> None:
    assert verifier_reward(tmp_path, test_outputs, planted=True) == reward


@pytest.mark.unit
def test_the_tmax_verifier_gives_pytest_a_config_file_with_an_extension(tmp_path: Path) -> None:
    # Newer pytest refuses an extensionless -c file, and every task would then score 0.
    assert verifier_reward(tmp_path, "def test_one():\n    pass\n", checks_config=True) == "1.0000\n"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("files", "unhandled", "message"),
    [
        ((("fixture.txt", ""),), (), "a %files entry is a source and a destination path"),
        ((), ("labels", "post"), "the conversion reads the sections"),
    ],
    ids=("empty-destination", "handled-section"),
)
def test_a_container_definition_refuses_a_malformed_value(
    files: tuple[tuple[str, str], ...], unhandled: tuple[str, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        tmax.ContainerDefinition("ubuntu:24.04", files, (), (), unhandled)
