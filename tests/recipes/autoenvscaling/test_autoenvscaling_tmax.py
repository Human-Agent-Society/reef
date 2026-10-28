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


def verifier_reward(tmp_path: Path, test_outputs: str | None, *, planted: bool = False) -> str:
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
    (bin_dir / "python3").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
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
