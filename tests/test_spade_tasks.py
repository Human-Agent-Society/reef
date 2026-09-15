"""A generated SPADE environment as a Harbor task: the observe and act commands, the verifier, the metadata, the split."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from recipes.beta.spade import GeneratedEnvironment, environment_loader, environment_task, split_generation
from recipes.beta.spade.environment_loader import write_actions
from reef.core.tasks import read_harbor_task, write_harbor_task

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

WORDLE = """import random
import re


class WordleEnv:
    WORDS = ["spade", "trace", "lemon", "graph"]

    def reset(self, seed=None):
        self.target = random.Random(seed).choice(self.WORDS)
        self.turns_left = 6
        return "Guess a 5-letter word in 6 tries. Answer with \\\\boxed{word}.", {}

    def step(self, action):
        match = re.search(r"\\\\boxed\\{([^}]*)\\}", action)
        if not match:
            return "Use \\\\boxed{word}.", 0.0, False, False, {}
        guess = match.group(1).strip()
        self.turns_left -= 1
        if guess == self.target:
            return "GGGGG", 1.0, True, False, {}
        return "-----", 0.0, False, self.turns_left == 0, {}
"""


def environment(**overrides: object) -> GeneratedEnvironment:
    fields: dict[str, object] = {
        "code": WORDLE,
        "skill": "deduction",
        "generation": 12,
        "index": 3,
        "hint": "The target is one of four words; start with a word that shares letters with all of them.",
        "source_record_id": "rec-designer-12",
        "step": 48,
        "difficulty": "medium",
        "document_id": "dclm-000123",
        "seed": 7,
    }
    fields.update(overrides)
    return GeneratedEnvironment(**fields)  # type: ignore[arg-type]


def target_for(seed: int) -> str:
    import random

    return random.Random(seed).choice(["spade", "trace", "lemon", "graph"])


def interact(root: Path, work: Path, *arguments: str) -> str:
    """Run the container's observe (no arguments) or act command on the host, its files and log mapped into ``work``."""
    directory = work / "env"
    if not directory.exists():
        directory.mkdir()
        for name in ("env.py", "env_loader.py", "interact.py"):
            (directory / name).write_text((root / "environment" / name).read_text())
        dockerfile = (root / "environment" / "Dockerfile").read_text()
        config = dockerfile.split("printf '")[1].split("\\n'")[0]
        (directory / "config.json").write_text(config)
    completed = subprocess.run(
        [sys.executable, "-S", str(directory / "interact.py"), *arguments],
        env={**os.environ, "ENVIRONMENT_DIRECTORY": str(directory), "ENVIRONMENT_LOG": str(work / "actions.jsonl")},
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return completed.stdout


def verifier_return(root: Path, log_path: Path) -> float:
    """Run tests/test.sh's command line with the container paths mapped into ``root``."""
    command = (root / "tests" / "test.sh").read_text().splitlines()[-1].split()
    reward_path = log_path.parent / "reward.txt"
    subprocess.run(
        [
            sys.executable,
            "-S",
            str(root / "tests" / "replay.py"),
            str(root / "tests" / "env.py"),
            str(log_path),
            str(reward_path),
            command[-2],
            command[-1],
        ],
        check=True,
        timeout=60,
    )
    return float(reward_path.read_text())


# ----------------------------------------------------------------------------------------------- the task


def test_the_task_puts_the_environment_behind_root_only_commands_and_the_class_with_the_verifier(
    tmp_path: Path,
) -> None:
    task = environment_task(environment())
    assert task.name == "gym-00012-003-deduction"
    root = write_harbor_task(task, tmp_path)
    loader = inspect.getsource(environment_loader)
    assert sorted(p.name for p in (root / "environment").iterdir()) == [
        "Dockerfile",
        "act",
        "env.py",
        "env_loader.py",
        "interact.py",
        "observe",
    ]
    assert (root / "environment" / "env.py").read_text() == WORDLE == (root / "tests" / "env.py").read_text()
    assert (
        (root / "environment" / "env_loader.py").read_text()
        == loader
        == (root / "tests" / "env_loader.py").read_text()
    )
    dockerfile = (root / "environment" / "Dockerfile").read_text()
    assert dockerfile.startswith("FROM python:3.12-slim\n")
    assert "useradd --create-home --shell /bin/bash agent" in dockerfile
    assert "chmod 700 /opt/env /var/env" in dockerfile and "chmod 600 /opt/env/*" in dockerfile
    assert (
        "agent ALL=(root) NOPASSWD: /usr/local/bin/python3 -S /opt/env/interact.py, "
        "/usr/local/bin/python3 -S /opt/env/interact.py *" in dockerfile
    )
    assert '{"seed": 7, "max_turns": 12}' in dockerfile
    assert (
        root / "environment" / "observe"
    ).read_text() == "#!/bin/sh\nexec sudo -n /usr/local/bin/python3 -S /opt/env/interact.py\n"
    act = (root / "environment" / "act").read_text()
    assert act.startswith("#!/bin/sh\n") and act.endswith(
        'exec sudo -n /usr/local/bin/python3 -S /opt/env/interact.py "$@"\n'
    )
    assert "usage: act" in act
    assert (
        "/tests/replay.py /tests/env.py /var/env/actions.jsonl /logs/verifier/reward.txt 12 7"
        in (root / "tests" / "test.sh").read_text()
    )
    instruction = (root / "instruction.md").read_text()
    assert "act '\\boxed{north}'" in instruction and "after 12 turns" in instruction and "env.py" not in instruction
    assert "Run `observe`" in instruction
    assert (root / "solution" / "hint.txt").read_text().startswith("The target is one of four words")
    document = tomllib.loads((root / "task.toml").read_text())
    assert document["agent"] == {"timeout_sec": 900, "user": "agent"}
    assert document["environment"]["network_mode"] == "no-network"
    assert document["metadata"] == {
        "kind": "gym",
        "skill": "deduction",
        "generation": 12,
        "step": 48,
        "index": 3,
        "env_class": "WordleEnv",
        "max_turns": 12,
        "seed": 7,
        "difficulty": "medium",
        "document": "dclm-000123",
        "reef": {"digest": task.digest, "source_agent_record_ids": ["rec-designer-12"]},
    }
    assert read_harbor_task(root) == task


def test_optional_fields_are_left_out_of_the_metadata_when_unset() -> None:
    task = environment_task(environment(difficulty=None, document_id=None), max_turns=20)
    assert "difficulty" not in task.metadata and "document" not in task.metadata
    assert task.metadata["max_turns"] == 20 and "after 20 turns" in task.instruction


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"code": "  "}, "non-empty text"),
        ({"code": "x = 1\n"}, "no top level class ending in 'Env'"),
        ({"code": "import numpy\nclass AEnv:\n    pass\n"}, "only the standard library"),
        ({"skill": "Deduction"}, "skill"),
        ({"skill": "a/b"}, "skill"),
        ({"generation": -1}, "generation must be"),
        ({"index": True}, "index must be"),
        ({"hint": ""}, "hint must be"),
        ({"source_record_id": ""}, "source_record_id"),
        ({"difficulty": ""}, "difficulty"),
    ],
)
def test_a_bad_environment_is_refused(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        environment_task(environment(**overrides))


def test_a_bad_turn_limit_is_refused() -> None:
    with pytest.raises(ValueError, match="max_turns"):
        environment_task(environment(), max_turns=0)


# ----------------------------------------------------------------------------------------------- observe and act


def test_observe_and_act_run_the_environment_and_the_verifier_agrees(tmp_path: Path) -> None:
    root = write_harbor_task(environment_task(environment(seed=7)), tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    target = target_for(7)
    assert interact(root, work) == "Guess a 5-letter word in 6 tries. Answer with \\boxed{word}.\nTurns left: 12\n"
    assert interact(root, work, "\\boxed{lemon}") in (
        "-----\nTurns left: 11\n",
        "GGGGG\nThe episode is over. Return: 1.0\n",
    )
    if target == "lemon":
        return
    assert interact(root, work, "I think", "\\boxed{" + target + "}") == "GGGGG\nThe episode is over. Return: 1.0\n"
    assert interact(root, work) == "GGGGG\nThe episode is over. Return: 1.0\n"
    assert interact(root, work, "\\boxed{graph}") == "The episode is over; no more actions are taken.\n"
    log_path = work / "actions.jsonl"
    assert [json.loads(line) for line in log_path.read_text().splitlines()] == [
        "\\boxed{lemon}",
        "I think \\boxed{" + target + "}",
    ]
    assert verifier_return(root, log_path) == 1.0


def test_act_stops_at_the_turn_limit_and_the_verifier_agrees(tmp_path: Path) -> None:
    root = write_harbor_task(environment_task(environment(seed=7), max_turns=2), tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    interact(root, work, "\\boxed{zzzzz}")
    assert interact(root, work, "\\boxed{zzzzz}") == "-----\nTurns left: 0\n"
    assert (
        interact(root, work, "\\boxed{" + target_for(7) + "}") == "The episode is over; no more actions are taken.\n"
    )
    assert interact(root, work) == "-----\nThe episode is over. Return: 0.0\n"
    assert verifier_return(root, work / "actions.jsonl") == 0.0


def test_a_reply_without_a_box_is_a_reminder_and_still_a_turn(tmp_path: Path) -> None:
    root = write_harbor_task(environment_task(environment(seed=7), max_turns=3), tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    assert interact(root, work, "lemon") == "Use \\boxed{word}.\nTurns left: 2\n"
    assert verifier_return(root, work / "actions.jsonl") == 0.0


def test_the_verifier_writes_a_reward_even_when_the_environment_breaks(tmp_path: Path) -> None:
    code = "class AEnv:\n    def reset(self, seed=None):\n        return 'o', {}\n    def step(self, action):\n        raise KeyError(action)\n"
    root = write_harbor_task(environment_task(environment(code=code)), tmp_path)
    log_path = tmp_path / "log" / "actions.jsonl"
    log_path.parent.mkdir()
    write_actions(str(log_path), ["\\boxed{go}"])
    assert verifier_return(root, log_path) == -1.0


def test_the_shipped_scripts_need_only_the_standard_library() -> None:
    import ast

    task = environment_task(environment())
    modules: set[str] = set()
    for text in (task.tests["replay.py"], task.tests["env_loader.py"], task.environment["interact.py"]):
        tree = ast.parse(text)
        modules |= {
            alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
        }
        modules |= {
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0
        }
    assert modules <= {
        "__future__",
        "ast",
        "collections",
        "copy",
        "env_loader",
        "functools",
        "heapq",
        "importlib",
        "inspect",
        "itertools",
        "json",
        "math",
        "operator",
        "os",
        "random",
        "re",
        "statistics",
        "string",
        "sys",
        "typing",
    }


# ----------------------------------------------------------------------------------------------- the split


def test_environments_of_one_designer_call_stay_in_one_split() -> None:
    tasks = [
        environment_task(environment(index=index, source_record_id=f"rec-designer-{index // 2}")) for index in range(8)
    ]
    for seed in range(10):
        split = split_generation(tasks, eval_fraction=0.5, seed=seed)
        for index in range(0, 8, 2):
            first, second = f"gym-00012-{index:03d}-deduction", f"gym-00012-{index + 1:03d}-deduction"
            assert (first in split.eval) == (second in split.eval), (seed, split)
        assert len(split.eval) >= 4


def test_two_tasks_with_one_name_are_refused_instead_of_dropped() -> None:
    first = environment_task(environment())
    second = environment_task(environment(code=WORDLE + "#\n", source_record_id="rec-other"))
    with pytest.raises(ValueError, match="gym-00012-003-deduction"):
        split_generation([first, second], eval_fraction=0.5, seed=0)


# ----------------------------------------------------------------------------------------------- the consumer


def test_harbor_itself_loads_a_generated_environment_task(tmp_path: Path) -> None:
    config_module = pytest.importorskip("harbor.models.task.config")
    paths_module = pytest.importorskip("harbor.models.task.paths")
    root = write_harbor_task(environment_task(environment()), tmp_path)
    config = config_module.TaskConfig.model_validate_toml((root / "task.toml").read_text())
    assert config.environment.network_mode.value == "no-network" and config.agent.user == "agent"
    assert config.metadata["env_class"] == "WordleEnv"
    paths = paths_module.TaskPaths(root)
    assert paths.test_path.is_file() and paths.environment_dir.is_dir()
