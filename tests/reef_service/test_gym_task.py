"""A Gym style environment as a Harbor task: written, read back as a gym task, and replayed by its verifier."""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import pytest

from reef.core.tasks import (
    GymTaskError,
    HarborTask,
    gym_loader,
    gym_task,
    read_gym_task,
    read_harbor_task,
    write_harbor_task,
)
from reef.core.tasks.gym_loader import write_actions

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
        left = [t for g, t in zip(guess, self.target) if g != t]
        fb = ""
        for g, t in zip(guess, self.target):
            if g == t:
                fb += "G"
            elif g in left:
                fb += "Y"
                left.remove(g)
            else:
                fb += "-"
        if fb == "GGGGG":
            return fb, 1.0, True, False, {}
        return fb, 0.0, False, self.turns_left == 0, {}
"""


def task(**overrides: object) -> HarborTask:
    fields: dict[str, object] = {
        "name": "wordle-007",
        "code": WORDLE,
        "seed": 7,
        "metadata": {"skill": "deduction"},
        "solution": {"hint.txt": "Start with a word that shares letters with all four.\n"},
        "source_agent_record_ids": ("rec-1",),
    }
    fields.update(overrides)
    return gym_task(**fields)  # type: ignore[arg-type]


def target_for(seed: int) -> str:
    import random

    return random.Random(seed).choice(["spade", "trace", "lemon", "graph"])


def replay(root: Path, actions: list[str] | bytes) -> tuple[int, float | None, str]:
    """Run tests/test.sh's command line with the container paths mapped into ``root``; (exit, reward, stderr)."""
    work = root / "work"
    work.mkdir(exist_ok=True)
    if isinstance(actions, bytes):
        (work / "actions.jsonl").write_bytes(actions)
    else:
        write_actions(str(work / "actions.jsonl"), actions)
    command = (root / "tests" / "test.sh").read_text().splitlines()[-1].split()
    max_turns, seed = command[-2], command[-1]
    reward = work / "reward.txt"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "tests" / "replay.py"),
            str(root / "tests" / "env.py"),
            str(work / "actions.jsonl"),
            str(reward),
            max_turns,
            seed,
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    value = float(reward.read_text()) if reward.exists() else None
    return completed.returncode, value, completed.stderr


# ----------------------------------------------------------------------------------------------- the task


def test_the_task_keeps_the_code_with_the_verifier_and_away_from_the_agent(tmp_path: Path) -> None:
    written = task()
    root = write_harbor_task(written, tmp_path)
    assert sorted(p.name for p in (root / "environment").iterdir()) == ["Dockerfile"]
    assert (root / "environment" / "Dockerfile").read_text() == "FROM python:3.12-slim\nWORKDIR /workspace\n"
    assert (root / "tests" / "env.py").read_text() == WORDLE
    assert (root / "tests" / "env_loader.py").read_text() == inspect.getsource(gym_loader)
    assert (root / "solution" / "hint.txt").read_text().startswith("Start with a word")
    instruction = (root / "instruction.md").read_text()
    assert "one turn at a time" in instruction and "after 12 turns" in instruction
    assert "never see the game's code" in instruction and "env.py" not in instruction
    assert "/tests/replay.py /tests/env.py /workspace/actions.jsonl" in (root / "tests" / "test.sh").read_text()
    document = tomllib.loads((root / "task.toml").read_text())
    assert document["metadata"] == {
        "skill": "deduction",
        "env_class": "WordleEnv",
        "max_turns": 12,
        "seed": 7,
        "reef": {"digest": written.digest, "source_agent_record_ids": ["rec-1"]},
    }
    assert document["environment"]["network_mode"] == "no-network"
    assert read_harbor_task(root) == written


def test_the_task_reads_back_as_a_gym_task(tmp_path: Path) -> None:
    root = write_harbor_task(task(max_turns=20), tmp_path)
    gym = read_gym_task(root)
    assert gym.code == WORDLE and gym.max_turns == 20 and gym.seed == 7
    assert gym.task.name == "wordle-007"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"code": "  "}, "non-empty text"),
        ({"code": "x = 1\n"}, "no top level class ending in 'Env'"),
        ({"code": "class TicketEnv(ToolUseBaseEnv):\n    pass\n"}, "ToolUseBaseEnv"),
        ({"code": "def broken(:\n"}, "not valid Python"),
        ({"max_turns": 0}, "max_turns"),
        ({"max_turns": True}, "max_turns"),
        ({"seed": -1}, "seed"),
        ({"metadata": {"seed": 3}}, "metadata carries 'seed'"),
        ({"metadata": {"env_class": "X"}}, "metadata carries 'env_class'"),
    ],
)
def test_a_bad_environment_is_refused(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        task(**overrides)


def test_a_harbor_task_that_is_not_a_gym_task_is_refused(tmp_path: Path) -> None:
    plain = HarborTask(
        name="plain",
        instruction="do it\n",
        tests={"test.sh": "#!/bin/sh\nexit 0\n"},
        environment={"Dockerfile": "FROM python:3.12-slim\n"},
        config={},
        metadata={"max_turns": 12, "seed": 0},
        solution={},
        source_agent_record_ids=(),
    )
    root = write_harbor_task(plain, tmp_path)
    with pytest.raises(GymTaskError, match=r"tests/env\.py is missing"):
        read_gym_task(root)


def test_a_gym_task_written_with_another_loader_is_refused(tmp_path: Path) -> None:
    written = task()
    other = HarborTask(
        name=written.name,
        instruction=written.instruction,
        tests={**written.tests, "env_loader.py": "# an older loader\n"},
        environment=written.environment,
        config=written.config,
        metadata=written.metadata,
        solution=written.solution,
        source_agent_record_ids=written.source_agent_record_ids,
    )
    root = write_harbor_task(other, tmp_path)
    with pytest.raises(GymTaskError, match="another loader"):
        read_gym_task(root)


# ----------------------------------------------------------------------------------------------- the verifier


def test_the_verifier_scores_a_win_a_loss_and_a_walkover(tmp_path: Path) -> None:
    root = write_harbor_task(task(), tmp_path)
    target = target_for(7)
    assert replay(root, ["\\boxed{lemon}", f"reasoning then \\boxed{{{target}}}"])[:2] == (0, 1.0)
    assert replay(root, ["\\boxed{zzzzz}"] * 6)[:2] == (0, 0.0)
    assert replay(root, [])[:2] == (0, 0.0)
    assert replay(root, [target])[:2] == (0, 0.0)


def test_the_verifier_replays_a_reply_that_spans_lines_as_written(tmp_path: Path) -> None:
    root = write_harbor_task(task(), tmp_path)
    assert replay(root, [f"Let me think.\n\\boxed{{{target_for(7)}}}"])[:2] == (0, 1.0)


def test_the_verifier_stops_at_the_turn_limit(tmp_path: Path) -> None:
    root = write_harbor_task(task(max_turns=2), tmp_path)
    target = target_for(7)
    assert replay(root, ["\\boxed{zzzzz}", "\\boxed{zzzzz}", f"\\boxed{{{target}}}"])[:2] == (0, 0.0)


def test_the_verifier_writes_a_reward_even_when_the_game_breaks(tmp_path: Path) -> None:
    code = "class AEnv:\n    def reset(self, seed=None):\n        return 'o', {}\n    def step(self, action):\n        raise KeyError(action)\n"
    root = write_harbor_task(task(code=code), tmp_path)
    assert replay(root, ["\\boxed{go}"])[:2] == (0, -1.0)
    broken_reset = "class AEnv:\n    def reset(self, seed=None):\n        raise RuntimeError('no game')\n"
    root = write_harbor_task(task(name="broken", code=broken_reset), tmp_path)
    exit_code, value, stderr = replay(root, ["\\boxed{go}"])
    assert exit_code == 1 and value == 0.0 and "no game" in stderr


def test_the_verifier_tolerates_an_action_log_that_is_not_json_lines(tmp_path: Path) -> None:
    root = write_harbor_task(task(), tmp_path)
    assert replay(root, b"\xe9 \\boxed{" + target_for(7).encode() + b"}\n")[:2] == (0, 1.0)


def test_a_game_that_forgot_its_imports_still_runs(tmp_path: Path) -> None:
    root = write_harbor_task(task(code=WORDLE.replace("import random\nimport re\n", "")), tmp_path)
    assert replay(root, [f"\\boxed{{{target_for(7)}}}"])[:2] == (0, 1.0)


def test_the_verifier_needs_only_the_standard_library() -> None:
    import ast

    written = task()
    modules: set[str] = set()
    for text in (written.tests["replay.py"], written.tests["env_loader.py"]):
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
        "random",
        "re",
        "statistics",
        "string",
        "sys",
        "typing",
    }


# ----------------------------------------------------------------------------------------------- the consumer


def test_harbor_itself_loads_a_gym_task(tmp_path: Path) -> None:
    config_module = pytest.importorskip("harbor.models.task.config")
    paths_module = pytest.importorskip("harbor.models.task.paths")
    root = write_harbor_task(task(), tmp_path)
    config = config_module.TaskConfig.model_validate_toml((root / "task.toml").read_text())
    assert config.environment.network_mode.value == "no-network"
    assert config.metadata["env_class"] == "WordleEnv"
    paths = paths_module.TaskPaths(root)
    assert paths.test_path.is_file() and paths.environment_dir.is_dir()
