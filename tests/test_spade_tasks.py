"""A generated SPADE environment as a Harbor task: written, read back, replayed and split."""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import pytest

from recipes.beta.spade import GeneratedEnvironment, environment_loader, environment_task, split_generation
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


def replay(root: Path, actions: list[str] | bytes) -> tuple[int, float | None, str]:
    """Run tests/test.sh's command line with the container paths mapped into ``root``; (exit, reward, stderr)."""
    work = root / "work"
    work.mkdir(exist_ok=True)
    if isinstance(actions, bytes):
        (work / "actions.txt").write_bytes(actions)
    else:
        (work / "actions.txt").write_text("\n".join(actions) + "\n")
    command = (root / "tests" / "test.sh").read_text().splitlines()[-1].split()
    max_turns, seed = command[-2], command[-1]
    reward = work / "reward.txt"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "tests" / "replay.py"),
            str(root / "tests" / "env.py"),
            str(work / "actions.txt"),
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


def test_the_task_keeps_the_code_away_from_the_agent_and_ships_the_loader_with_the_verifier(tmp_path: Path) -> None:
    task = environment_task(environment())
    assert task.name == "game-00012-003-deduction"
    root = write_harbor_task(task, tmp_path)
    assert sorted(p.name for p in (root / "environment").iterdir()) == ["Dockerfile"]
    assert (root / "environment" / "Dockerfile").read_text() == "FROM python:3.12-slim\nWORKDIR /workspace\n"
    assert (root / "tests" / "env.py").read_text() == WORDLE
    assert (root / "tests" / "env_loader.py").read_text() == inspect.getsource(environment_loader)
    assert (root / "solution" / "hint.txt").read_text().startswith("The target is one of four words")
    instruction = (root / "instruction.md").read_text()
    assert "one turn at a time" in instruction and "after 12 turns" in instruction
    assert "never see the game's code" in instruction and "env.py" not in instruction
    assert "/tests/replay.py /tests/env.py /workspace/actions.txt" in (root / "tests" / "test.sh").read_text()
    document = tomllib.loads((root / "task.toml").read_text())
    assert document["metadata"]["skill"] == "deduction"
    assert document["metadata"]["env_class"] == "WordleEnv"
    assert document["metadata"]["generation"] == 12 and document["metadata"]["index"] == 3
    assert document["metadata"]["difficulty"] == "medium" and document["metadata"]["document"] == "dclm-000123"
    assert document["metadata"]["max_turns"] == 12 and document["metadata"]["seed"] == 7
    assert document["metadata"]["reef"]["source_agent_record_ids"] == ["rec-designer-12"]
    assert document["environment"]["network_mode"] == "no-network"
    assert read_harbor_task(root) == task


def test_optional_fields_are_left_out_of_the_metadata_when_unset() -> None:
    task = environment_task(environment(difficulty=None, document_id=None), max_turns=20)
    assert "difficulty" not in task.metadata and "document" not in task.metadata
    assert task.metadata["max_turns"] == 20
    assert "after 20 turns" in task.instruction


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"code": "  "}, "non-empty text"),
        ({"code": "x = 1\n"}, "no top level class ending in 'Env'"),
        ({"code": "class Env:\n    pass\n"}, "no top level class ending in 'Env'"),
        ({"code": "class TicketEnv(ToolUseBaseEnv):\n    pass\n"}, "ToolUseBaseEnv"),
        ({"code": "def broken(:\n"}, "not valid Python"),
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


# ----------------------------------------------------------------------------------------------- the verifier


def test_the_verifier_scores_a_win_a_loss_and_a_walkover(tmp_path: Path) -> None:
    root = write_harbor_task(environment_task(environment(seed=7)), tmp_path)
    target = target_for(7)
    assert replay(root, ["\\boxed{lemon}", f"reasoning then \\boxed{{{target}}}"])[:2] == (0, 1.0)
    assert replay(root, ["\\boxed{zzzzz}"] * 6)[:2] == (0, 0.0)
    assert replay(root, [])[:2] == (0, 0.0)
    assert replay(root, [target])[:2] == (0, 0.0)


def test_the_verifier_stops_at_the_turn_limit(tmp_path: Path) -> None:
    root = write_harbor_task(environment_task(environment(seed=7), max_turns=2), tmp_path)
    target = target_for(7)
    assert replay(root, ["\\boxed{zzzzz}", "\\boxed{zzzzz}", f"\\boxed{{{target}}}"])[:2] == (0, 0.0)


def test_the_verifier_writes_a_reward_even_when_the_game_breaks(tmp_path: Path) -> None:
    code = "class AEnv:\n    def reset(self, seed=None):\n        return 'o', {}\n    def step(self, action):\n        raise KeyError(action)\n"
    root = write_harbor_task(environment_task(environment(code=code)), tmp_path)
    assert replay(root, ["\\boxed{go}"])[:2] == (0, -1.0)
    broken_reset = "class AEnv:\n    def reset(self, seed=None):\n        raise RuntimeError('no game')\n"
    root = write_harbor_task(environment_task(environment(code=broken_reset, index=4)), tmp_path)
    exit_code, value, stderr = replay(root, ["\\boxed{go}"])
    assert exit_code == 1 and value == 0.0 and "no game" in stderr


def test_the_verifier_tolerates_an_action_log_that_is_not_utf8(tmp_path: Path) -> None:
    root = write_harbor_task(environment_task(environment(seed=7)), tmp_path)
    assert replay(root, b"\xe9 \\boxed{" + target_for(7).encode() + b"}\n")[:2] == (0, 1.0)


def test_a_game_that_forgot_its_imports_still_runs(tmp_path: Path) -> None:
    code = WORDLE.replace("import random\nimport re\n", "")
    root = write_harbor_task(environment_task(environment(code=code, seed=7)), tmp_path)
    assert replay(root, [f"\\boxed{{{target_for(7)}}}"])[:2] == (0, 1.0)


def test_the_verifier_needs_only_the_standard_library() -> None:
    import ast

    task = environment_task(environment())
    modules: set[str] = set()
    for text in (task.tests["replay.py"], task.tests["env_loader.py"]):
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
        "env_loader",
        "importlib",
        "itertools",
        "json",
        "math",
        "random",
        "re",
        "sys",
        "typing",
    }


# ----------------------------------------------------------------------------------------------- the split


def test_environments_of_one_designer_call_stay_on_one_side() -> None:
    tasks = [
        environment_task(environment(index=index, source_record_id=f"rec-designer-{index // 2}")) for index in range(8)
    ]
    for seed in range(10):
        split = split_generation(tasks, eval_fraction=0.5, seed=seed)
        for index in range(0, 8, 2):
            first, second = f"game-00012-{index:03d}-deduction", f"game-00012-{index + 1:03d}-deduction"
            assert (first in split.eval) == (second in split.eval), (seed, split)
        assert len(split.eval) >= 4


def test_two_tasks_with_one_name_are_refused_instead_of_dropped() -> None:
    first = environment_task(environment())
    second = environment_task(environment(code=WORDLE + "#\n", source_record_id="rec-other"))
    with pytest.raises(ValueError, match="game-00012-003-deduction"):
        split_generation([first, second], eval_fraction=0.5, seed=0)


# ----------------------------------------------------------------------------------------------- the consumer


def test_harbor_itself_loads_a_generated_environment_task(tmp_path: Path) -> None:
    config_module = pytest.importorskip("harbor.models.task.config")
    paths_module = pytest.importorskip("harbor.models.task.paths")
    root = write_harbor_task(environment_task(environment()), tmp_path)
    config = config_module.TaskConfig.model_validate_toml((root / "task.toml").read_text())
    assert config.environment.network_mode.value == "no-network"
    assert config.metadata["env_class"] == "WordleEnv"
    paths = paths_module.TaskPaths(root)
    assert paths.test_path.is_file() and paths.environment_dir.is_dir()
