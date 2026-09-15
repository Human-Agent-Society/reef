"""A generated SPADE environment as a gym task: SPADE's naming, metadata and hint, and the split per generation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from recipes.beta.spade import GeneratedEnvironment, environment_task, split_generation
from reef.core.tasks import read_gym_task, read_harbor_task, write_harbor_task
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


def test_the_task_carries_the_class_the_hint_and_the_metadata_as_a_gym_task(tmp_path: Path) -> None:
    task = environment_task(environment())
    assert task.name == "game-00012-003-deduction"
    root = write_harbor_task(task, tmp_path)
    gym = read_gym_task(root)
    assert gym.code == WORDLE and gym.max_turns == 12 and gym.seed == 7
    assert (root / "solution" / "hint.txt").read_text().startswith("The target is one of four words")
    document = tomllib.loads((root / "task.toml").read_text())
    assert document["metadata"] == {
        "skill": "deduction",
        "generation": 12,
        "step": 48,
        "index": 3,
        "difficulty": "medium",
        "document": "dclm-000123",
        "env_class": "WordleEnv",
        "max_turns": 12,
        "seed": 7,
        "reef": {"digest": task.digest, "source_agent_record_ids": ["rec-designer-12"]},
    }
    assert read_harbor_task(root) == task


def test_optional_fields_are_left_out_of_the_metadata_when_unset() -> None:
    task = environment_task(environment(difficulty=None, document_id=None), max_turns=20)
    assert "difficulty" not in task.metadata and "document" not in task.metadata
    assert task.metadata["max_turns"] == 20


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


def test_the_written_task_replays_a_win_through_its_own_verifier(tmp_path: Path) -> None:
    root = write_harbor_task(environment_task(environment(seed=7)), tmp_path)
    work = root / "work"
    work.mkdir()
    write_actions(str(work / "actions.jsonl"), ["\\boxed{lemon}", f"so \\boxed{{{target_for(7)}}}"])
    subprocess.run(
        [
            sys.executable,
            str(root / "tests" / "replay.py"),
            str(root / "tests" / "env.py"),
            str(work / "actions.jsonl"),
            str(work / "reward.txt"),
            "12",
            "7",
        ],
        check=True,
        timeout=60,
    )
    assert (work / "reward.txt").read_text() == "1.0\n"


def test_environments_of_one_designer_call_stay_in_one_split() -> None:
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
