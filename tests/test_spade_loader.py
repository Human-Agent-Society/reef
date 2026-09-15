"""The shared loader and rules: the class, the injected names, the action, the step, the return."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from recipes.beta.spade import (
    environment_class_name,
    episode_return,
    extract_boxed_answer,
    load_environment_class,
    make_environment,
    normalized_action,
    replay,
)

GUESS = """import re


class GuessEnv:
    def __init__(self, max_turns=20):
        self.max_turns = max_turns
        self.turns = 0
        self.target = 0

    def reset(self, seed=None):
        self.target = random.Random(seed).randint(1, 3)
        self.turns = 0
        return "Guess a number from 1 to 3. Answer with \\\\boxed{n}.", {}

    def step(self, action):
        self.turns += 1
        match = re.search(r"\\\\boxed\\{([^}]*)\\}", action)
        if not match:
            return "Use \\\\boxed{n}.", 0.0, False, False, {}
        if match.group(1).strip() == str(self.target):
            return "Right.", 1.0, True, False, {}
        if self.turns >= self.max_turns:
            return "The fort fell.", -1.0, True, False, {}
        return "Wrong.", 0.0, False, False, {}
"""


def written(tmp_path: Path, code: str) -> str:
    path = tmp_path / "env.py"
    path.write_text(code, encoding="utf-8")
    return str(path)


# ----------------------------------------------------------------------------------------------- the class


def test_the_class_is_the_first_top_level_class_ending_in_env() -> None:
    assert environment_class_name("import os\nos.system('rm -rf /')\nclass BombEnv:\n    pass\n") == "BombEnv"
    assert environment_class_name("class Env:\n    pass\n\nclass GameEnv(Env):\n    pass\n") == "GameEnv"


@pytest.mark.parametrize(
    ("code", "message"),
    [
        ("x = 1\n", "no top level class ending in 'Env'"),
        ("class Env:\n    pass\n", "no top level class ending in 'Env'"),
        ("class TicketEnv(ToolUseBaseEnv):\n    pass\n", "ToolUseBaseEnv"),
        ("def broken(:\n", "not valid Python"),
        ("if True:\n    class GameEnv:\n        pass\n", "no top level class"),
    ],
)
def test_code_without_a_usable_class_is_refused_before_it_runs(code: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        environment_class_name(code)


def test_a_tool_use_environment_is_refused_because_its_base_never_ships() -> None:
    with pytest.raises(ValueError, match="ToolUseBaseEnv"):
        environment_class_name("class TicketEnv(ToolUseBaseEnv):\n    pass\n")


def test_the_loader_runs_the_named_class_with_the_injected_names(tmp_path: Path) -> None:
    environment_class = load_environment_class(written(tmp_path, GUESS))
    assert environment_class.__name__ == "GuessEnv"
    environment = make_environment(environment_class, 12)
    observation, info = environment.reset(seed=7)  # type: ignore[attr-defined]
    assert observation.startswith("Guess a number") and info == {}


def test_the_loader_registers_the_module_so_dataclasses_work(tmp_path: Path) -> None:
    code = (
        "from __future__ import annotations\nfrom dataclasses import dataclass, field\n\n@dataclass\nclass StateEnv:\n"
        "    turns: int = 0\n    history: list[str] = field(default_factory=list)\n"
        "    def reset(self, seed=None):\n        return 'o', {}\n"
        "    def step(self, action):\n        return 'o', 1.0, True, False, {}\n"
    )
    environment = make_environment(load_environment_class(written(tmp_path, code)), 12)
    assert environment.reset(seed=0) == ("o", {})  # type: ignore[attr-defined]


def test_the_loader_picks_the_class_the_metadata_names_not_the_first_binding(tmp_path: Path) -> None:
    code = "BaseEnv = type('BaseEnv', (), {})\n\nclass GameEnv(BaseEnv):\n    def reset(self, seed=None):\n        return 'o', {}\n"
    assert load_environment_class(written(tmp_path, code)).__name__ == "GameEnv"


# ----------------------------------------------------------------------------------------------- the instance


def test_a_class_that_needs_max_turns_gets_it_and_a_designed_limit_is_capped(tmp_path: Path) -> None:
    needs = "class NeedsEnv:\n    def __init__(self, max_turns):\n        self.max_turns = max_turns\n"
    environment = make_environment(load_environment_class(written(tmp_path, needs)), 5)
    assert environment.max_turns == 5  # type: ignore[attr-defined]
    guess = make_environment(load_environment_class(written(tmp_path, GUESS)), 12)
    assert guess.max_turns == 12  # type: ignore[attr-defined]
    guess = make_environment(load_environment_class(written(tmp_path, GUESS)), 40)
    assert guess.max_turns == 20  # type: ignore[attr-defined]


def test_a_class_limit_that_is_not_a_positive_int_is_left_alone(tmp_path: Path) -> None:
    code = "class OddEnv:\n    max_turns = 'many'\n"
    environment = make_environment(load_environment_class(written(tmp_path, code)), 5)
    assert environment.max_turns == "many"  # type: ignore[attr-defined]


# ----------------------------------------------------------------------------------------------- the action


@pytest.mark.parametrize(
    ("text", "answer"),
    [
        (r"I think \boxed{trace}", "trace"),
        (r"\boxed{{28}}", "28"),
        (r"\boxed{{{7}}}", "7"),
        (r"first \boxed{ a } then \boxed{b}", "b"),
        (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),
        (r"\boxed{a{b}c}", "a{b}c"),
        ("no box here", None),
        (r"\boxed{}", ""),
        (r"\boxed{open", None),
    ],
)
def test_the_boxed_answer_is_read_like_the_reference(text: str, answer: str | None) -> None:
    assert extract_boxed_answer(text) == answer


@pytest.mark.parametrize(
    ("text", "action"),
    [
        (r"I think \boxed{ trace }", r"\boxed{trace}"),
        ("  no box  ", "no box"),
        (r"\boxed{\frac{1}{2}}", r"\boxed{\frac{1}{2}}"),
    ],
)
def test_step_receives_the_answer_re_boxed_or_the_raw_text(text: str, action: str) -> None:
    assert normalized_action(text) == action


# ----------------------------------------------------------------------------------------------- the play


def test_a_template_conforming_game_wins_loses_and_walks_over(tmp_path: Path) -> None:
    environment_class = load_environment_class(written(tmp_path, GUESS))
    environment = make_environment(environment_class, 12)
    environment.reset(seed=7)  # type: ignore[attr-defined]
    target = environment.target  # type: ignore[attr-defined]
    rewards, terminated = replay(environment, [r"\boxed{9}", f"so \\boxed{{{target}}}"], 12)
    assert (rewards, terminated) == ([0.0, 1.0], True)
    environment = make_environment(environment_class, 3)
    environment.reset(seed=7)  # type: ignore[attr-defined]
    rewards, terminated = replay(environment, [r"\boxed{9}"] * 5, 3)
    assert (rewards, terminated) == ([0.0, 0.0, -1.0], True)
    assert replay(make_environment(environment_class, 3), [], 3) == ([], False)


def test_a_missing_box_is_a_format_reminder_not_a_loss(tmp_path: Path) -> None:
    environment = make_environment(load_environment_class(written(tmp_path, GUESS)), 12)
    environment.reset(seed=7)  # type: ignore[attr-defined]
    assert replay(environment, ["2"], 12) == ([0.0], False)


@pytest.mark.parametrize(
    ("body", "rewards", "terminated"),
    [
        ("raise KeyError('x')", [-1.0], True),
        ("return 'o', None, True, False, {}", [0.0], True),
        ("return 'o', 'win', True, False, {}", [-1.0], True),
        ("return 'o', 1.0, True, {}", [-1.0], True),
        ("return 'o', float('nan'), True, False, {}", [0.0], True),
        ("return 'o', 5, 1, 0, {}", [5.0], True),
    ],
)
def test_a_broken_step_ends_the_episode_the_way_the_reference_does(
    tmp_path: Path, body: str, rewards: list[float], terminated: bool
) -> None:
    code = f"class AEnv:\n    def reset(self, seed=None):\n        return 'o', {{}}\n    def step(self, action):\n        {body}\n"
    environment = make_environment(load_environment_class(written(tmp_path, code)), 12)
    environment.reset(seed=0)  # type: ignore[attr-defined]
    assert replay(environment, [r"\boxed{go}", r"\boxed{again}"], 12) == (rewards, terminated)


@pytest.mark.parametrize(
    ("rewards", "terminated", "value"),
    [
        ([0.0, 0.0, 1.0], True, 1.0),
        ([0.0, 1.0], False, 0.0),
        ([], True, 0.0),
        ([5.0], True, 1.0),
        ([-3.0], True, -1.0),
        ([0.5, 0.25], True, 0.25),
        ([math.nan], True, 0.0),
        ([math.inf], True, 0.0),
    ],
)
def test_the_episode_return_is_the_last_reward_of_a_terminated_episode(
    rewards: list[float], terminated: bool, value: float
) -> None:
    assert episode_return(rewards, terminated) == value
