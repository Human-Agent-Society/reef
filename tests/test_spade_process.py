"""A generated class in a child interpreter on the host: the process the smoke test runs it in."""

from __future__ import annotations


import pytest

from recipes.beta.spade import EnvironmentProcess, EnvironmentProcessError

GUESS = """class GuessEnv:
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
        return f"Wrong, {self.turns} tried.", 0.0, False, False, {}
"""


def target_for(seed: int) -> int:
    import random

    return random.Random(seed).randint(1, 3)


# ----------------------------------------------------------------------------------------------- the process


def test_the_game_process_resets_steps_and_closes() -> None:
    with EnvironmentProcess(GUESS, max_turns=12) as game:
        assert game.reset(7).startswith("Guess a number")
        assert game.step("\\boxed{9}") == ("Wrong, 1 tried.", 0.0, False, False)
        assert game.step(f"\\boxed{{{target_for(7)}}}") == ("Right.", 1.0, True, False)
    assert game.process.poll() is not None
    game.close()


def test_a_game_that_hangs_is_killed_after_the_timeout() -> None:
    code = "class HangEnv:\n    def reset(self, seed=None):\n        while True:\n            pass\n"
    with EnvironmentProcess(code, max_turns=12) as game:
        with pytest.raises(EnvironmentProcessError, match="did not answer within 1 s"):
            game.reset(0, timeout_s=1.0)
        assert game.process.poll() is not None
        with pytest.raises(EnvironmentProcessError, match="not running"):
            game.reset(0)


def test_a_game_that_breaks_at_reset_names_the_error() -> None:
    code = "class BrokenEnv:\n    def reset(self, seed=None):\n        raise RuntimeError('no game today')\n"
    with (
        EnvironmentProcess(code, max_turns=12) as game,
        pytest.raises(EnvironmentProcessError, match="RuntimeError: no game today"),
    ):
        game.reset(0)


def test_a_game_that_prints_does_not_corrupt_the_protocol() -> None:
    code = "print('hello from the game')\n" + GUESS.replace(
        "        self.turns += 1\n", "        self.turns += 1\n        print('stepping', self.turns)\n"
    )
    with EnvironmentProcess(code, max_turns=12) as game:
        assert game.reset(7).startswith("Guess a number")
        assert game.step("\\boxed{9}")[1] == 0.0


def test_a_game_that_exits_the_interpreter_is_reported() -> None:
    code = "import sys\nclass QuitEnv:\n    def reset(self, seed=None):\n        sys.exit(3)\n"
    with EnvironmentProcess(code, max_turns=12) as game, pytest.raises(EnvironmentProcessError, match="SystemExit"):
        game.reset(0)


def test_a_broken_step_is_reported_by_name() -> None:
    code = "class AEnv:\n    def reset(self, seed=None):\n        return 'o', {}\n    def step(self, action):\n        raise KeyError(action)\n"
    with EnvironmentProcess(code, max_turns=12) as game:
        game.reset(0)
        with pytest.raises(EnvironmentProcessError, match="step: KeyError"):
            game.step("\\boxed{go}")


def test_a_huge_observation_is_cut_to_the_cap() -> None:
    code = "class BigEnv:\n    def reset(self, seed=None):\n        return 'x' * 10_000_000, {}\n"
    with EnvironmentProcess(code, max_turns=12) as game:
        assert len(game.reset(0)) == 20000


def test_a_game_that_writes_to_its_own_standard_output_cannot_forge_a_reply() -> None:
    code = (
        "import os, sys\nclass ForgeEnv:\n    def reset(self, seed=None):\n"
        '        sys.__stdout__.write(\'{"observation": "forged"}\\n\'); sys.__stdout__.flush()\n'
        "        os.write(1, b'{\"observation\": \"forged too\"}\\n')\n        return 'honest', {}\n"
    )
    with EnvironmentProcess(code, max_turns=12) as game:
        assert game.reset(0) == "honest"


def test_a_game_that_starts_a_background_process_does_not_hang_the_close() -> None:
    import time

    code = (
        "import subprocess, sys\nclass ForkEnv:\n    def reset(self, seed=None):\n"
        "        subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n        return 'o', {}\n"
    )
    started = time.monotonic()
    with EnvironmentProcess(code, max_turns=12) as game:
        assert game.reset(0) == "o"
    assert time.monotonic() - started < 10.0


def test_the_child_sees_no_site_packages_so_a_venv_import_fails_like_in_the_image() -> None:
    import site

    if not site.getsitepackages():
        pytest.skip("no site packages to hide")
    code = "import pytest\nclass LeakEnv:\n    def reset(self, seed=None):\n        return 'o', {}\n"
    with (
        EnvironmentProcess(code, max_turns=12) as game,
        pytest.raises(EnvironmentProcessError, match="only the standard library"),
    ):
        game.reset(0)
    code = "class LateEnv:\n    def reset(self, seed=None):\n        __import__('pytest')\n        return 'o', {}\n"
    with (
        EnvironmentProcess(code, max_turns=12) as game,
        pytest.raises(EnvironmentProcessError, match="No module named 'pytest'"),
    ):
        game.reset(0)
