"""The agent plays a gym environment turn by turn in a child process, and the task's verifier agrees with it."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from reef.core.tasks import gym_task, write_harbor_task
from reef.core.tasks.gym_loader import write_actions
from reef.harness.runners.gym.play import Episode, GameProcess, GameProcessError, gameplay_messages, play_episode

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


class ScriptedAgent:
    """Answers from a fixed list and keeps every message list it was shown."""

    def __init__(self, replies: Sequence[str]) -> None:
        self.replies = list(replies)
        self.seen: list[list[dict[str, str]]] = []

    def __call__(self, messages: Sequence[Mapping[str, str]]) -> str:
        self.seen.append([dict(m) for m in messages])
        if not self.replies:
            raise RuntimeError("the agent has nothing left to say")
        return self.replies.pop(0)


def verifier_return(root: Path, actions: list[str]) -> float:
    """Run tests/replay.py the way test.sh does, with the container paths mapped into ``root``."""
    work = root / "work"
    work.mkdir(exist_ok=True)
    write_actions(str(work / "actions.jsonl"), actions)
    command = (root / "tests" / "test.sh").read_text().splitlines()[-1].split()
    subprocess.run(
        [
            sys.executable,
            str(root / "tests" / "replay.py"),
            str(root / "tests" / "env.py"),
            str(work / "actions.jsonl"),
            str(work / "reward.txt"),
            command[-2],
            command[-1],
        ],
        check=True,
        timeout=60,
    )
    return float((work / "reward.txt").read_text())


# ----------------------------------------------------------------------------------------------- the messages


def test_the_first_turn_wraps_the_observation_in_the_gameplay_prompt() -> None:
    assert gameplay_messages("Guess.", None) == [
        {
            "role": "user",
            "content": (
                "You are playing a language game. Make valid actions to win.\n"
                "Observation: Guess.\n"
                "Please reason step by step, and put your final answer within \\boxed{}."
            ),
        }
    ]


def test_the_hint_arm_appends_the_hint_to_the_first_observation() -> None:
    content = gameplay_messages("Guess.", "Start low.")[0]["content"]
    assert "Observation: Guess.\n\nHINT: Start low.\n" in content


def test_a_system_prompt_comes_first_and_an_empty_one_is_left_out() -> None:
    messages = gameplay_messages("Guess.", None, "Be brief.\n")
    assert messages[0] == {"role": "system", "content": "Be brief."} and messages[1]["role"] == "user"
    assert [m["role"] for m in gameplay_messages("Guess.", None, "  ")] == ["user"]


def test_braces_in_the_observation_and_the_hint_are_kept_as_text() -> None:
    content = gameplay_messages("Set {a, b}. Reply with \\boxed{x}.", "Use {observation} literally.")[0]["content"]
    assert "Observation: Set {a, b}. Reply with \\boxed{x}.\n\nHINT: Use {observation} literally." in content


# ----------------------------------------------------------------------------------------------- the process


def test_the_game_process_resets_steps_and_closes() -> None:
    with GameProcess(GUESS, max_turns=12) as game:
        assert game.reset(7).startswith("Guess a number")
        assert game.step("\\boxed{9}") == ("Wrong, 1 tried.", 0.0, False, False)
        assert game.step(f"\\boxed{{{target_for(7)}}}") == ("Right.", 1.0, True, False)
    assert game.process.poll() is not None
    game.close()


def test_a_game_that_hangs_is_killed_after_the_timeout() -> None:
    code = "class HangEnv:\n    def reset(self, seed=None):\n        while True:\n            pass\n"
    with GameProcess(code, max_turns=12) as game:
        with pytest.raises(GameProcessError, match="did not answer within 1 s"):
            game.reset(0, timeout_s=1.0)
        assert game.process.poll() is not None
        with pytest.raises(GameProcessError, match="not running"):
            game.reset(0)


def test_a_game_that_breaks_at_reset_names_the_error() -> None:
    code = "class BrokenEnv:\n    def reset(self, seed=None):\n        raise RuntimeError('no game today')\n"
    with GameProcess(code, max_turns=12) as game, pytest.raises(GameProcessError, match="RuntimeError: no game today"):
        game.reset(0)


def test_a_game_that_prints_does_not_corrupt_the_protocol() -> None:
    code = "print('hello from the game')\n" + GUESS.replace(
        "        self.turns += 1\n", "        self.turns += 1\n        print('stepping', self.turns)\n"
    )
    with GameProcess(code, max_turns=12) as game:
        assert game.reset(7).startswith("Guess a number")
        assert game.step("\\boxed{9}")[1] == 0.0


def test_a_game_that_exits_the_interpreter_is_reported() -> None:
    code = "import sys\nclass QuitEnv:\n    def reset(self, seed=None):\n        sys.exit(3)\n"
    with GameProcess(code, max_turns=12) as game, pytest.raises(GameProcessError, match="SystemExit"):
        game.reset(0)


def test_a_broken_step_ends_the_episode_with_the_error_reward() -> None:
    code = "class AEnv:\n    def reset(self, seed=None):\n        return 'o', {}\n    def step(self, action):\n        raise KeyError(action)\n"
    with GameProcess(code, max_turns=12) as game:
        game.reset(0)
        observation, reward, terminated, truncated = game.step("\\boxed{go}")
        assert reward == -1.0 and terminated and not truncated
        assert observation.startswith("Error: KeyError")


def test_a_huge_observation_is_cut_to_the_cap() -> None:
    code = "class BigEnv:\n    def reset(self, seed=None):\n        return 'x' * 10_000_000, {}\n"
    with GameProcess(code, max_turns=12) as game:
        assert len(game.reset(0)) == 20000


def test_a_game_that_writes_to_its_own_standard_output_cannot_forge_a_reply() -> None:
    code = (
        "import os, sys\nclass ForgeEnv:\n    def reset(self, seed=None):\n"
        '        sys.__stdout__.write(\'{"observation": "forged"}\\n\'); sys.__stdout__.flush()\n'
        "        os.write(1, b'{\"observation\": \"forged too\"}\\n')\n        return 'honest', {}\n"
    )
    with GameProcess(code, max_turns=12) as game:
        assert game.reset(0) == "honest"


def test_a_game_that_starts_a_background_process_does_not_hang_the_close() -> None:
    import time

    code = (
        "import subprocess, sys\nclass ForkEnv:\n    def reset(self, seed=None):\n"
        "        subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n        return 'o', {}\n"
    )
    started = time.monotonic()
    with GameProcess(code, max_turns=12) as game:
        assert game.reset(0) == "o"
    assert time.monotonic() - started < 10.0


def test_the_child_sees_no_site_packages_so_a_venv_import_fails_like_in_the_image() -> None:
    import site

    if not site.getsitepackages():
        pytest.skip("no site packages to hide")
    code = "import pytest\nclass LeakEnv:\n    def reset(self, seed=None):\n        return 'o', {}\n"
    with GameProcess(code, max_turns=12) as game, pytest.raises(GameProcessError, match="only the standard library"):
        game.reset(0)
    code = "class LateEnv:\n    def reset(self, seed=None):\n        __import__('pytest')\n        return 'o', {}\n"
    with GameProcess(code, max_turns=12) as game, pytest.raises(GameProcessError, match="No module named 'pytest'"):
        game.reset(0)


# ----------------------------------------------------------------------------------------------- the episode


def test_the_agent_wins_in_two_turns_and_the_verifier_agrees(tmp_path: Path) -> None:
    target = target_for(7)
    agent = ScriptedAgent(["Let me try\n\\boxed{9}", f"Then it must be \\boxed{{{target}}}"])
    episode = play_episode(GUESS, agent, seed=7, max_turns=12, system_prompt="Be brief.")
    assert episode.failure is None and episode.terminated
    assert episode.rewards == [0.0, 1.0] and episode.return_value == 1.0
    assert [turn.action for turn in episode.turns] == ["\\boxed{9}", f"\\boxed{{{target}}}"]
    assert episode.actions == ["Let me try\n\\boxed{9}", f"Then it must be \\boxed{{{target}}}"]
    assert [turn.observation for turn in episode.turns] == ["Wrong, 1 tried.", "Right."]
    # The second call saw the first observation as a plain user turn after the assistant's reply.
    assert [m["role"] for m in agent.seen[1]] == ["system", "user", "assistant", "user"]
    assert agent.seen[1][3] == {"role": "user", "content": "Wrong, 1 tried."}
    root = write_harbor_task(gym_task(name="guess-007", code=GUESS, seed=7), tmp_path)
    assert verifier_return(root, episode.actions) == episode.return_value


def test_a_boxed_answer_that_spans_lines_replays_to_the_same_return(tmp_path: Path) -> None:
    target = target_for(7)
    agent = ScriptedAgent([f"\\boxed{{{target}\n}}"])
    episode = play_episode(GUESS, agent, seed=7, max_turns=12)
    root = write_harbor_task(gym_task(name="guess-007", code=GUESS, seed=7), tmp_path)
    assert verifier_return(root, episode.actions) == episode.return_value == 1.0


def test_an_empty_reply_counts_as_a_turn_for_the_driver_and_the_verifier_alike(tmp_path: Path) -> None:
    agent = ScriptedAgent(["", "\\boxed{9}", "\\boxed{9}"])
    episode = play_episode(GUESS, agent, seed=7, max_turns=3)
    assert episode.actions == ["", "\\boxed{9}", "\\boxed{9}"] and episode.return_value == -1.0
    root = write_harbor_task(gym_task(name="guess-007", code=GUESS, seed=7, max_turns=3), tmp_path)
    assert verifier_return(root, episode.actions) == -1.0


def test_the_hint_arm_shows_the_hint_once_in_the_first_message() -> None:
    agent = ScriptedAgent([f"\\boxed{{{target_for(3)}}}"])
    episode = play_episode(GUESS, agent, seed=3, max_turns=12, hint="The number is small.")
    assert episode.return_value == 1.0
    assert "HINT: The number is small." in agent.seen[0][0]["content"]


def test_the_turn_limit_ends_an_unfinished_episode_as_a_walkover() -> None:
    agent = ScriptedAgent(["\\boxed{9}"] * 5)
    episode = play_episode(GUESS, agent, seed=7, max_turns=3)
    assert len(episode.turns) == 3 and episode.failure is None
    assert episode.rewards == [0.0, 0.0, -1.0] and episode.terminated and episode.return_value == -1.0


def test_a_reply_without_a_box_is_passed_raw_and_the_game_reminds_the_agent() -> None:
    agent = ScriptedAgent(["2", f"\\boxed{{{target_for(7)}}}"])
    episode = play_episode(GUESS, agent, seed=7, max_turns=12)
    assert episode.turns[0].action == "2" and episode.turns[0].observation == "Use \\boxed{n}."
    assert episode.return_value == 1.0


def test_a_model_failure_is_the_episodes_failure_not_the_callers() -> None:
    agent = ScriptedAgent(["\\boxed{9}"])
    episode = play_episode(GUESS, agent, seed=7, max_turns=12)
    assert episode.failure == "model: the agent has nothing left to say"
    assert len(episode.turns) == 1 and episode.return_value == 0.0


def test_a_reply_that_is_not_text_fails_the_episode() -> None:
    episode = play_episode(GUESS, lambda messages: None, seed=7, max_turns=12)  # type: ignore[arg-type,return-value]
    assert episode.failure == "model: reply is not text" and episode.return_value == 0.0


def test_a_game_that_cannot_start_fails_the_episode_before_the_model_is_called() -> None:
    code = "class BrokenEnv:\n    def reset(self, seed=None):\n        raise RuntimeError('no game today')\n"
    agent = ScriptedAgent(["\\boxed{1}"])
    episode = play_episode(code, agent, seed=0, max_turns=12)
    assert episode.failure is not None and episode.failure.startswith("reset: RuntimeError: no game today")
    assert agent.seen == [] and episode.return_value == 0.0


def test_a_game_that_hangs_mid_episode_fails_the_episode() -> None:
    code = GUESS.replace(
        "        self.turns += 1\n", "        self.turns += 1\n        while self.turns == 2:\n            pass\n"
    )
    agent = ScriptedAgent(["\\boxed{9}", "\\boxed{9}", "\\boxed{9}"])
    episode = play_episode(code, agent, seed=7, max_turns=12, step_timeout_s=1.0)
    assert episode.failure == "step: the game did not answer within 1 s"
    assert len(episode.turns) == 1 and episode.return_value == 0.0


def test_an_episode_with_no_turns_is_a_walkover() -> None:
    assert Episode(first_observation="o").return_value == 0.0 and Episode(first_observation="o").actions == []
