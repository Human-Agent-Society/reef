"""The agent plays a generated environment turn by turn in a child process, and the verifier agrees with it."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from recipes.beta.spade import (
    Episode,
    GameProcess,
    GameProcessError,
    GeneratedEnvironment,
    environment_task,
    gameplay_messages,
    play_episode,
)
from reef.core.tasks import write_harbor_task

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
    (work / "actions.txt").write_text("\n".join(actions) + "\n")
    command = (root / "tests" / "test.sh").read_text().splitlines()[-1].split()
    subprocess.run(
        [
            sys.executable,
            str(root / "tests" / "replay.py"),
            str(root / "tests" / "env.py"),
            str(work / "actions.txt"),
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
    messages = gameplay_messages("Guess.", None)
    assert messages == [
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


# ----------------------------------------------------------------------------------------------- the process


def test_the_game_process_resets_steps_and_closes() -> None:
    with GameProcess(GUESS, max_turns=12) as game:
        assert game.reset(7).startswith("Guess a number")
        observation, reward, terminated, truncated = game.step("\\boxed{9}")
        assert (observation, reward, terminated, truncated) == ("Wrong, 1 tried.", 0.0, False, False)
        assert game.step(f"\\boxed{{{target_for(7)}}}") == ("Right.", 1.0, True, False)
    assert game.process.poll() is not None
    game.close()


def test_a_game_that_hangs_is_killed_after_the_timeout() -> None:
    code = "class HangEnv:\n    def reset(self, seed=None):\n        while True:\n            pass\n"
    with GameProcess(code, max_turns=12) as game:
        with pytest.raises(GameProcessError, match="did not answer within 1 s"):
            game.reset(0, timeout_s=1.0)
        assert game.process.poll() is not None


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


# ----------------------------------------------------------------------------------------------- the episode


def test_the_agent_wins_in_two_turns_and_the_verifier_agrees(tmp_path: Path) -> None:
    target = target_for(7)
    agent = ScriptedAgent(["Let me try\n\\boxed{9}", f"Then it must be \\boxed{{{target}}}"])
    episode = play_episode(GUESS, agent, seed=7, max_turns=12)
    assert episode.failure is None and episode.terminated
    assert episode.rewards == [0.0, 1.0] and episode.return_value == 1.0
    assert [turn.action for turn in episode.turns] == ["\\boxed{9}", f"\\boxed{{{target}}}"]
    assert episode.actions == ["Let me try \\boxed{9}", f"Then it must be \\boxed{{{target}}}"]
    assert [turn.observation for turn in episode.turns] == ["Wrong, 1 tried.", "Right."]
    # The second call saw the first observation as a plain user turn after the assistant's reply.
    assert [m["role"] for m in agent.seen[1]] == ["user", "assistant", "user"]
    assert agent.seen[1][2] == {"role": "user", "content": "Wrong, 1 tried."}
    task = environment_task(
        GeneratedEnvironment(
            code=GUESS, skill="guessing", generation=1, index=1, hint="Count.", source_record_id="r", seed=7
        )
    )
    root = write_harbor_task(task, tmp_path)
    assert verifier_return(root, episode.actions) == episode.return_value


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


def test_a_model_failure_is_the_episodes_failure_not_the_drivers() -> None:
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
