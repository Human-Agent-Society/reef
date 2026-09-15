"""A generated environment as a Harbor task directory.

The Environment Designer writes one Python class ending in ``Env`` with Gym style ``reset(seed=None)`` and
``step(action)`` (SPADE Sec. 3.1). Here it becomes the task layout every reef consumer reads
(``reef.core.tasks``)::

    game-00012-003-deduction/
      task.toml               metadata: skill, generation, step, index, difficulty, document, env_class, max_turns, seed
      instruction.md          the gameplay prompt the agent is played with, turn by turn
      environment/Dockerfile  the verifier's image; the agent never holds the game's code
      tests/env.py            the generated class, unchanged
      tests/env_loader.py     the shared loader and rules (``environment_loader``), shipped verbatim
      tests/replay.py         replays an action log through the game and writes the episode return
      tests/test.sh
      solution/hint.txt       the privileged hint (Harbor never mounts solution/ for the agent)

The agent sees observations only, as in the paper: a driver on the host plays the game turn by turn and
keeps the action log, and the verifier scores that log. The game's code lives under ``tests/`` because
Harbor uploads ``tests/`` after the agent ran, so nothing the agent does can change what the verifier
replays. The episode return follows ``spade.core.utils.rewards.episode_reward``: the last step's reward
clipped to [-1, 1] when the episode terminated, else 0. The source record of every task is the Designer's
generation record, so ``split_generation`` keeps the environments of one generation call on one side.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Sequence
from dataclasses import dataclass

from recipes.beta.spade import environment_loader
from recipes.beta.spade.environment_loader import environment_class_name
from reef.core.tasks import HarborTask, TaskSplit, split_by_source

GAMEPLAY_PROMPT = (
    "You are playing a language game. Make valid actions to win.\n"
    "Observation: {observation}\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}."
)
ACTIONS_PATH = "/workspace/actions.txt"
DEFAULT_IMAGE = "python:3.12-slim"
DEFAULT_MAX_TURNS = 12
SKILL_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")


@dataclass(frozen=True)
class GeneratedEnvironment:
    """One environment as the Designer emitted it, with where it came from."""

    code: str
    skill: str
    generation: int
    index: int
    hint: str
    source_record_id: str
    step: int = 0
    difficulty: str | None = None
    document_id: str | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code.strip():
            raise ValueError("environment code must be non-empty text")
        if not isinstance(self.skill, str) or not SKILL_PATTERN.fullmatch(self.skill):
            raise ValueError(f"skill {self.skill!r} must match {SKILL_PATTERN.pattern}")
        for label, number in (
            ("generation", self.generation),
            ("index", self.index),
            ("step", self.step),
            ("seed", self.seed),
        ):
            if isinstance(number, bool) or not isinstance(number, int) or number < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        if not isinstance(self.hint, str) or not self.hint.strip():
            raise ValueError("hint must be non-empty text")
        if not isinstance(self.source_record_id, str) or not self.source_record_id:
            raise ValueError("source_record_id must name the Designer's generation record")
        for label, text in (("difficulty", self.difficulty), ("document_id", self.document_id)):
            if text is not None and (not isinstance(text, str) or not text):
                raise ValueError(f"{label} must be a non-empty string when set")

    @property
    def name(self) -> str:
        """The task directory name: unique per generation and index, readable by skill."""
        return f"game-{self.generation:05d}-{self.index:03d}-{self.skill}"


def environment_task(
    environment: GeneratedEnvironment, *, max_turns: int = DEFAULT_MAX_TURNS, image: str = DEFAULT_IMAGE
) -> HarborTask:
    """The Harbor task that holds ``environment``: instruction, code, replay verifier, hint and metadata."""
    if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
        raise ValueError("max_turns must be a positive integer")
    class_name = environment_class_name(environment.code)
    metadata: dict[str, object] = {
        "skill": environment.skill,
        "generation": environment.generation,
        "step": environment.step,
        "index": environment.index,
        "env_class": class_name,
        "max_turns": max_turns,
        "seed": environment.seed,
    }
    if environment.difficulty is not None:
        metadata["difficulty"] = environment.difficulty
    if environment.document_id is not None:
        metadata["document"] = environment.document_id
    instruction = (
        "You are playing a language game. Make valid actions to win.\n"
        "\n"
        "The game is served to you one turn at a time: each turn you receive an observation, reason step by "
        f"step, and put your final action within \\boxed{{}}. The game ends when it terminates or after {max_turns} "
        "turns. You never see the game's code; the verifier replays your actions through it and scores the last "
        "step of a terminated episode.\n"
    )
    test_script = (
        "#!/bin/sh\nset -eu\nmkdir -p /logs/verifier\n"
        f"python3 /tests/replay.py /tests/env.py {ACTIONS_PATH} /logs/verifier/reward.txt {max_turns} "
        f"{environment.seed}\n"
    )
    return HarborTask(
        name=environment.name,
        instruction=instruction,
        tests={
            "test.sh": test_script,
            "replay.py": REPLAY_SCRIPT,
            "env_loader.py": inspect.getsource(environment_loader),
            "env.py": environment.code,
        },
        environment={"Dockerfile": f"FROM {image}\nWORKDIR /workspace\n"},
        config={
            "agent": {"timeout_sec": 600},
            "verifier": {"timeout_sec": 120},
            "environment": {"cpus": 1, "memory_mb": 1024, "storage_mb": 1024, "gpus": 0, "network_mode": "no-network"},
        },
        metadata=metadata,
        solution={"hint.txt": environment.hint.strip() + "\n"},
        source_agent_record_ids=(environment.source_record_id,),
    )


def split_generation(tasks: Sequence[HarborTask], *, eval_fraction: float, seed: int) -> TaskSplit:
    """Split one generation's tasks so that every environment of one Designer call lands on one side."""
    names = [task.name for task in tasks]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"tasks share a name: {', '.join(repeated)}")
    return split_by_source(
        {task.name: task.source_agent_record_ids for task in tasks}, eval_fraction=eval_fraction, seed=seed
    )


# The verifier runs in the task's own image with /tests on sys.path, so it imports the shipped loader.
REPLAY_SCRIPT = '''"""Replay an action log through a generated environment and write the episode return."""

import sys

from env_loader import episode_return, load_environment_class, make_environment, play


def main(env_path, actions_path, reward_path, max_turns, seed):
    value = 0.0
    try:
        try:
            with open(actions_path, encoding="utf-8", errors="replace") as handle:
                actions = [line.rstrip("\\n") for line in handle if line.strip()]
        except OSError:
            actions = []
        environment = make_environment(load_environment_class(env_path), max_turns)
        environment.reset(seed=seed)
        rewards, terminated = play(environment, actions, max_turns)
        value = episode_return(rewards, terminated)
    finally:
        with open(reward_path, "w", encoding="utf-8") as handle:
            handle.write(f"{value}\\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
'''
