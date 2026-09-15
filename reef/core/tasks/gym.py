"""A Gym style environment as a Harbor task: the environment's code with the verifier, never with the agent.

A gym task is a Harbor task directory whose environment is a Python class with ``reset(seed=None)`` and
``step(action)`` (SPADE Sec. 3.1): a game, a simulated tool use setting, anything a class can hold. The
agent never holds the class: the ``gym`` harness adapter serves the environment one turn at a time and
records the agent's replies, and the verifier replays that log through the class with the rules of
:mod:`reef.core.tasks.gym_loader`::

    <name>/
      task.toml               metadata: max_turns, seed, and whatever the writer adds
      instruction.md          the prompt the agent is played with
      environment/Dockerfile  the verifier's image; nothing of the environment is in it
      tests/env.py            the class, unchanged
      tests/env_loader.py     the shared loader and rules, shipped verbatim
      tests/replay.py         replays /workspace/actions.jsonl through the class and writes the episode return
      tests/test.sh
      solution/               whatever the writer adds (a hint, a walkthrough); Harbor never mounts it for the agent

Harbor uploads ``tests/`` after the agent ran. Under Harbor's default shared verifier mode the verifier
still runs in the container the agent had a root shell in, so a Harbor run of a gym task trusts the
agent not to shim ``python3``; the ``gym`` harness adapter plays and scores the task on the host instead,
and a Harbor run that must not trust the agent sets the verifier's separate environment mode.
``solution/`` carries no ``solve.sh``: a generated environment has no reference winning sequence, so
Harbor's oracle agent cannot run these tasks.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from reef.core.tasks import gym_loader
from reef.core.tasks.gym_loader import environment_class_name
from reef.core.tasks.harbor import HarborTask, HarborTaskError, read_harbor_task

ACTIONS_PATH = "/workspace/actions.jsonl"
ENV_FILE = "env.py"
LOADER_FILE = "env_loader.py"
DEFAULT_IMAGE = "python:3.12-slim"
DEFAULT_MAX_TURNS = 12


class GymTaskError(HarborTaskError):
    """A task directory that is not a gym task."""


@dataclass(frozen=True)
class GymTask:
    """A gym task as the runner reads it: the Harbor task, the class, the turn limit and the seed."""

    task: HarborTask
    code: str
    max_turns: int
    seed: int


def gym_task(
    *,
    name: str,
    code: str,
    seed: int,
    max_turns: int = DEFAULT_MAX_TURNS,
    metadata: Mapping[str, object] | None = None,
    solution: Mapping[str, str] | None = None,
    source_agent_record_ids: tuple[str, ...] = (),
    image: str = DEFAULT_IMAGE,
) -> HarborTask:
    """The Harbor task that holds one environment class; ``metadata`` rides beside ``max_turns`` and ``seed``."""
    if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
        raise ValueError("max_turns must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("environment code must be non-empty text")
    class_name = environment_class_name(code)
    extra = dict(metadata or {})
    for key in ("max_turns", "seed", "env_class"):
        if key in extra:
            raise ValueError(f"metadata carries {key!r}, which the gym task writes itself")
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
        f"python3 /tests/replay.py /tests/{ENV_FILE} {ACTIONS_PATH} /logs/verifier/reward.txt {max_turns} {seed}\n"
    )
    return HarborTask(
        name=name,
        instruction=instruction,
        tests={
            "test.sh": test_script,
            "replay.py": REPLAY_SCRIPT,
            LOADER_FILE: inspect.getsource(gym_loader),
            ENV_FILE: code,
        },
        environment={"Dockerfile": f"FROM {image}\nWORKDIR /workspace\n"},
        config={
            "agent": {"timeout_sec": 600},
            "verifier": {"timeout_sec": 120},
            "environment": {"cpus": 1, "memory_mb": 1024, "storage_mb": 1024, "gpus": 0, "network_mode": "no-network"},
        },
        metadata={**extra, "env_class": class_name, "max_turns": max_turns, "seed": seed},
        solution=dict(solution or {}),
        source_agent_record_ids=source_agent_record_ids,
    )


def read_gym_task(path: Path) -> GymTask:
    """Read a task directory back as a gym task; ``GymTaskError`` when it is not one."""
    task = read_harbor_task(path)
    code = task.tests.get(ENV_FILE)
    if code is None:
        raise GymTaskError(f"{path} is not a gym task: tests/{ENV_FILE} is missing")
    if task.tests.get(LOADER_FILE) != inspect.getsource(gym_loader):
        raise GymTaskError(f"{path} was written with another loader than this reef ships")
    max_turns = task.metadata.get("max_turns")
    seed = task.metadata.get("seed")
    if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
        raise GymTaskError(f"{path} is not a gym task: metadata.max_turns must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise GymTaskError(f"{path} is not a gym task: metadata.seed must be a non-negative integer")
    try:
        environment_class_name(code)
    except ValueError as exc:
        raise GymTaskError(f"{path}: {exc}") from exc
    return GymTask(task=task, code=code, max_turns=max_turns, seed=seed)


# The verifier runs in the task's own image with /tests on sys.path, so it imports the shipped loader.
REPLAY_SCRIPT = '''"""Replay an action log through the environment class and write the episode return."""

import sys

from env_loader import episode_return, load_environment_class, make_environment, read_actions, replay


def main(environment_path, actions_path, reward_path, max_turns, seed):
    value = 0.0
    try:
        environment = make_environment(load_environment_class(environment_path), max_turns)
        environment.reset(seed=seed)
        rewards, terminated = replay(environment, read_actions(actions_path), max_turns)
        value = episode_return(rewards, terminated)
    finally:
        with open(reward_path, "w", encoding="utf-8") as handle:
            handle.write(f"{value}\\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
'''
