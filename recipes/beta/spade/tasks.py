"""A generated SPADE environment as a Harbor task any Harbor agent can play inside its container.

The Environment Designer writes one Python class with the Gym interface (SPADE Sec. 3.1): a game, a
simulated tool use setting, a text world, anything a class with ``reset`` and ``step`` can hold. The task
keeps the class from the agent while letting it act: the container runs the agent as a non root user,
the class and its command live under ``/opt/env`` readable by root only, and a sudo rule lets the agent
run that one command through ``observe`` and ``act``, which replay the root held action log from the
seed, take one action when asked, and print the next observation. The verifier replays the same log
through the same loader and writes the episode return::

    gym-00012-003-deduction/
      task.toml               metadata: kind, skill, generation, step, index, difficulty, document, env_class, max_turns, seed
      instruction.md          how to act through the observe and act commands
      environment/Dockerfile  python:3.12-slim plus sudo, the agent user, the environment under /opt/env
      environment/env.py      the class, unchanged
      environment/env_loader.py
      environment/interact.py the commands' Python side
      environment/observe     the command on the agent's PATH that prints the current observation
      environment/act         the command on the agent's PATH that takes one action
      tests/env.py            the class again, for the verifier
      tests/env_loader.py
      tests/replay.py         replays /var/env/actions.jsonl and writes the episode return
      tests/test.sh
      solution/hint.txt       the privileged hint (Harbor never mounts solution/ for the agent)

Harbor's default shared verifier mode is what this layout needs: the verifier runs as root in the agent's
container and reads the log the agent user could only append to through the command. ``solution/`` holds
no ``solve.sh``: a generated environment has no reference winning sequence, so Harbor's oracle agent
cannot run these tasks. The source record of every task is the Designer's generation record, so
``split_generation`` keeps the environments of one generation call in one split. One task is one seeded
instance of the class: every play of it starts from the same hidden state, so a generation that wants the
reference's spread over instances writes one task per seed.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Sequence
from dataclasses import dataclass

from recipes.beta.spade import environment_loader
from recipes.beta.spade.environment_loader import environment_class_name
from reef.core.tasks import HarborTask, TaskSplit, split_by_source

DEFAULT_IMAGE = "python:3.12-slim"
DEFAULT_MAX_TURNS = 12
AGENT_USER = "agent"
ENVIRONMENT_DIRECTORY = "/opt/env"
LOG_PATH = "/var/env/actions.jsonl"
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
        return f"gym-{self.generation:05d}-{self.index:03d}-{self.skill}"


def environment_task(
    environment: GeneratedEnvironment, *, max_turns: int = DEFAULT_MAX_TURNS, image: str = DEFAULT_IMAGE
) -> HarborTask:
    """The Harbor task that holds ``environment``: the container commands, the verifier, the hint and the metadata."""
    if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
        raise ValueError("max_turns must be a positive integer")
    class_name = environment_class_name(environment.code)
    metadata: dict[str, object] = {
        "kind": "gym",
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
    loader = inspect.getsource(environment_loader)
    return HarborTask(
        name=environment.name,
        instruction=instruction_text(max_turns),
        tests={
            "test.sh": (
                "#!/bin/sh\nset -eu\nmkdir -p /logs/verifier\n"
                f"python3 -S /tests/replay.py /tests/env.py {LOG_PATH} /logs/verifier/reward.txt {max_turns} "
                f"{environment.seed}\n"
            ),
            "replay.py": REPLAY_SCRIPT,
            "env_loader.py": loader,
            "env.py": environment.code,
        },
        environment={
            "Dockerfile": dockerfile_text(image, max_turns=max_turns, seed=environment.seed),
            "env.py": environment.code,
            "env_loader.py": loader,
            "interact.py": INTERACT_SCRIPT,
            "observe": OBSERVE_COMMAND,
            "act": ACT_COMMAND,
        },
        config={
            "agent": {"timeout_sec": 900, "user": AGENT_USER},
            "verifier": {"timeout_sec": 120},
            "environment": {"cpus": 1, "memory_mb": 1024, "storage_mb": 1024, "gpus": 0, "network_mode": "no-network"},
        },
        metadata=metadata,
        solution={"hint.txt": environment.hint.strip() + "\n"},
        source_agent_record_ids=(environment.source_record_id,),
    )


def split_generation(tasks: Sequence[HarborTask], *, eval_fraction: float, seed: int) -> TaskSplit:
    """Split one generation's tasks so that every environment of one Designer call lands in one split."""
    names = [task.name for task in tasks]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"tasks share a name: {', '.join(repeated)}")
    return split_by_source(
        {task.name: task.source_agent_record_ids for task in tasks}, eval_fraction=eval_fraction, seed=seed
    )


def instruction_text(max_turns: int) -> str:
    return (
        "You are acting in an interactive environment. Reach its goal with valid actions.\n"
        "\n"
        "The environment runs behind two commands. Run `observe` to see the current observation. To act, run "
        "`act '<your action>'` with the action inside \\boxed{}, for example: act '\\boxed{north}'. Each act is one "
        f"turn; the episode ends when the environment terminates or after {max_turns} turns. Reason step by step "
        "before each turn. You never see the environment's code; the verifier scores the last step of a terminated "
        "episode from the actions you took.\n"
    )


def dockerfile_text(image: str, *, max_turns: int, seed: int) -> str:
    """The agent's image: sudo, a non root agent user, and the environment readable by root only."""
    return (
        f"FROM {image}\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends sudo && rm -rf /var/lib/apt/lists/* \\\n"
        f" && useradd --create-home --shell /bin/bash {AGENT_USER} \\\n"
        f" && mkdir -p {ENVIRONMENT_DIRECTORY} /var/env /workspace && chmod 700 {ENVIRONMENT_DIRECTORY} /var/env \\\n"
        f" && chown {AGENT_USER}:{AGENT_USER} /workspace\n"
        f"COPY env.py env_loader.py interact.py {ENVIRONMENT_DIRECTORY}/\n"
        "COPY observe act /usr/local/bin/\n"
        f'RUN printf \'{{"seed": {seed}, "max_turns": {max_turns}}}\\n\' > {ENVIRONMENT_DIRECTORY}/config.json \\\n'
        f" && chmod 600 {ENVIRONMENT_DIRECTORY}/* && chmod 755 /usr/local/bin/observe /usr/local/bin/act \\\n"
        # Two specs: sudo's trailing wildcard needs at least one argument, and observe passes none.
        f" && echo '{AGENT_USER} ALL=(root) NOPASSWD: /usr/local/bin/python3 -S {ENVIRONMENT_DIRECTORY}/interact.py, "
        f"/usr/local/bin/python3 -S {ENVIRONMENT_DIRECTORY}/interact.py *' > /etc/sudoers.d/environment \\\n"
        " && chmod 440 /etc/sudoers.d/environment\n"
        "WORKDIR /workspace\n"
    )


OBSERVE_COMMAND = f"#!/bin/sh\nexec sudo -n /usr/local/bin/python3 -S {ENVIRONMENT_DIRECTORY}/interact.py\n"
ACT_COMMAND = (
    "#!/bin/sh\n"
    'if [ "$#" -eq 0 ]; then echo "usage: act \'<your action>\'" >&2; exit 2; fi\n'
    f'exec sudo -n /usr/local/bin/python3 -S {ENVIRONMENT_DIRECTORY}/interact.py "$@"\n'
)


#: The commands' Python side: replays the log from the seed, takes one action when given, prints the observation.
#: The paths are environment variables so the same script runs in a test on the host.
INTERACT_SCRIPT = r'''"""observe and act: the log replayed from the seed each call, one action taken when given, the observation printed."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env_loader import episode_return, load_environment_class, make_environment, normalized_action, read_actions, step_once

ENVIRONMENT_DIRECTORY = os.environ.get("ENVIRONMENT_DIRECTORY", "/opt/env")
LOG_PATH = os.environ.get("ENVIRONMENT_LOG", "/var/env/actions.jsonl")


def main(argv):
    with open(os.path.join(ENVIRONMENT_DIRECTORY, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    seed, max_turns = int(config["seed"]), int(config["max_turns"])
    environment = make_environment(load_environment_class(os.path.join(ENVIRONMENT_DIRECTORY, "env.py")), max_turns)
    first, _ = environment.reset(seed=seed)
    observation = str(first)
    actions = read_actions(LOG_PATH)
    rewards = []
    terminated = False
    truncated = False
    for action in actions[:max_turns]:
        observation, reward, terminated, truncated = step_once(environment, normalized_action(action))
        rewards.append(reward)
        if terminated or truncated:
            break
    is_over = terminated or truncated or len(actions) >= max_turns
    if not argv:
        print(observation)
        if is_over:
            print(f"The episode is over. Return: {episode_return(rewards, terminated)}")
        else:
            print(f"Turns left: {max_turns - len(actions)}")
        return 0
    if is_over:
        print("The episode is over; no more actions are taken.")
        return 0
    action = " ".join(argv)
    observation, reward, terminated, truncated = step_once(environment, normalized_action(action))
    rewards.append(reward)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(action) + "\n")
    print(observation)
    if terminated or truncated:
        print(f"The episode is over. Return: {episode_return(rewards, terminated)}")
    else:
        print(f"Turns left: {max_turns - len(actions) - 1}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
'''

#: The verifier runs in the task's own image with /tests on sys.path, so it imports the shipped loader.
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
