"""``reef-gym``: play a gym task with the served model under the rendered tree, and write the trajectory.

The episode form of the ``gym`` adapter. The runner reads the tree from ``REEF_GYM_DIR`` (``RULES.md``
and every ``skills/*/SKILL.md`` become the system prompt, ``models.json`` the endpoint, ``config.json``
the optional ``temperature``, ``max_tokens`` and ``timeout_s``), reads the task directory the prompt
names through :func:`reef.core.tasks.gym.read_gym_task`, plays it once with
:func:`reef.harness.runners.gym.play.play_episode`, writes the action log into the working directory
and every turn plus the episode return under ``REEF_GYM_SESSION_DIR`` as ``session.jsonl`` for the
``gym-jsonl`` reader. The exit code is 1 when the episode failed before the game ended.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from reef.core.tasks.gym import read_gym_task
from reef.core.tasks.gym_loader import write_actions
from reef.core.tasks.harbor import HarborTaskError
from reef.harness.episodes.model_binding import ModelBinding
from reef.harness.episodes.run import EpisodeResult
from reef.harness.runners.gym.play import Episode, play_episode

TREE_DIR_ENV = "REEF_GYM_DIR"
SESSION_DIR_ENV = "REEF_GYM_SESSION_DIR"
RULES_FILE = "RULES.md"
SKILLS_DIR = "skills"
MODELS_FILE = "models.json"
CONFIG_FILE = "config.json"
SESSION_FILE = "session.jsonl"
ACTIONS_FILE = "actions.jsonl"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT_S = 600.0


def system_prompt_from(root: Path) -> str | None:
    """The rules text and every skill under ``root``, as one system prompt; None when the tree has neither."""
    parts: list[str] = []
    rules = root / RULES_FILE
    if rules.is_file():
        text = rules.read_text(encoding="utf-8").strip()
        if text:
            parts.append(text)
    for skill in sorted((root / SKILLS_DIR).glob("*/SKILL.md")):
        text = skill.read_text(encoding="utf-8").strip()
        if text:
            parts.append(f"# Skill: {skill.parent.name}\n\n{text}")
    if not parts:
        return None
    return "\n\n".join(parts)


def binding_from(models_path: Path) -> ModelBinding:
    """The served model as ``model_binding`` rendered it into ``models.json``."""
    data = json.loads(models_path.read_text(encoding="utf-8"))
    return ModelBinding(
        base_url=str(data["base_url"]),
        model=str(data["model"]),
        api_key=str(data.get("api_key") or ""),
        api=str(data.get("api") or "openai"),
    )


def chat_settings_from(config_path: Path) -> dict[str, float | int]:
    """``temperature``, ``max_tokens`` and ``timeout_s`` from ``config.json`` when they are numbers, else defaults."""
    settings: dict[str, float | int] = {"max_tokens": DEFAULT_MAX_TOKENS, "timeout_s": DEFAULT_TIMEOUT_S}
    if not config_path.is_file():
        return settings
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        return settings
    max_tokens = data.get("max_tokens")
    if isinstance(max_tokens, int) and not isinstance(max_tokens, bool) and max_tokens > 0:
        settings["max_tokens"] = max_tokens
    timeout_s = data.get("timeout_s")
    if isinstance(timeout_s, (int, float)) and not isinstance(timeout_s, bool) and timeout_s > 0:
        settings["timeout_s"] = float(timeout_s)
    temperature = data.get("temperature")
    if isinstance(temperature, (int, float)) and not isinstance(temperature, bool) and 0 <= temperature <= 2:
        settings["temperature"] = float(temperature)
    return settings


def verifier_event(task: str, episode: Episode | None, *, name: str | None, failure: str | None) -> dict[str, object]:
    """The event the scorer reads: the task as named, the episode return and how the play ended."""
    if episode is None:
        return {
            "type": "verifier",
            "task": task,
            "name": name,
            "reward": 0.0,
            "rewards": [],
            "terminated": False,
            "turns": 0,
            "failure": failure,
        }
    return {
        "type": "verifier",
        "task": task,
        "name": name,
        "reward": episode.return_value,
        "rewards": episode.rewards,
        "terminated": episode.terminated,
        "turns": len(episode.turns),
        "failure": episode.failure,
    }


def write_session(path: Path, events: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")


def run(task: str, root: Path, session_dir: Path, workspace: Path) -> int:
    """Play ``task`` under the tree at ``root``; the session goes under ``session_dir``, the action log into ``workspace``."""
    session = session_dir / SESSION_FILE
    try:
        gym = read_gym_task(Path(task))
    except HarborTaskError as exc:
        write_session(session, [verifier_event(task, None, name=None, failure=f"task: {exc}")])
        print(f"reef-gym: {exc}", file=sys.stderr)
        return 1
    binding = binding_from(root / MODELS_FILE)
    settings = chat_settings_from(root / CONFIG_FILE)
    timeout_s = float(settings["timeout_s"])
    params = {key: value for key, value in settings.items() if key != "timeout_s"}

    def chat(messages: Sequence[Mapping[str, str]]) -> str:
        return binding.chat(messages, timeout_s=timeout_s, **params)

    system_prompt = system_prompt_from(root)
    episode = play_episode(gym.code, chat, seed=gym.seed, max_turns=gym.max_turns, system_prompt=system_prompt)
    events: list[dict[str, object]] = [
        {
            "type": "opening",
            "task": task,
            "name": gym.task.name,
            "model": binding.model,
            "base_url": binding.base_url,
            "system_prompt": system_prompt,
            "first_observation": episode.first_observation,
        }
    ]
    for index, turn in enumerate(episode.turns):
        events.append(
            {
                "type": "step",
                "turn": index,
                "observation": turn.observation,
                "reply": turn.reply,
                "action": turn.action,
                "reward": turn.reward,
                "terminated": turn.terminated,
                "truncated": turn.truncated,
            }
        )
    events.append(verifier_event(task, episode, name=gym.task.name, failure=episode.failure))
    write_session(session, events)
    write_actions(str(workspace / ACTIONS_FILE), episode.actions)
    if episode.failure is not None:
        print(f"reef-gym: {episode.failure}", file=sys.stderr)
        return 1
    return 0


def evaluate(task: str, result: EpisodeResult) -> float:
    """The ``evolution.evaluate`` for gym tasks: the episode return the runner recorded for ``task``."""
    rows = [event for event in result.trajectory if event.get("type") == "verifier"]
    if len(rows) != 1:
        # The runner never reached the end of the play: a walkover, with the exit stage recorded by the gate.
        return 0.0
    row = rows[0]
    if row.get("task") != task:
        raise ValueError(f"the verifier event names task {row.get('task')!r}, not {task!r}")
    reward = row.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
        raise ValueError(f"the verifier event carries no finite reward: {reward!r}")
    return float(reward)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reef-gym", description="Play a gym task with the served model.")
    parser.add_argument("--task", required=True, help="a gym task directory")
    arguments = parser.parse_args(argv)
    root = Path(os.environ.get(TREE_DIR_ENV) or "gym")
    session_dir = Path(os.environ.get(SESSION_DIR_ENV) or root / "sessions")
    return run(arguments.task, root, session_dir, Path.cwd())


if __name__ == "__main__":
    sys.exit(main())
