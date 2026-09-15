"""Play one Gym style environment turn by turn: the game in a child interpreter, the agent behind a chat callable.

The loop is SPADE's actor loop (``spade/core/orchestrator.py``): the first observation is wrapped in the
gameplay prompt, with the hint appended for a with hint arm (``{observation}\\n\\nHINT: {hint}``), every
later observation is a plain user turn, the model's replies accumulate as assistant turns, the reply
reaches ``step`` boxed again, and the episode ends when the game terminates or truncates, or after
``max_turns``. The game runs in a child interpreter with the shipped loader and no site packages, in its
own process group, and talks over a pipe of its own, so a game that hangs, exits, forks or floods its
standard streams costs one episode, never the caller.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType

from reef.core.tasks import gym_loader
from reef.core.tasks.gym_loader import episode_return, normalized_action

ChatCallable = Callable[[Sequence[Mapping[str, str]]], str]

GAMEPLAY_PROMPT = (
    "You are playing a language game. Make valid actions to win.\n"
    "Observation: {observation}\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}."
)
HINT_TEMPLATE = "{observation}\n\nHINT: {hint}"
STEP_TIMEOUT_S = 30.0
RESET_TIMEOUT_S = 30.0
CLOSE_TIMEOUT_S = 5.0
OBSERVATION_CHARS = 20000
QUEUED_LINES = 64


class GameProcessError(RuntimeError):
    """The child that runs the game died, hung or answered with something that is not a game event."""


@dataclass(frozen=True)
class Turn:
    """One exchange: what the agent saw, what it replied, what reached the game and what came back."""

    observation: str
    reply: str
    action: str
    reward: float
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class Episode:
    """One play of one environment; ``failure`` names why it ended early when it did."""

    first_observation: str
    turns: tuple[Turn, ...] = field(default_factory=tuple)
    failure: str | None = None

    @property
    def rewards(self) -> list[float]:
        return [turn.reward for turn in self.turns]

    @property
    def terminated(self) -> bool:
        return bool(self.turns) and self.turns[-1].terminated

    @property
    def return_value(self) -> float:
        """The episode return under the loader's rule; a failed episode is a walkover, 0.0."""
        if self.failure is not None:
            return 0.0
        return episode_return(self.rewards, self.terminated)

    @property
    def actions(self) -> list[str]:
        """The action log as the verifier replays it: every reply as written."""
        return [turn.reply for turn in self.turns]


def gameplay_messages(
    first_observation: str, hint: str | None, system_prompt: str | None = None
) -> list[dict[str, str]]:
    """The opening messages: an optional system prompt, then the gameplay prompt around the first observation."""
    observation = first_observation if hint is None else HINT_TEMPLATE.format(observation=first_observation, hint=hint)
    messages: list[dict[str, str]] = []
    if system_prompt is not None and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": GAMEPLAY_PROMPT.format(observation=observation)})
    return messages


GAME_SERVER = r'''"""Serve one environment class over stdin and a pipe of its own, one JSON line per request and reply."""

import json
import os
import sys

from env_loader import load_environment_class, make_environment, step_once


def main(environment_path, max_turns, protocol):
    environment = make_environment(load_environment_class(environment_path), max_turns)
    for line in sys.stdin:
        request = json.loads(line)
        if request["op"] == "reset":
            observation, _ = environment.reset(seed=request["seed"])
            reply = {"observation": str(observation)}
        elif request["op"] == "step":
            observation, reward, terminated, truncated = step_once(environment, request["action"])
            reply = {"observation": observation, "reward": reward, "terminated": terminated, "truncated": truncated}
        else:
            reply = {"error": f"unknown op {request['op']!r}"}
        protocol.write(json.dumps(reply) + "\n")
        protocol.flush()


if __name__ == "__main__":
    channel = os.fdopen(int(sys.argv[3]), "w", encoding="utf-8")
    try:
        main(sys.argv[1], int(sys.argv[2]), channel)
    except BaseException as exc:
        channel.write(json.dumps({"error": f"{type(exc).__name__}: {exc}"[:500]}) + "\n")
        channel.flush()
        raise SystemExit(1)
'''


class GameProcess:
    """One environment in a child interpreter: ``reset`` and ``step`` with a timeout each; kill the group on exit."""

    def __init__(self, code: str, *, max_turns: int) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(prefix="reef-gym-")
        root = Path(self.temporary_directory.name)
        (root / "env.py").write_text(code, encoding="utf-8")
        (root / "env_loader.py").write_text(inspect.getsource(gym_loader), encoding="utf-8")
        (root / "server.py").write_text(GAME_SERVER, encoding="utf-8")
        read_end, write_end = os.pipe()
        try:
            # -E -s -S: no PYTHONPATH, no user site, no site packages, the way the task image has none.
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-E",
                    "-s",
                    "-S",
                    str(root / "server.py"),
                    str(root / "env.py"),
                    str(max_turns),
                    str(write_end),
                ],
                cwd=root,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                pass_fds=(write_end,),
                start_new_session=True,
                text=True,
                encoding="utf-8",
            )
        except OSError:
            os.close(read_end)
            os.close(write_end)
            self.temporary_directory.cleanup()
            raise
        os.close(write_end)
        self.protocol = os.fdopen(read_end, "r", encoding="utf-8", errors="replace")
        self.lines: queue.Queue[str | None] = queue.Queue(maxsize=QUEUED_LINES)
        self.reader = threading.Thread(target=self.read_lines, daemon=True)
        self.reader.start()

    def read_lines(self) -> None:
        with contextlib.suppress(ValueError, OSError):
            for line in self.protocol:
                self.lines.put(line)
        self.lines.put(None)

    def request(self, payload: Mapping[str, object], timeout_s: float) -> dict[str, object]:
        if self.process.stdin is None or self.process.poll() is not None:
            raise GameProcessError("the game process is not running")
        try:
            self.process.stdin.write(json.dumps(payload) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise GameProcessError(f"the game process closed its input: {exc}") from exc
        try:
            line = self.lines.get(timeout=timeout_s)
        except queue.Empty:
            self.close()
            raise GameProcessError(f"the game did not answer within {timeout_s:g} s") from None
        if line is None:
            raise GameProcessError("the game process exited")
        try:
            reply = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GameProcessError(f"the game process wrote something that is not a reply: {line[:200]!r}") from exc
        if not isinstance(reply, dict):
            raise GameProcessError("the game process wrote something that is not a reply")
        if "error" in reply:
            raise GameProcessError(str(reply["error"]))
        return reply

    def reset(self, seed: int, *, timeout_s: float = RESET_TIMEOUT_S) -> str:
        """The first observation; ``GameProcessError`` when the game cannot start."""
        reply = self.request({"op": "reset", "seed": seed}, timeout_s)
        try:
            return str(reply["observation"])[:OBSERVATION_CHARS]
        except KeyError as exc:
            raise GameProcessError("the game process wrote a reply without an observation") from exc

    def step(self, action: str, *, timeout_s: float = STEP_TIMEOUT_S) -> tuple[str, float, bool, bool]:
        """One game step: (observation, reward, terminated, truncated), under the loader's rules."""
        reply = self.request({"op": "step", "action": action}, timeout_s)
        try:
            observation, reward, terminated, truncated = (
                reply["observation"],
                reply["reward"],
                reply["terminated"],
                reply["truncated"],
            )
        except KeyError as exc:
            raise GameProcessError(f"the game process wrote a reply without {exc}") from exc
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise GameProcessError(f"the game process wrote a reward that is not a number: {reward!r}")
        return str(observation)[:OBSERVATION_CHARS], float(reward), bool(terminated), bool(truncated)

    def close(self) -> None:
        """Kill the child's whole process group and remove its directory; safe to call more than once."""
        if self.process.poll() is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(self.process.pid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            self.process.wait(timeout=CLOSE_TIMEOUT_S)
        if self.process.stdin is not None:
            self.process.stdin.close()
        # A grandchild that kept the pipe open would block a close of the protocol; the reader keeps it instead.
        self.reader.join(timeout=1.0)
        if not self.reader.is_alive():
            self.protocol.close()
        self.temporary_directory.cleanup()

    def __enter__(self) -> GameProcess:
        return self

    def __exit__(
        self, kind: type[BaseException] | None, value: BaseException | None, trace: TracebackType | None
    ) -> None:
        self.close()


def play_episode(
    code: str,
    chat: ChatCallable,
    *,
    seed: int,
    max_turns: int,
    hint: str | None = None,
    system_prompt: str | None = None,
    step_timeout_s: float = STEP_TIMEOUT_S,
) -> Episode:
    """Play ``code`` once with the agent behind ``chat``: the actor loop, the game in a child process."""
    with GameProcess(code, max_turns=max_turns) as game:
        try:
            first_observation = game.reset(seed)
        except GameProcessError as exc:
            return Episode(first_observation="", failure=f"reset: {exc}")
        messages = gameplay_messages(first_observation, hint, system_prompt)
        turns: list[Turn] = []
        for _ in range(max_turns):
            try:
                reply = chat(messages)
            except Exception as exc:
                return Episode(first_observation=first_observation, turns=tuple(turns), failure=f"model: {exc}")
            if not isinstance(reply, str):
                return Episode(
                    first_observation=first_observation, turns=tuple(turns), failure="model: reply is not text"
                )
            messages.append({"role": "assistant", "content": reply})
            action = normalized_action(reply)
            try:
                observation, reward, terminated, truncated = game.step(action, timeout_s=step_timeout_s)
            except GameProcessError as exc:
                return Episode(first_observation=first_observation, turns=tuple(turns), failure=f"step: {exc}")
            turns.append(
                Turn(
                    observation=observation,
                    reply=reply,
                    action=action,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                )
            )
            if terminated or truncated:
                break
            messages.append({"role": "user", "content": observation})
    return Episode(first_observation=first_observation, turns=tuple(turns))
