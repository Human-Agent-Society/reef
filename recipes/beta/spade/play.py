"""The Reasoning Agent plays one generated environment, turn by turn, in a child process.

The loop is the reference's actor loop (``spade/core/orchestrator.py``): the first observation is wrapped in
the gameplay prompt, with the hint appended for the with hint arm (``{observation}\\n\\nHINT: {hint}``),
every later observation is a plain user turn, the model's replies accumulate as assistant turns, the
reply reaches ``step`` re-boxed, and the episode ends when the game terminates or truncates, or after
``max_turns``. The game runs in a child interpreter with the shipped loader so a game that hangs, exits
or floods stdout costs one episode, never the driver.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import queue
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType

from recipes.beta.spade import environment_loader
from recipes.beta.spade.environment_loader import ERROR_REWARD, episode_return, normalized_action
from recipes.beta.spade.tasks import GAMEPLAY_PROMPT

ChatCallable = Callable[[Sequence[Mapping[str, str]]], str]

HINT_TEMPLATE = "{observation}\n\nHINT: {hint}"
STEP_TIMEOUT_S = 30.0
RESET_TIMEOUT_S = 30.0
OBSERVATION_CHARS = 20000


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
        """The episode return under the reference rule; a failed episode is a walkover, 0.0."""
        if self.failure is not None:
            return 0.0
        return episode_return(self.rewards, self.terminated)

    @property
    def actions(self) -> list[str]:
        """The action log as the verifier replays it: one reply per line, newlines folded."""
        return [" ".join(turn.reply.split()) for turn in self.turns]


def gameplay_messages(first_observation: str, hint: str | None) -> list[dict[str, str]]:
    """The first user turn: the gameplay prompt around the observation, the hint appended when given."""
    observation = first_observation
    if hint is not None:
        observation = HINT_TEMPLATE.format(observation=first_observation, hint=hint)
    return [{"role": "user", "content": GAMEPLAY_PROMPT.format(observation=observation)}]


GAME_SERVER = r'''"""Serve one generated environment over stdin and stdout, one JSON line per request and reply."""

import json
import sys

from env_loader import load_environment_class, make_environment, step_once


def main(env_path, max_turns):
    protocol = sys.stdout
    sys.stdout = sys.stderr
    environment = make_environment(load_environment_class(env_path), max_turns)
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
    try:
        main(sys.argv[1], int(sys.argv[2]))
    except BaseException as exc:
        sys.__stdout__.write(json.dumps({"error": f"{type(exc).__name__}: {exc}"[:500]}) + "\n")
        sys.__stdout__.flush()
        raise SystemExit(1)
'''


class GameProcess:
    """One environment in a child interpreter: ``reset`` and ``step`` with a timeout each; kill on exit."""

    def __init__(self, code: str, *, max_turns: int) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="spade-game-")
        root = Path(self.directory.name)
        (root / "env.py").write_text(code, encoding="utf-8")
        (root / "env_loader.py").write_text(inspect.getsource(environment_loader), encoding="utf-8")
        (root / "server.py").write_text(GAME_SERVER, encoding="utf-8")
        self.process = subprocess.Popen(
            [sys.executable, "-E", "-s", str(root / "server.py"), str(root / "env.py"), str(max_turns)],
            cwd=root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.reader = threading.Thread(target=self.read_lines, daemon=True)
        self.reader.start()

    def read_lines(self) -> None:
        if self.process.stdout is None:
            self.lines.put(None)
            return
        for line in self.process.stdout:
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
        return str(reply.get("observation", ""))[:OBSERVATION_CHARS]

    def step(self, action: str, *, timeout_s: float = STEP_TIMEOUT_S) -> tuple[str, float, bool, bool]:
        """One game step: (observation, reward, terminated, truncated), under the loader's rules."""
        reply = self.request({"op": "step", "action": action}, timeout_s)
        reward = reply.get("reward", ERROR_REWARD)
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            reward = ERROR_REWARD
        return (
            str(reply.get("observation", ""))[:OBSERVATION_CHARS],
            float(reward),
            bool(reply.get("terminated", False)),
            bool(reply.get("truncated", False)),
        )

    def close(self) -> None:
        """Kill the child and remove its directory; safe to call more than once."""
        if self.process.poll() is None:
            self.process.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            self.process.wait(timeout=5.0)
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                stream.close()
        self.directory.cleanup()

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
    step_timeout_s: float = STEP_TIMEOUT_S,
) -> Episode:
    """Play ``code`` once with the agent behind ``chat``: the reference loop, the game in a child process."""
    with GameProcess(code, max_turns=max_turns) as game:
        try:
            first_observation = game.reset(seed)
        except GameProcessError as exc:
            return Episode(first_observation="", failure=f"reset: {exc}")
        messages = gameplay_messages(first_observation, hint)
        turns: list[Turn] = []
        observation = first_observation
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
