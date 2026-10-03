"""What the process that starts a native episode sets and the tree cannot: the budget, the stop, where team git runs.

``run_episode`` passes the budget as ``REEF_EPISODE_TOKENS`` from ``evolution.episode_tokens``; no node renders it,
so a candidate tree cannot raise the budget it is judged under. The time budget comes the same way, as
``REEF_EPISODE_SECONDS`` from ``evolution.episode_seconds``: the deadline it sets ends the episode through the stop
flag, with reason ``deadline``. The budget and the stop flag are shared by every agent turn of the episode, the
members of a team stage included, and read before each step. The Harbor agent
(``reef.harness.runners.native.harbor``) also sets the reply budget of a model call and the retry policy.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from reef.harness.episodes.executor import EPISODE_SECONDS_ENV, EPISODE_TOKENS_ENV
from reef.harness.runners.native.workspaces import CommandRunner, HostCommandRunner


def episode_token_limit(environ: Mapping[str, str]) -> int | None:
    """The token budget ``REEF_EPISODE_TOKENS`` names: None when unset, else a positive integer."""
    text = environ.get(EPISODE_TOKENS_ENV)
    if text is None:
        return None
    if not (text.isascii() and text.isdigit()) or int(text) <= 0:
        raise ValueError(f"{EPISODE_TOKENS_ENV}={text!r} must be a positive integer of tokens")
    return int(text)


def episode_seconds_limit(environ: Mapping[str, str]) -> float | None:
    """The time budget ``REEF_EPISODE_SECONDS`` names, in seconds: None when unset, else a positive number."""
    text = environ.get(EPISODE_SECONDS_ENV)
    if text is None:
        return None
    try:
        seconds = float(text)
    except ValueError:
        seconds = math.nan
    if not seconds > 0 or math.isinf(seconds):
        raise ValueError(f"{EPISODE_SECONDS_ENV}={text!r} must be a positive number of seconds")
    return seconds


def deadline_after(seconds_limit: float | None) -> float | None:
    """The episode's deadline on the monotonic clock, ``seconds_limit`` seconds from now; None without a limit."""
    return None if seconds_limit is None else time.monotonic() + seconds_limit


class TeamBudget:
    """One token budget per episode, spent by every agent turn of it; safe to spend from any thread.

    ``token_limit`` None counts the tokens and never ends a turn."""

    def __init__(self, token_limit: int | None) -> None:
        self.token_limit = token_limit
        self.input_tokens = 0
        self.output_tokens = 0
        self.spend_lock = threading.Lock()

    def spend(self, input_tokens: int, output_tokens: int) -> None:
        with self.spend_lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens

    @property
    def spent_tokens(self) -> int:
        with self.spend_lock:
            return self.input_tokens + self.output_tokens

    @property
    def is_spent(self) -> bool:
        return self.token_limit is not None and self.spent_tokens >= self.token_limit


class EpisodeStop:
    """Set once by whoever runs the episode; every run reads it before its next step and ends its turn there."""

    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.reason = ""

    def set(self, reason: str) -> None:
        if not self.stop_event.is_set():
            self.reason = reason
            self.stop_event.set()

    @property
    def is_set(self) -> bool:
        return self.stop_event.is_set()

    def wait(self, timeout_seconds: float) -> bool:
        """Wait up to ``timeout_seconds``; whether the flag is set, so a wait between retries ends at the stop."""
        return self.stop_event.wait(timeout_seconds)


@dataclass(frozen=True)
class RequestPolicy:
    """How a model stage treats a failed call; the default leaves it to the tree's request_error hooks."""

    #: Retry a transient failure (no answer, or 408, 425, 429, 5xx) until the stop flag is set, whatever the hooks
    #: say: a long Harbor task outlives an endpoint outage, and the stop is the episode's own end.
    is_retry_until_stopped: bool = False


@dataclass(frozen=True)
class EpisodeControl:
    """What the process that starts an episode sets and the tree cannot; the default sets no limit."""

    budget: TeamBudget = field(default_factory=lambda: TeamBudget(None))
    stop: EpisodeStop = field(default_factory=EpisodeStop)
    #: Where team stages run git, and where their git directory and member clones live; None is ``.reef/team``
    #: under the main worktree, which git never tracks.
    command_runner: CommandRunner = field(default_factory=HostCommandRunner)
    team_path: PurePosixPath | None = None
    request_policy: RequestPolicy = field(default_factory=RequestPolicy)
    #: Tokens one model call may generate; None is the loop's own cap, ``MAX_COMPLETION_TOKENS``.
    max_completion_tokens: int | None = None
    #: When the episode must end, on the monotonic clock (``deadline_after``); None is no time budget.
    deadline: float | None = None

    @property
    def is_ending(self) -> bool:
        """Whether the stop flag is set or the budget spent, so every turn ends at its next step."""
        return self.stop.is_set or self.budget.is_spent

    def seconds_left(self) -> float | None:
        """Seconds to the deadline, 0 once it passed; None without one."""
        return None if self.deadline is None else max(0.0, self.deadline - time.monotonic())

    def start_deadline_timer(self) -> threading.Timer | None:
        """A daemon timer that sets the stop flag with reason ``deadline`` when the time budget runs out, so every
        turn ends at its next step and a team stage merges; None without a deadline. The caller cancels it once
        the turn ended."""
        left = self.seconds_left()
        if left is None:
            return None
        timer = threading.Timer(left, self.stop.set, ("deadline",))
        timer.daemon = True
        timer.start()
        return timer
