"""What the process that starts a native episode sets and the tree cannot: the episode's token budget.

``run_episode`` passes the budget as ``REEF_EPISODE_TOKENS`` from ``evolution.episode_tokens``; no node renders it,
so a candidate tree cannot raise the budget it is judged under.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, field

from reef.harness.episodes.executor import EPISODE_TOKENS_ENV


def episode_token_limit(environ: Mapping[str, str]) -> int | None:
    """The token budget ``REEF_EPISODE_TOKENS`` names: None when unset, else a positive integer."""
    text = environ.get(EPISODE_TOKENS_ENV)
    if text is None:
        return None
    if not (text.isascii() and text.isdigit()) or int(text) <= 0:
        raise ValueError(f"{EPISODE_TOKENS_ENV}={text!r} must be a positive integer of tokens")
    return int(text)


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


@dataclass(frozen=True)
class EpisodeControl:
    """What the process that starts an episode sets and the tree cannot; the default sets no limit."""

    budget: TeamBudget = field(default_factory=lambda: TeamBudget(None))
