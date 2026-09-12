"""What the MLX runtime and its engine exchange: plain data and contracts, no MLX.

They live apart from the engine so that the runtime's contract with Reef —
preparation, candidate export, activation, rollback, serving — can be
exercised against a fake engine on a machine without the optional ``mlx``
extra. The engine imports them from here; MLX is only reached when an engine
is built.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class GenerationListener(Protocol):
    """Who a streaming generation reports to while it runs.

    ``emit`` is called on the engine thread with each piece of text as the
    detokenizer settles it, so it must only hand the piece off. ``cancelled``
    is polled once per token; answering ``True`` ends the generation there.
    """

    def emit(self, piece: str) -> None: ...

    def cancelled(self) -> bool: ...


@dataclass(frozen=True)
class TrainingRow:
    """One reserved sample, already carrying its algorithm-supplied advantage."""

    tokens: tuple[int, ...]
    loss_mask: tuple[int, ...]
    rollout_log_probs: tuple[float, ...]
    #: One advantage per response token. TTT-Discover starts from a single
    #: trajectory advantage and then adds a per-token frozen-base KL term, so
    #: the row carries the expanded vector rather than a scalar.
    advantages: tuple[float, ...]

    def __post_init__(self) -> None:
        response_length = len(self.loss_mask)
        if response_length == 0 or len(self.tokens) <= response_length:
            raise ValueError("a training row needs at least one prompt token and one response token")
        if len(self.rollout_log_probs) != response_length:
            raise ValueError("rollout_log_probs must cover exactly the response tokens")
        if len(self.advantages) != response_length:
            raise ValueError("advantages must cover exactly the response tokens")


@dataclass(frozen=True)
class TeacherCandidate:
    """One hindsight hint, already rendered to the tokens the teacher scores.

    ``tokens`` is the hint-enhanced prompt followed by the response the policy
    actually produced, so the teacher answers "what would the base model have
    said here, had the user asked for this up front".
    """

    hint: str
    tokens: tuple[int, ...]


@dataclass(frozen=True)
class DistillationRow:
    """One judged turn: the sampled tokens, the candidates, and the hints.

    Extends what a policy-gradient row carries with the two things a
    distillation term needs and nothing else can reconstruct — the candidate
    set the policy considered at each step, and the teacher sequences the
    judge's hints produced.
    """

    tokens: tuple[int, ...]
    loss_mask: tuple[int, ...]
    rollout_log_probs: tuple[float, ...]
    reward: float
    topk_indices: tuple[tuple[int, ...], ...]
    topk_log_probs: tuple[tuple[float, ...], ...]
    candidates: tuple[TeacherCandidate, ...]

    def __post_init__(self) -> None:
        response_length = len(self.loss_mask)
        if response_length == 0 or len(self.tokens) <= response_length:
            raise ValueError("a distillation row needs at least one prompt token and one response token")
        if len(self.topk_indices) != response_length or len(self.topk_log_probs) != response_length:
            raise ValueError("captured candidates must cover exactly the response tokens")
        if not self.candidates:
            raise ValueError("a distillation row needs at least one teacher candidate")
        for candidate in self.candidates:
            if len(candidate.tokens) <= response_length:
                raise ValueError("a teacher sequence must be its prompt plus the whole response")


__all__ = ["DistillationRow", "GenerationListener", "TeacherCandidate", "TrainingRow"]
