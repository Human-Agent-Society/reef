"""Backend data shaping; methods register their own Tinker loss adapters."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from reef.core.batches import TrajectoryItem
from reef.train.types.rows import policy_row_violation


@dataclass(frozen=True)
class TokenRow:
    tokens: tuple[int, ...]
    mask: tuple[int, ...]
    logprobs: tuple[float, ...]
    advantage: float

    @classmethod
    def from_item(cls, item: TrajectoryItem, advantage: float) -> TokenRow:
        training = item.training
        row = cls(
            tuple(training.get("tokens", ())),
            tuple(training.get("loss_mask", ())),
            tuple(training.get("rollout_log_probs", ())),
            advantage,
        )
        violation = policy_row_violation(row.tokens, row.mask, row.logprobs)
        if violation:
            raise ValueError(f"invalid Tinker training row: {violation}")
        if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in row.tokens):
            raise ValueError("Tinker tokens must be non-negative integers")
        if training.get("response_length", len(row.mask)) != len(row.mask):
            raise ValueError("Tinker response_length must match the response loss mask")
        if not math.isfinite(advantage):
            raise ValueError("Tinker advantages must be finite")
        return row

    def inputs(self, advantages: Sequence[float]) -> dict[str, list[Any]]:
        # Position i predicts token i+1; prompt positions contribute zero loss.
        padding = len(self.tokens) - len(self.mask) - 1
        return {
            "target_tokens": list(self.tokens[1:]),
            "logprobs": [0.0] * padding + list(self.logprobs),
            "advantages": [0.0] * padding + list(advantages),
        }


class TinkerLoss(ABC):
    """Shape one optimizer batch without importing SDK or tensor dependencies."""

    loss_fn = "importance_sampling"
    needs_base_logprobs = False

    @abstractmethod
    def inputs(
        self, rows: Sequence[TokenRow], base_logprobs: Sequence[Sequence[float]], *, kl_coef: float
    ) -> list[dict[str, list[Any]]]: ...


class ImportanceSamplingLoss(TinkerLoss):
    def inputs(
        self, rows: Sequence[TokenRow], base_logprobs: Sequence[Sequence[float]], *, kl_coef: float
    ) -> list[dict[str, list[Any]]]:
        if kl_coef:
            raise ValueError("importance_sampling does not implement a KL penalty; select a method loss adapter")
        return [row.inputs([row.advantage * mask for mask in row.mask]) for row in rows]


_LOSSES: dict[str, TinkerLoss] = {"importance_sampling": ImportanceSamplingLoss()}


def register_tinker_loss(name: str, loss: TinkerLoss) -> None:
    if not name or not isinstance(loss, TinkerLoss):
        raise TypeError("a Tinker loss must have a name and implement TinkerLoss")
    if name in _LOSSES and _LOSSES[name] is not loss:
        raise ValueError(f"Tinker loss {name!r} is already registered")
    _LOSSES[name] = loss


def resolve_tinker_loss(name: str) -> TinkerLoss:
    try:
        return _LOSSES[name]
    except KeyError as exc:
        raise ValueError(f"unsupported Tinker loss family {name!r}; registered: {', '.join(sorted(_LOSSES))}") from exc


def row_from_payload(value: Mapping[str, Any]) -> TokenRow:
    return TokenRow(tuple(value["tokens"]), tuple(value["mask"]), tuple(value["logprobs"]), value["advantage"])
