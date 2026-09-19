"""Loss families for the Tinker backend.

Tinker computes the loss on its side: ``forward_backward`` takes the name of a
Tinker built-in loss function and, per sample, the ``loss_fn_inputs`` that
function reads. A :class:`TinkerLoss` therefore names the built-in ``loss_fn``
it trains with and shapes those inputs from Reef's captured rows. The
built-in family here uses ``importance_sampling``; a method may select
another built-in and shape its inputs the same way, or compute the loss on
the Reef host through :class:`TinkerCustomLoss`.

Methods register their loss in the shared loss-family table with
``register_loss_family_ref(name, "package.module:CLASS", backend="tinker")``;
the reference is imported the first time the name is resolved, so the recipe
package never imports this module.
"""

from __future__ import annotations

import importlib
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from reef.core.batches import TrajectoryItem
from reef.train.algos.registry import loss_family_refs
from reef.train.types.rows import policy_row_violation

#: The backend key under which methods register their loss.
BACKEND = "tinker"


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
        """The ``loss_fn_inputs`` of ``importance_sampling`` for one response-token advantage each."""
        if len(advantages) != len(self.mask):
            raise ValueError("Tinker advantages must cover every response token")
        # Position i predicts token i+1; prompt positions contribute zero loss.
        padding = len(self.tokens) - len(self.mask) - 1
        return {
            "target_tokens": list(self.tokens[1:]),
            "logprobs": [0.0] * padding + list(self.logprobs),
            "advantages": [0.0] * padding + list(advantages),
        }


class TinkerLoss(ABC):
    """One loss family on Tinker: a built-in ``loss_fn`` and the inputs it reads.

    ``base_logprobs`` holds one frozen-base log probability per response token
    of each row when ``needs_base_logprobs`` is set and the KL coefficient is
    nonzero, and is empty otherwise.
    """

    loss_fn = "importance_sampling"
    needs_base_logprobs = False

    @abstractmethod
    def inputs(
        self, rows: Sequence[TokenRow], base_logprobs: Sequence[Sequence[float]], *, kl_coef: float
    ) -> list[dict[str, list[Any]]]: ...


class ImportanceSamplingLoss(TinkerLoss):
    """The trajectory advantage on every response token the loss mask selects."""

    def inputs(
        self, rows: Sequence[TokenRow], base_logprobs: Sequence[Sequence[float]], *, kl_coef: float
    ) -> list[dict[str, list[Any]]]:
        if kl_coef:
            raise ValueError("importance_sampling does not implement a KL penalty; select a method loss family")
        return [row.inputs([row.advantage * mask for mask in row.mask]) for row in rows]


class TinkerCustomLoss(TinkerLoss):
    """A loss family computed on the Reef host through ``forward_backward_custom``.

    Tinker runs the forward pass and hands back each sample's token log
    probabilities as torch tensors. ``loss`` combines them with the rows'
    ``loss_fn_inputs`` into one loss tensor and its metrics, and Tinker runs
    the backward pass from that tensor. This path needs torch on the Reef
    host; the built-in path does not. No shipped recipe uses it yet.
    """

    loss_fn = "custom"

    @abstractmethod
    def loss(self, data: Sequence[Any], logprobs: Sequence[Any]) -> tuple[Any, Mapping[str, float]]: ...


_BUILTIN: dict[str, TinkerLoss] = {"importance_sampling": ImportanceSamplingLoss()}
_resolved: dict[str, TinkerLoss] = {}


def resolve_tinker_loss(loss_family: str) -> TinkerLoss:
    """Resolve a built-in family or import the one a method registered for Tinker."""
    loss = _BUILTIN.get(loss_family) or _resolved.get(loss_family)
    if loss is not None:
        return loss
    references = loss_family_refs(BACKEND)
    reference = references.get(loss_family)
    if reference is None:
        available = ", ".join(sorted(set(_BUILTIN) | set(references)))
        raise ValueError(f"unsupported Tinker loss family {loss_family!r}; registered: {available}")
    module_name, _, attribute = reference.partition(":")
    try:
        candidate = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"cannot import Tinker loss family {reference!r}: {exc}") from exc
    if isinstance(candidate, type) and issubclass(candidate, TinkerLoss):
        candidate = candidate()
    if not isinstance(candidate, TinkerLoss):
        raise TypeError(f"Tinker loss family {reference!r} is not a TinkerLoss")
    _resolved[loss_family] = candidate
    return candidate


def row_from_payload(value: Mapping[str, Any]) -> TokenRow:
    return TokenRow(tuple(value["tokens"]), tuple(value["mask"]), tuple(value["logprobs"]), value["advantage"])
