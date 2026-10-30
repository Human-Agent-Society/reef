"""Loss families for the Tinker backend.

Tinker computes the loss on its side: ``forward_backward`` takes the name of a
Tinker built-in loss function and, per sample, the ``loss_fn_inputs`` that
function reads. A :class:`TinkerLoss` therefore names the built-in ``loss_fn``
it trains with and shapes those inputs from Reef's captured rows. The
built-in families here are ``importance_sampling`` and ``sft`` (Tinker's
``cross_entropy``); a method may select another built-in and shape its inputs
the same way, or compute the loss on the Reef host through
:class:`TinkerCustomLoss`. :class:`ScoreCenteredLoss` is no family: a
component with score centering trains ``importance_sampling`` with it.

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
from typing import Any, Literal

from reef.core.job_data import TrajectorySample
from reef.train.algos.registry import loss_family_refs
from reef.train.tinker_backend.config import ScoreCentering
from reef.train.types.rows import policy_row_violation

#: The backend key under which methods register their loss.
BACKEND = "tinker"
#: The sampler log probability of an untrained position's head; its probability is zero in float32.
UNTRAINED_LOGPROB = -1.0e4


@dataclass(frozen=True)
class TokenRow:
    """One trajectory's training row on Tinker.

    ``tokens`` is the prompt and the response, ``mask`` has one 0/1 entry per
    response token, and ``logprobs`` one rollout log probability per response
    token, or is empty for a loss that reads none (``TinkerLoss.uses_rollout_log_probs``).
    ``advantage`` is finite; a loss without advantages ignores it.
    ``topk_indices`` and ``topk_log_probs`` are both empty, for a loss that
    reads no sampler head (``TinkerLoss.sampler_topk`` 0), or hold one row per
    response token: the sampler's most likely token IDs and their log
    probabilities, rows of equal length at each token (empty at an untrained one).
    """

    tokens: tuple[int, ...]
    mask: tuple[int, ...]
    logprobs: tuple[float, ...]
    advantage: float
    topk_indices: tuple[tuple[int, ...], ...] = ()
    topk_log_probs: tuple[tuple[float, ...], ...] = ()

    def __post_init__(self) -> None:
        violation = policy_row_violation(self.tokens, self.mask, self.logprobs or None)
        if violation:
            raise ValueError(f"invalid Tinker training row: {violation}")
        if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in self.tokens):
            raise ValueError("Tinker tokens must be non-negative integers")
        if not math.isfinite(self.advantage):
            raise ValueError("Tinker advantages must be finite")
        if (self.topk_indices or self.topk_log_probs) and not (
            len(self.topk_indices) == len(self.topk_log_probs) == len(self.mask)
            and all(
                len(ids) == len(values) for ids, values in zip(self.topk_indices, self.topk_log_probs, strict=True)
            )
        ):
            raise ValueError("invalid Tinker training row: the sampler's top-k needs one row pair per response token")

    @classmethod
    def from_item(
        cls, item: TrajectorySample, advantage: float, *, rollout_log_probs: bool = True, sampler_topk: int = 0
    ) -> TokenRow:
        """The row of ``item``; ``rollout_log_probs`` requires one rollout log probability per response token, and
        a positive ``sampler_topk`` the sampler's top-k rows of every response token, which the row carries."""
        training = item.training
        logprobs = tuple(training.get("rollout_log_probs", ())) if rollout_log_probs else ()
        mask = tuple(training.get("loss_mask", ()))
        if rollout_log_probs and len(logprobs) != len(mask):
            raise ValueError("invalid Tinker training row: rollout_log_probs must have one value per response token")
        if training.get("response_length", len(mask)) != len(mask):
            raise ValueError("Tinker response_length must match the response loss mask")
        topk_indices = tuple(tuple(row) for row in training.get("topk_indices", ())) if sampler_topk else ()
        topk_log_probs = tuple(tuple(row) for row in training.get("topk_log_probs", ())) if sampler_topk else ()
        if sampler_topk and len(topk_indices) != len(mask):
            raise ValueError(
                "score centering needs the sampler's top-k log probabilities of every response token: train task "
                "rollouts of this component, or trajectories recorded with capture_topk"
            )
        return cls(tuple(training.get("tokens", ())), mask, logprobs, advantage, topk_indices, topk_log_probs)

    def inputs(self, advantages: Sequence[float]) -> dict[str, list[Any] | MatrixInput]:
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


@dataclass(frozen=True)
class MatrixInput:
    """A two-dimensional ``loss_fn_inputs`` value, which the SDK client sends as a ``[len(rows), width]`` tensor.

    Values: ``rows`` is non-empty, of one width of at least 1, with ints for ``dtype`` ``int64`` and finite
    numbers for ``float32``. Purpose: a ``TinkerLoss`` returns it among its inputs; ``TinkerSDKClient.train``
    converts it.
    """

    rows: tuple[tuple[float, ...], ...]
    dtype: Literal["int64", "float32"]

    def __post_init__(self) -> None:
        if not self.rows or len({len(row) for row in self.rows}) != 1 or not self.rows[0]:
            raise ValueError("a Tinker matrix input needs rows of one width of at least 1")
        if self.dtype == "int64" and any(not isinstance(value, int) for row in self.rows for value in row):
            raise ValueError("an int64 Tinker matrix input holds ints")
        if self.dtype == "float32" and any(not math.isfinite(value) for row in self.rows for value in row):
            raise ValueError("a float32 Tinker matrix input holds finite numbers")


class TinkerLoss(ABC):
    """One loss family on Tinker: a built-in ``loss_fn`` and the inputs it reads.

    ``base_logprobs`` holds one frozen-base log probability per response token
    of each row when ``needs_base_logprobs`` is set and the KL coefficient is
    nonzero, and is empty otherwise. ``loss_fn_config`` goes with every
    ``forward_backward`` call; rows carry ``sampler_topk`` sampler log
    probabilities per response token when it is positive.
    """

    loss_fn = "importance_sampling"
    needs_base_logprobs = False
    #: Whether the inputs read each row's rollout log probabilities and the trajectory advantage.
    uses_rollout_log_probs = True
    uses_advantages = True

    @property
    def loss_fn_config(self) -> Mapping[str, float]:
        """The ``loss_fn_config`` of the built-in ``loss_fn``; none by default."""
        return {}

    @property
    def sampler_topk(self) -> int:
        """The sampler log probabilities per response token that the inputs read; none by default."""
        return 0

    def check_kl_coef(self, kl_coef: float) -> None:
        """Raise ``ValueError`` if this family cannot train with ``kl_coef``; any coefficient is accepted by default."""
        return

    @abstractmethod
    def inputs(
        self, rows: Sequence[TokenRow], base_logprobs: Sequence[Sequence[float]], *, kl_coef: float
    ) -> list[dict[str, list[Any] | MatrixInput]]: ...


class ImportanceSamplingLoss(TinkerLoss):
    """The trajectory advantage on every response token the loss mask selects."""

    def check_kl_coef(self, kl_coef: float) -> None:
        if kl_coef:
            raise ValueError("importance_sampling does not implement a KL penalty; set kl-coef to 0")

    def inputs(
        self, rows: Sequence[TokenRow], base_logprobs: Sequence[Sequence[float]], *, kl_coef: float
    ) -> list[dict[str, list[Any] | MatrixInput]]:
        return [row.inputs([row.advantage * mask for mask in row.mask]) for row in rows]


class ScoreCenteredLoss(TinkerLoss):
    """``importance_sampling``'s advantages on Spindle's ``cispo`` loss with score centering (arXiv:2609.20807).

    ``cispo`` weights each sampled token's score by ``clamp(p / q, weight_low, weight_high)`` instead of the full
    ratio, and score centering adds, at each trained token, the term that removes that weight's drift toward the
    sampler ``q``; Spindle computes it from the sampler's top-k at the token (``sampler_topk_token_ids`` and
    ``sampler_topk_logprobs``). Under the full ratio the term is zero, so it needs the clipped weight. Each row
    carries ``settings.top_k`` or more sampler log probabilities at every trained token; the first
    ``settings.top_k`` train.
    """

    loss_fn = "cispo"

    def __init__(self, settings: ScoreCentering) -> None:
        self.settings = settings

    @property
    def loss_fn_config(self) -> Mapping[str, float]:
        return {
            "score_centering": 1.0,
            "score_centering_min_tail_mass": self.settings.min_tail_mass,
            "clip_low_threshold": self.settings.weight_low,
            "clip_high_threshold": self.settings.weight_high,
        }

    @property
    def sampler_topk(self) -> int:
        return self.settings.top_k

    def check_kl_coef(self, kl_coef: float) -> None:
        if kl_coef:
            raise ValueError("score-centered cispo does not implement a KL penalty; set kl-coef to 0")

    def inputs(
        self, rows: Sequence[TokenRow], base_logprobs: Sequence[Sequence[float]], *, kl_coef: float
    ) -> list[dict[str, list[Any] | MatrixInput]]:
        width = self.settings.top_k
        result: list[dict[str, list[Any] | MatrixInput]] = []
        for row in rows:
            padding = len(row.tokens) - len(row.mask) - 1
            ids: list[tuple[float, ...]] = [(0,) * width] * padding
            logprobs: list[tuple[float, ...]] = [(UNTRAINED_LOGPROB,) * width] * padding
            for trained, row_ids, row_logprobs in zip(row.mask, row.topk_indices, row.topk_log_probs, strict=True):
                if not trained:
                    ids.append((0,) * width)
                    logprobs.append((UNTRAINED_LOGPROB,) * width)
                elif len(row_ids) < width:
                    raise ValueError(
                        f"score centering reads {width} sampler log probabilities per trained token; a row has "
                        f"{len(row_ids)}"
                    )
                else:
                    ids.append(tuple(row_ids[:width]))
                    logprobs.append(tuple(row_logprobs[:width]))
            result.append(
                {
                    **row.inputs([row.advantage * mask for mask in row.mask]),
                    "weights": [0.0] * padding + [float(mask) for mask in row.mask],
                    "sampler_topk_token_ids": MatrixInput(tuple(ids), "int64"),
                    "sampler_topk_logprobs": MatrixInput(tuple(logprobs), "float32"),
                }
            )
        return result


class CrossEntropyLoss(TinkerLoss):
    """Supervised fine-tuning: Tinker's ``cross_entropy`` on every response token the loss mask selects."""

    loss_fn = "cross_entropy"
    uses_rollout_log_probs = False
    uses_advantages = False

    def check_kl_coef(self, kl_coef: float) -> None:
        if kl_coef:
            raise ValueError("sft trains no KL penalty; set kl-coef to 0")

    def inputs(
        self, rows: Sequence[TokenRow], base_logprobs: Sequence[Sequence[float]], *, kl_coef: float
    ) -> list[dict[str, list[Any] | MatrixInput]]:
        # Position i predicts token i+1; prompt positions weigh 0.
        return [
            {
                "target_tokens": list(row.tokens[1:]),
                "weights": [0.0] * (len(row.tokens) - len(row.mask) - 1) + [float(mask) for mask in row.mask],
            }
            for row in rows
        ]


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


_BUILTIN: dict[str, TinkerLoss] = {"importance_sampling": ImportanceSamplingLoss(), "sft": CrossEntropyLoss()}
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
