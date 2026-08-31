"""Structural validity of one tokenized policy row.

The Slime bridge rejects malformed rows at train time; reported-feedback processors
refuse the same rows at ingest time so a bad rollout is dropped as terminal
instead of failing a whole training step. Both sides call this one predicate
so the two judgments can never drift apart.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real


def policy_row_violation(
    tokens: Sequence[int],
    loss_mask: Sequence[int],
    rollout_log_probs: Sequence[float] | None,
    *,
    action_mask: Sequence[int] | None = None,
) -> str | None:
    """Describe why a policy row is untrainable, or ``None`` when it is valid.

    ``rollout_log_probs=None`` skips the length and value checks for loss
    families that accept rows without a behaviour proxy (a supervised
    objective).
    ``action_mask=None`` skips the action-alignment checks.
    """
    response_length = len(loss_mask)
    if response_length == 0:
        return "has an empty loss_mask; the inference harness must return response training tensors"
    if len(tokens) <= response_length:
        return f"tokens must contain at least one prompt token plus {response_length} response tokens"
    # Truthiness (not sum) so a malformed mask value falls through to the
    # 0/1 value check below instead of failing here with a TypeError.
    if not any(loss_mask):
        return "loss_mask must select at least one response token"
    if any(value not in (0, 1) or isinstance(value, bool) for value in loss_mask):
        return "loss_mask must contain only 0 or 1"
    if action_mask is not None:
        if len(action_mask) != response_length:
            return f"action_mask length {len(action_mask)} does not match response length {response_length}"
        if not any(action_mask):
            return "action_mask must select at least one action token"
        # Every trained token must be an action token: a method that only
        # propagates advantage over actions would otherwise train a token
        # marked observation with no advantage signal behind it.
        if any(loss and not action for loss, action in zip(loss_mask, action_mask, strict=True)):
            return "trains a token that is not an action token"
    if rollout_log_probs is not None:
        if len(rollout_log_probs) != response_length:
            return (
                f"rollout_log_probs length {len(rollout_log_probs)} does not match response length {response_length}"
            )
        if any(
            not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value)
            for value in rollout_log_probs
        ):
            return "rollout_log_probs must contain finite numbers"
    return None
