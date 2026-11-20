"""The policy row that most Slime loss families train, as Slime's external rollout data.

``build_policy_rollout_data`` builds the rollout data of every family whose wire row is the default five columns
(``SlimeAlgorithm.build_rollout_data``). The row and rollout-id checks live here, so that families with their own
row build on the same wire contract. Structural row validity is the
:func:`reef.train.types.policy_row_violation` predicate. Violations raise training data errors rather than changing
report eligibility. This module imports no loss family, so ``reef.train.slime_backend.algorithm`` imports it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

from reef.train.types import policy_row_violation


def normalized_rollout_ids(payload: Mapping[str, Any], samples: Sequence) -> list[int]:
    """Validate the per-sample comparison-group ids shared by every family."""
    rollout_ids = payload.get("rollout_ids")
    if not isinstance(rollout_ids, Sequence) or isinstance(rollout_ids, str | bytes):
        raise ValueError("Reef training payload must preserve rollout_ids for every sample")
    if len(rollout_ids) != len(samples):
        raise ValueError(f"rollout_ids length {len(rollout_ids)} does not match sample count {len(samples)}")
    if any(not isinstance(value, Integral) or isinstance(value, bool) for value in rollout_ids):
        raise ValueError("rollout_ids must contain integer group ids")
    return [int(value) for value in rollout_ids]


def rollout_mask_sums(rollout_ids: Sequence[int], loss_masks: Sequence[Sequence[int]]) -> list[int]:
    """Aggregate trained-token counts per rollout group, broadcast per sample."""
    sums_by_rollout: dict[int, int] = {}
    for group_id, loss_mask in zip(rollout_ids, loss_masks, strict=True):
        sums_by_rollout[group_id] = sums_by_rollout.get(group_id, 0) + sum(loss_mask)
    return [sums_by_rollout[group_id] for group_id in rollout_ids]


def validate_policy_columns(
    label: str,
    row_tokens: Any,
    row_loss_mask: Any,
    row_log_probs: Any,
    reward: Any,
    *,
    action_mask: Any = None,
    require_log_probs: bool = False,
) -> tuple[list[int], list[int], list[float], list[int] | None, float]:
    """Validate the columns every policy-family row shares; return cast copies.

    ``label`` prefixes error messages (e.g. ``"sample 3"``). Structural row
    validity — mask/log-prob lengths and values, action alignment — is the
    shared :func:`policy_row_violation` predicate; this adds only the
    wire-boundary type checks (sequence-ness, integer tokens, numeric reward).
    ``require_log_probs`` refuses an empty ``rollout_log_probs`` column instead
    of treating it as an intentionally omitted behaviour proxy.
    """
    named_columns = [
        ("tokens", row_tokens),
        ("loss_mask", row_loss_mask),
        ("rollout_log_probs", row_log_probs),
    ] + ([] if action_mask is None else [("action_mask", action_mask)])
    for name, seq in named_columns:
        if not isinstance(seq, Sequence) or isinstance(seq, str | bytes):
            raise ValueError(f"{label} {name} must be a sequence")

    tokens = list(row_tokens)
    loss_mask = list(row_loss_mask)
    log_probs = list(row_log_probs)
    action_mask_values = None if action_mask is None else list(action_mask)
    violation = policy_row_violation(
        tokens,
        loss_mask,
        log_probs if (log_probs or require_log_probs) else None,
        action_mask=action_mask_values,
    )
    if violation is not None:
        raise ValueError(f"{label} {violation}")
    if any(not isinstance(token, Integral) or isinstance(token, bool) for token in tokens):
        raise ValueError(f"{label} tokens must contain integers")
    if action_mask_values is not None and any(
        value not in (0, 1) or isinstance(value, bool) for value in action_mask_values
    ):
        raise ValueError(f"{label} action_mask must contain only 0 or 1")
    if not isinstance(reward, Real) or isinstance(reward, bool):
        raise ValueError(f"{label} reward must be numeric")

    return (
        [int(token) for token in tokens],
        [int(value) for value in loss_mask],
        [float(value) for value in log_probs],
        None if action_mask_values is None else [int(value) for value in action_mask_values],
        float(reward),
    )


def build_policy_rollout_data(payload: Mapping[str, Any], samples: Sequence, loss_family: str) -> dict:
    """Convert policy 5-tuple rows of the family named ``loss_family``, which has no custom builder."""
    normalized_ids = normalized_rollout_ids(payload, samples)

    tokens: list[list[int]] = []
    loss_masks: list[list[int]] = []
    rollout_log_probs: list[list[float]] = []
    rewards: list[float] = []
    sample_indices: list[int] = []

    for sample_index, row in enumerate(samples):
        if not isinstance(row, Sequence) or isinstance(row, str | bytes) or len(row) != 5:
            raise ValueError(
                f"sample {sample_index} must be [source_id, tokens, loss_mask, rollout_log_probs, reward]"
            )
        _, row_tokens, row_loss_mask, row_log_probs, reward = row
        row_tokens, row_loss_mask, row_log_probs, _, reward = validate_policy_columns(
            f"sample {sample_index}", row_tokens, row_loss_mask, row_log_probs, reward
        )
        tokens.append(row_tokens)
        loss_masks.append(row_loss_mask)
        rollout_log_probs.append(row_log_probs)
        rewards.append(reward)
        sample_indices.append(sample_index)

    # The actor computes the old-policy log-probabilities of every row; recorded ones ride along only when every
    # row has them, such as the rows of a task job's own rollouts.
    has_log_probs = [bool(values) for values in rollout_log_probs]

    data: dict[str, Any] = {
        "tokens": tokens,
        "loss_masks": loss_masks,
        "rewards": rewards,
        "response_lengths": [len(loss_mask) for loss_mask in loss_masks],
        # Reef's ``TrajectorySample`` carries no truncation flag (the processor
        # never captures the engine's finish reason), so every row ships as
        # non-truncated. If a truncation bit is ever added to the sample and
        # its wire row, thread it through here instead of this constant.
        "truncated": [0] * len(samples),
        "sample_indices": sample_indices,
        "rollout_ids": normalized_ids,
        "rollout_mask_sums": rollout_mask_sums(normalized_ids, loss_masks),
        "loss": loss_family,
    }
    if all(has_log_probs):
        data["rollout_log_probs"] = rollout_log_probs

    advantages = payload.get("advantages")
    if advantages is not None:
        if not isinstance(advantages, Sequence) or isinstance(advantages, str | bytes):
            raise ValueError("advantages must be a sequence")
        if len(advantages) != len(samples):
            raise ValueError(f"advantages length {len(advantages)} does not match sample count {len(samples)}")
        if any(not isinstance(value, Real) or isinstance(value, bool) for value in advantages):
            raise ValueError("advantages must contain numeric scalars")
        data["advantages"] = [
            [float(advantage)] * response_length
            for advantage, response_length in zip(advantages, data["response_lengths"], strict=True)
        ]
    rollout_count = len(set(normalized_ids))
    step_sizes = payload.get("external_step_sizes")
    if (
        not isinstance(step_sizes, Sequence)
        or isinstance(step_sizes, str | bytes)
        or not step_sizes
        or any(not isinstance(size, Integral) or isinstance(size, bool) or size <= 0 for size in step_sizes)
    ):
        raise ValueError("external_step_sizes must be a non-empty sequence of positive integers")
    if sum(step_sizes) != rollout_count:
        raise ValueError(
            f"external_step_sizes sum {sum(step_sizes)} must equal the {rollout_count} distinct rollout_ids"
        )
    data["external_step_sizes"] = [int(size) for size in step_sizes]
    return data
