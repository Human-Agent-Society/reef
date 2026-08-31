"""Convert Reef wire payloads into Slime's external rollout format.

Each loss family's spec names its payload builder (``build_rollout_data``);
the shared policy builder below covers every 5-tuple family, and the common
row and rollout-id checks live here so families cannot drift on the shared
wire contract. Structural row validity is the same
:func:`reef.train.types.policy_row_violation` predicate the pairing
processors apply at ingest time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

from reef.core.artifact_ref import parse_runtime_load_spans
from reef.train.slime_backend.algorithm import SlimeAlgorithm
from reef.train.types import policy_row_violation


def to_slime_rollout_data(payload: Mapping[str, Any]) -> dict:
    """Validate and convert Reef rows into Slime's external rollout payload."""
    if not isinstance(payload, Mapping):
        raise ValueError("Reef training payload must be an object")

    samples = payload.get("samples")
    if not isinstance(samples, Sequence) or isinstance(samples, str | bytes) or not samples:
        raise ValueError(
            "Reef training payload contains no samples; inference responses must include non-empty training tensors"
        )

    # Imported here so this module's helpers stay importable from the family
    # payload modules the registry itself imports.
    from reef.train.slime_backend.loss_families import LOSS_FAMILIES, UnknownLossFamilyError

    loss = payload.get("loss")
    if not isinstance(loss, str):
        raise ValueError(f"unsupported Reef training loss: {loss!r}")
    try:
        # Registered names and dotted "package.module:SPEC" references alike;
        # the registry is the single vocabulary authority.
        spec = LOSS_FAMILIES.resolve(loss)
    except UnknownLossFamilyError as exc:
        raise ValueError(f"unsupported Reef training loss: {loss!r} ({exc})") from exc

    data = spec.build_rollout_data(payload, samples)
    _attach_runtime_load_ids(data, payload, samples)
    _enforce_spec_advantages(spec, payload)
    if spec.requires_rollout_logprobs and not data.get("rollout_log_probs"):
        raise ValueError(
            f"{spec.loss_family} requires one rollout_log_prob per response token; "
            "the inference harness did not return complete policy training tensors"
        )
    return data


def _attach_runtime_load_ids(data: dict, payload: Mapping[str, Any], samples: Sequence) -> None:
    """Preserve sample and token runtime load IDs beside Slime's policy rows."""
    versions = payload.get("producing_runtime_load_ids")
    if versions is not None:
        if not isinstance(versions, Sequence) or isinstance(versions, str | bytes) or len(versions) != len(samples):
            raise ValueError("producing_runtime_load_ids must contain one value per sample")
        recorded_versions = data.get("producing_runtime_load_ids")
        if recorded_versions is not None and list(recorded_versions) != list(versions):
            raise ValueError("loss-family row runtime load IDs do not match the shared training payload")
        data["producing_runtime_load_ids"] = list(versions)

    raw_groups = payload.get("producing_runtime_load_spans")
    if raw_groups is None:
        return
    if not isinstance(raw_groups, Sequence) or isinstance(raw_groups, str | bytes) or len(raw_groups) != len(samples):
        raise ValueError("producing_runtime_load_spans must contain one span list per sample")
    response_lengths = data.get("response_lengths")
    if not isinstance(response_lengths, Sequence) or len(response_lengths) != len(samples):
        raise ValueError("token runtime-load-ID spans require one response length per sample")
    normalized: list[list[dict[str, Any]]] = []
    for sample_index, (raw_spans, response_length) in enumerate(zip(raw_groups, response_lengths, strict=True)):
        if not raw_spans:
            normalized.append([])
            continue
        spans = parse_runtime_load_spans(
            raw_spans,
            field_name=f"producing_runtime_load_spans[{sample_index}]",
            response_length=response_length,
        )
        normalized.append(
            [{"start": span.start, "end": span.end, "runtime_load_id": span.runtime_load_id} for span in spans]
        )
    data["producing_runtime_load_spans"] = normalized


def _enforce_spec_advantages(spec: SlimeAlgorithm, payload: Mapping[str, Any]) -> None:
    """Enforce the spec's advantage-presence contract on the wire payload."""
    advantages = payload.get("advantages")
    if spec.advantages == "required" and advantages is None:
        raise ValueError(f"{spec.loss_family} requires one Reef advantage per sample")
    if spec.advantages == "forbidden" and advantages is not None:
        raise ValueError(
            spec.forbidden_advantages_message
            or f"{spec.loss_family} advantages are computed by the training backend; the Reef payload must omit them"
        )


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


def build_policy_rollout_data(
    payload: Mapping[str, Any],
    samples: Sequence,
    spec: SlimeAlgorithm,
) -> dict:
    """Convert policy 5-tuple rows for any spec without a custom builder."""
    loss = spec.loss_family
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

    has_log_probs = [bool(values) for values in rollout_log_probs]
    if any(has_log_probs) and not all(has_log_probs):
        raise ValueError("rollout_log_probs must be present for every sample or omitted for the whole batch")

    data: dict[str, Any] = {
        "tokens": tokens,
        "loss_masks": loss_masks,
        "rewards": rewards,
        "response_lengths": [len(loss_mask) for loss_mask in loss_masks],
        # Reef's ``PolicySample`` carries no truncation flag (the processor
        # never captures the engine's finish reason), so every row ships as
        # non-truncated. If a truncation bit is ever added to the sample and
        # its wire row, thread it through here instead of this constant.
        "truncated": [0] * len(samples),
        "sample_indices": sample_indices,
        "rollout_ids": normalized_ids,
        "rollout_mask_sums": rollout_mask_sums(normalized_ids, loss_masks),
        "loss": loss,
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
    if step_sizes is not None:
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
    remainder = payload.get("external_remainder")
    if remainder is not None:
        if step_sizes is not None:
            raise ValueError("external_remainder applies only when external_step_sizes is absent")
        if remainder not in ("partial", "error"):
            raise ValueError(f"external_remainder must be 'partial' or 'error', got {remainder!r}")
        data["external_remainder"] = remainder
    return data
