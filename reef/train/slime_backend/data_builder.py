"""Convert Reef wire payloads into Slime's external rollout format.

``to_slime_rollout_data`` resolves the payload's loss family, lets the family
build its rollout data (``SlimeAlgorithm.build_rollout_data``), and enforces
what every family shares: the producing artifact of each row and the family's
advantage contract. The shared policy row is ``reef.train.slime_backend.policy_rows``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from reef.train.slime_backend.algorithm import SlimeAlgorithm
from reef.train.slime_backend.loss_families import UnknownLossFamilyError, resolve_loss_family


def to_slime_rollout_data(payload: Mapping[str, Any]) -> dict:
    """Validate and convert Reef rows into Slime's external rollout payload."""
    if not isinstance(payload, Mapping):
        raise ValueError("Reef training payload must be an object")

    samples = payload.get("samples")
    if not isinstance(samples, Sequence) or isinstance(samples, str | bytes) or not samples:
        raise ValueError(
            "Reef training payload contains no samples; inference responses must include non-empty training tensors"
        )

    loss = payload.get("loss")
    if not isinstance(loss, str):
        raise ValueError(f"unsupported Reef training loss: {loss!r}")
    try:
        # Registered names and dotted "package.module:SPEC" references alike.
        spec = resolve_loss_family(loss)
    except UnknownLossFamilyError as exc:
        raise ValueError(f"unsupported Reef training loss: {loss!r} ({exc})") from exc

    data = spec.build_rollout_data(payload, samples)
    _attach_producing_artifact_ids(data, payload, samples)
    _enforce_spec_advantages(spec, payload)
    return data


def _attach_producing_artifact_ids(data: dict, payload: Mapping[str, Any], samples: Sequence) -> None:
    """Keep each sample's producing artifact beside Slime's policy rows, as training metadata."""
    artifact_ids = payload.get("producing_artifact_ids")
    if artifact_ids is None:
        return
    if (
        not isinstance(artifact_ids, Sequence)
        or isinstance(artifact_ids, str | bytes)
        or len(artifact_ids) != len(samples)
    ):
        raise ValueError("producing_artifact_ids must contain one value per sample")
    recorded = data.get("producing_artifact_ids")
    if recorded is not None and list(recorded) != list(artifact_ids):
        raise ValueError("loss-family row artifact IDs do not match the shared training payload")
    data["producing_artifact_ids"] = list(artifact_ids)


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
