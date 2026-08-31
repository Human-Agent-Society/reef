"""Shared readers and sample builders both engines use.

The report readers (``report_score``, ``report_is_trainable``) answer what a
report carries; the sample builders turn one inference record — or an ordered
multi-call episode — into a ``PolicySample`` with the tensors the training
bridge requires.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from reef.core.artifact_ref import WeightVersionSpan, parse_weight_version_spans
from reef.core.records_types import AgentRecord
from reef.train.types import PolicySample


def report_is_trainable(report: AgentRecord) -> bool:
    """Honor an optional framework-neutral report eligibility marker."""
    metadata = report.payload.get("metadata", {})
    training = metadata.get("training") if isinstance(metadata, Mapping) else None
    return not isinstance(training, Mapping) or training.get("eligible", True) is not False


def sample_assembly_config_fields(config: Mapping[str, Any]) -> tuple[bool, int, int]:
    """Parse and validate the sample-assembly settings in one place.

    Every reported-feedback processor accepts these config keys, so
    ``ReportedFeedbackProcessor`` runs this at construction to fail fast on a bad deployment — even for
    specs that never assemble a policy sample. ``SampleAssembly`` is the one
    consumer of the returned values.
    """
    accept_multi_turn = config.get("accept_multi_turn_policy_samples", False)
    if not isinstance(accept_multi_turn, bool):
        raise ValueError("accept_multi_turn_policy_samples must be a boolean")
    realign_threshold = int(config.get("realign_threshold", 1024))
    if realign_threshold < 0:
        raise ValueError("realign_threshold must be non-negative")
    scaffold_tolerance = int(config.get("scaffold_tolerance", 0))
    if scaffold_tolerance < 0:
        raise ValueError("scaffold_tolerance must be non-negative")
    return accept_multi_turn, realign_threshold, scaffold_tolerance


def report_score(report: AgentRecord) -> float | None:
    """Return the report's numeric score, or ``None`` when absent or non-numeric.

    Bool is not a score. Callers layer their own selection policy (windows,
    finiteness, minimums) on top of this one shared reading of the payload.
    """
    score = report.payload.get("score")
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        return None
    return float(score)


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    for index, (left_token, right_token) in enumerate(zip(left, right, strict=False)):
        if left_token != right_token:
            return index
    return min(len(left), len(right))


def _extend_with_prompt_delta(
    tokens: list[int],
    token_mask: list[int],
    token_log_probs: list[float],
    prompt: list[int],
    retain_log_probs: bool,
) -> None:
    """Exact extension: the new prompt starts with the whole assembled
    transcript, so append only the masked context it adds."""
    prompt_delta = prompt[len(tokens) :]
    tokens.extend(prompt_delta)
    token_mask.extend([0] * len(prompt_delta))
    if retain_log_probs:
        token_log_probs.extend([0.0] * len(prompt_delta))


def _drift_is_realignable(
    common_prefix: int,
    latest_response_start: int | None,
    transcript_length: int,
    realign_threshold: int,
    scaffold_tolerance: int,
) -> bool:
    """Whether prompt drift stays inside the latest response (allowing at most
    ``scaffold_tolerance`` tokens of masked end-of-prompt scaffold before it)
    and the budget."""
    if latest_response_start is None or common_prefix < latest_response_start - scaffold_tolerance:
        return False
    return transcript_length - common_prefix <= realign_threshold


def _realign_latest_response(
    tokens: list[int],
    token_mask: list[int],
    token_log_probs: list[float],
    prompt: list[int],
    realign_start: int,
    retain_log_probs: bool,
) -> None:
    """Short drift confined to the latest response (plus at most the scaffold
    tolerance of masked context before it): replace that span with the
    prompt's masked rendering of it."""
    prompt_tail = prompt[realign_start:]
    tokens[realign_start:] = prompt_tail
    token_mask[realign_start:] = [0] * len(prompt_tail)
    if retain_log_probs:
        token_log_probs[realign_start:] = [0.0] * len(prompt_tail)


def make_policy_sample(
    item: AgentRecord,
    reward: float,
) -> PolicySample:
    """Convert inference data and its evaluated reward into a sample.

    ``response.training`` attached by an inference backend is authoritative
    for policy tensors. A top-level ``weight_version`` stamped after response
    validation is authoritative for provenance; if both locations provide a
    version, they must agree. Top-level tensors remain supported for harnesses
    that already ship exact policy data. Missing tensors stay empty and are
    rejected by policy processors; Reef never reconstructs sampled ids from
    decoded text.
    """
    payload = item.payload
    response = payload.get("response", {})
    training = response.get("training", {}) if isinstance(response, dict) else {}

    def field(name: str):
        return training.get(name, payload.get(name, ())) if isinstance(training, dict) else payload.get(name, ())

    tokens = field("tokens")
    loss_mask = field("loss_mask")
    rollout_log_probs = field("rollout_log_probs")
    training_weight_version = training.get("weight_version") if isinstance(training, dict) else None
    recorded_weight_version = payload.get("weight_version")
    if recorded_weight_version and training_weight_version and recorded_weight_version != training_weight_version:
        raise ValueError(
            "response training weight_version disagrees with the validated record: "
            f"{training_weight_version!r} != {recorded_weight_version!r}"
        )
    training_spans = training.get("weight_version_spans") if isinstance(training, dict) else None
    recorded_spans = payload.get("weight_version_spans")
    weight_version_spans = _policy_version_spans(
        training_spans if training_spans is not None else recorded_spans,
        len(loss_mask),
    )
    if (
        training_spans is not None
        and recorded_spans is not None
        and weight_version_spans != _policy_version_spans(recorded_spans, len(loss_mask))
    ):
        raise ValueError("response token weight-version spans disagree with the validated record")
    span_versions = {span.weight_version for span in weight_version_spans}
    if weight_version_spans:
        if recorded_weight_version and span_versions != {recorded_weight_version}:
            raise ValueError("validated record weight_version disagrees with its token weight-version spans")
        if training_weight_version and span_versions != {training_weight_version}:
            raise ValueError("response training weight_version disagrees with its token weight-version spans")
        weight_version = next(iter(span_versions)) if len(span_versions) == 1 else None
    else:
        weight_version = (
            recorded_weight_version or training_weight_version or getattr(item.artifact_ref, "weight_version", None)
        )
    topk_indices = field("topk_indices") or ()
    topk_log_probs = field("topk_log_probs") or ()
    return PolicySample(
        source_agent_record_id=item.agent_record_id,
        tokens=tuple(int(token) for token in tokens),
        loss_mask=tuple(int(value) for value in loss_mask),
        rollout_log_probs=tuple(float(value) for value in rollout_log_probs),
        reward=reward,
        weight_version=weight_version,
        weight_version_spans=weight_version_spans,
        topk_indices=tuple(tuple(int(v) for v in row) for row in topk_indices),
        topk_log_probs=tuple(tuple(float(v) for v in row) for row in topk_log_probs),
    )


def _policy_version_spans(value: Any, response_length: int) -> tuple[WeightVersionSpan, ...]:
    if value is None:
        return ()
    return parse_weight_version_spans(value, response_length=response_length)


def make_multi_turn_policy_sample(
    items: Sequence[AgentRecord],
    reward: float,
    *,
    source_agent_record_id: str,
    realign_threshold: int = 1024,
    scaffold_tolerance: int = 0,
) -> PolicySample | None:
    """Assemble one linear, terminally selected episode into one sample.

    This follows slime's trajectory builder: exact prompt extension appends
    masked context, while short drift confined to the latest response is
    realigned as masked context. A genuine fork returns ``None`` rather than
    reconstructing policy evidence from decoded text.

    ``scaffold_tolerance`` lets the drift reach a bounded number of tokens
    *before* the latest response span. Thinking chat templates need it: the
    generation prompt ends with an opening ``<think>`` scaffold that the
    history re-rendering of the same turn drops, so every turn's prompt
    diverges a couple of tokens ahead of the previous response. That span is
    always masked prompt context (the previous turn's trained tokens start at
    the response boundary), so the realignment replaces only masked scaffold,
    never policy evidence. Default 0 keeps the strict boundary.
    """
    if not items or not math.isfinite(reward) or realign_threshold < 0 or scaffold_tolerance < 0:
        return None

    turns = [make_policy_sample(item, reward) for item in items]
    versions = {turn.weight_version for turn in turns}
    if len(versions) != 1 or None in versions or "" in versions:
        return None

    has_log_probs = [bool(turn.rollout_log_probs) for turn in turns]
    if any(has_log_probs) and not all(has_log_probs):
        return None
    retain_log_probs = all(has_log_probs)

    tokens: list[int] = []
    token_mask: list[int] = []
    token_log_probs: list[float] = []
    leading_prompt_length = 0
    latest_response_start: int | None = None

    for turn_index, turn in enumerate(turns):
        response_length = len(turn.loss_mask)
        if (
            response_length == 0
            or len(turn.tokens) <= response_length
            or any(value not in (0, 1) for value in turn.loss_mask)
            or (turn.rollout_log_probs and len(turn.rollout_log_probs) != response_length)
            or any(not math.isfinite(value) for value in turn.rollout_log_probs)
        ):
            return None

        prompt = list(turn.tokens[:-response_length])
        output = list(turn.tokens[-response_length:])

        if turn_index == 0:
            tokens.extend(prompt)
            token_mask.extend([0] * len(prompt))
            if retain_log_probs:
                token_log_probs.extend([0.0] * len(prompt))
            leading_prompt_length = len(prompt)
        else:
            common_prefix = _common_prefix_length(tokens, prompt)
            if common_prefix == len(tokens):
                _extend_with_prompt_delta(tokens, token_mask, token_log_probs, prompt, retain_log_probs)
            elif _drift_is_realignable(
                common_prefix, latest_response_start, len(tokens), realign_threshold, scaffold_tolerance
            ):
                if latest_response_start is None:
                    raise RuntimeError("realignable response drift has no response boundary")
                _realign_latest_response(
                    tokens,
                    token_mask,
                    token_log_probs,
                    prompt,
                    min(latest_response_start, common_prefix),
                    retain_log_probs,
                )
            else:
                # A genuine fork: the divergence reaches back before the latest
                # response or exceeds the realign budget.
                return None

        latest_response_start = len(tokens)
        tokens.extend(output)
        token_mask.extend(turn.loss_mask)
        if retain_log_probs:
            token_log_probs.extend(turn.rollout_log_probs)

    loss_mask = token_mask[leading_prompt_length:]
    if not loss_mask or sum(loss_mask) == 0:
        return None
    return PolicySample(
        source_agent_record_id=source_agent_record_id,
        tokens=tuple(tokens),
        loss_mask=tuple(loss_mask),
        rollout_log_probs=tuple(token_log_probs[leading_prompt_length:]) if retain_log_probs else (),
        reward=float(reward),
        weight_version=versions.pop(),
        turn_count=len(turns),
    )
