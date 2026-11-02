"""Readers of recorded inferences, and the sample builders that record-to-sample stages use.

The report reader ``report_score`` lives in ``reef.record2sample.reports``.
``recorded_request`` and ``recorded_response`` read the messages, tools and
answer text of a recorded inference; the sample builders turn one inference
record — or an ordered multi-call rollout — into a ``TrajectorySample`` with the
tensors the training component requires.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from reef.core.chat_rendering import flatten_content
from reef.core.job_data import TrajectorySample
from reef.core.records_types import AgentRecord
from reef.core.trajectories import make_trajectory

#: One response position's sampler top-k: token ids and their log-probs.
TopkRow = tuple[list[int], list[float]]


def recorded_training(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """The ``response.training`` section of an inference: what a training component's own rollout engine attached
    for training, such as the calls of a task rollout.

    Empty for a record that serving made, which carries no training tensors, and for a harness that ships its
    policy tensors at the top level of the payload.
    """
    response = payload.get("response")
    training = response.get("training") if isinstance(response, Mapping) else None
    return training if isinstance(training, Mapping) else {}


def recorded_request(payload: Mapping[str, Any]) -> tuple[list[Any], list[Any] | None]:
    """The messages and tools of a recorded inference request.

    A training component's rollout engine retains its provider-neutral copy under ``response.training``; a record
    without it, such as every record that serving made, carries the request body.
    """
    training = recorded_training(payload)
    if isinstance(training.get("request_messages"), list):
        messages = list(training["request_messages"])
        tools = training.get("request_tools", payload.get("tools"))
    else:
        messages = list(payload.get("messages") or [])
        tools = payload.get("tools")
    return messages, list(tools) if isinstance(tools, list) and tools else None


def recorded_response_message(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the final assistant message of a recorded inference's response, or None if there is no message.

    For a completion choice without a message, return an assistant message with the text of the choice.
    """
    message = recorded_training(payload).get("response_message")
    if isinstance(message, Mapping):
        return message
    response = payload.get("response")
    if not isinstance(response, Mapping):
        return None
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        message = choices[0].get("message")
        if isinstance(message, Mapping):
            return message
        return {"role": "assistant", "content": flatten_content(choices[0].get("text"))}
    return None


def recorded_response(payload: Mapping[str, Any]) -> str:
    """The text of a recorded inference's response: the final assistant message, or empty when it has none."""
    message = recorded_response_message(payload)
    return "" if message is None else flatten_content(message.get("content"))


def common_prefix_length(left: list[int], right: list[int]) -> int:
    for index, (left_token, right_token) in enumerate(zip(left, right, strict=False)):
        if left_token != right_token:
            return index
    return min(len(left), len(right))


def extend_with_masked_context(
    tokens: list[int],
    token_mask: list[int],
    token_log_probs: list[float],
    context: list[int],
    retain_log_probs: bool,
    token_topk: list[TopkRow] | None,
) -> None:
    """Append ``context``, the tokens that a new prompt adds after the history that it repeats, as masked context."""
    tokens.extend(context)
    token_mask.extend([0] * len(context))
    if retain_log_probs:
        token_log_probs.extend([0.0] * len(context))
    if token_topk is not None:
        token_topk.extend(([], []) for _ in context)


def drift_is_realignable(
    common_prefix: int,
    previous_prompt_length: int,
    previous_call_length: int,
    realign_threshold: int,
    scaffold_tolerance: int,
) -> bool:
    """Whether a new prompt that first differs from the previous call (its prompt and output, ``previous_call_length``
    tokens) at ``common_prefix`` differs at most ``scaffold_tolerance`` tokens before that call's output, and at most
    ``realign_threshold`` tokens before the call's end."""
    if common_prefix < previous_prompt_length - scaffold_tolerance:
        return False
    return previous_call_length - common_prefix <= realign_threshold


def realign_latest_response(
    tokens: list[int],
    token_mask: list[int],
    token_log_probs: list[float],
    replacement: list[int],
    realign_start: int,
    retain_log_probs: bool,
    token_topk: list[TopkRow] | None,
) -> None:
    """Replace the transcript from ``realign_start``, the latest response and at most the scaffold tolerance of context
    before it, with ``replacement``, a new prompt's rendering of it, as masked context: that response does not
    train."""
    tokens[realign_start:] = replacement
    token_mask[realign_start:] = [0] * len(replacement)
    if retain_log_probs:
        token_log_probs[realign_start:] = [0.0] * len(replacement)
    if token_topk is not None:
        token_topk[realign_start:] = [([], []) for _ in replacement]


def make_policy_trajectory(
    item: AgentRecord,
    reward: float,
) -> TrajectorySample:
    """Convert inference data and its evaluated reward into a sample.

    ``response.training`` attached by an inference backend is authoritative
    for policy tensors. The record's ``artifact_id``, which the inference
    service writes, names the artifact that produced the tokens; a rollout of
    Reef's own names it in its ``training`` block. Top-level tensors remain supported
    for harnesses that already ship exact policy data. Missing tensors stay empty and are
    rejected at the training boundary; Reef never reconstructs sampled ids from
    decoded text.
    """
    payload = item.payload
    training = recorded_training(payload)

    def field(name: str) -> Any:
        return training.get(name, payload.get(name, ()))

    tokens = field("tokens")
    loss_mask = field("loss_mask")
    rollout_log_probs = field("rollout_log_probs")
    artifact_id = payload.get("artifact_id") or training.get("artifact_id")
    topk_indices = field("topk_indices") or ()
    topk_log_probs = field("topk_log_probs") or ()
    return make_trajectory((item,), reward).with_training(
        tokens=[int(token) for token in tokens],
        loss_mask=[int(value) for value in loss_mask],
        rollout_log_probs=[float(value) for value in rollout_log_probs],
        artifact_id=artifact_id if isinstance(artifact_id, str) and artifact_id else None,
        topk_indices=[[int(v) for v in row] for row in topk_indices],
        topk_log_probs=[[float(v) for v in row] for row in topk_log_probs],
    )


def make_multi_turn_policy_trajectory(
    items: Sequence[AgentRecord],
    reward: float,
    *,
    source_agent_record_id: str,
    realign_threshold: int = 1024,
    scaffold_tolerance: int = 0,
) -> TrajectorySample | None:
    """Assemble one linear, terminally selected rollout into one sample.

    Each call's prompt is compared with the previous call as its sampler saw it: the previous prompt and output. A
    prompt that repeats the whole previous call appends the rest of it as masked context. A prompt that drops at most
    ``scaffold_tolerance`` tokens at the end of the previous prompt and then repeats the previous output exactly also
    appends only the rest of it: the sample keeps the dropped tokens and the sampled output, and that output trains.
    Qwen3.5 with thinking off needs this, because each generation prompt ends with an empty think block that the next
    call's history drops. A later output then trains in a context that holds the dropped tokens, which its sampler did
    not see.

    Other drift that starts at most ``scaffold_tolerance`` tokens before the previous output, and at most
    ``realign_threshold`` tokens before the previous call's end, realigns as in slime's trajectory builder: the
    prompt's tokens from the first difference or from the latest output, whichever comes first, replace the end of the
    sample as masked context, so the latest output does not train. Any other prompt is a fork and returns ``None``, as
    does drift that reaches tokens which an earlier prompt dropped; Reef never reconstructs training tokens or log
    probabilities from decoded text. With the default ``scaffold_tolerance`` 0, a prompt that does not repeat the whole
    previous prompt is a fork.
    """
    if not items or not math.isfinite(reward) or realign_threshold < 0 or scaffold_tolerance < 0:
        return None

    turns = [make_policy_trajectory(item, reward) for item in items]
    # One rollout is one policy: every turn must come from the same artifact.
    versions = {turn.training.get("artifact_id", None) for turn in turns}
    if len(versions) != 1 or None in versions or "" in versions:
        return None

    has_log_probs = [bool(turn.training.get("rollout_log_probs", [])) for turn in turns]
    if any(has_log_probs) and not all(has_log_probs):
        return None
    retain_log_probs = all(has_log_probs)
    # Sampler top-k rows ride along only when every turn recorded them, one
    # row per response token; a masked context position gets an empty row.
    retain_topk = all(
        0
        < len(turn.training.get("topk_indices", []))
        == len(turn.training.get("topk_log_probs", []))
        == len(turn.training.get("loss_mask", []))
        for turn in turns
    )

    tokens: list[int] = []
    token_mask: list[int] = []
    token_log_probs: list[float] = []
    token_topk: list[TopkRow] | None = [] if retain_topk else None
    leading_prompt_length = 0
    # The previous call as its sampler saw it. The transcript also keeps the scaffold that later prompts dropped: from
    # index ``aligned_start`` on, transcript token ``i`` is the previous call's token ``i - offset``.
    previous_prompt: list[int] = []
    previous_output: list[int] = []
    offset = 0
    aligned_start = 0

    for turn_index, turn in enumerate(turns):
        response_length = len(turn.training.get("loss_mask", []))
        if (
            response_length == 0
            or len(turn.training.get("tokens", [])) <= response_length
            or any(value not in (0, 1) for value in turn.training.get("loss_mask", []))
            or (
                turn.training.get("rollout_log_probs", [])
                and len(turn.training.get("rollout_log_probs", [])) != response_length
            )
            or any(not math.isfinite(value) for value in turn.training.get("rollout_log_probs", []))
        ):
            return None

        prompt = list(turn.training.get("tokens", [])[:-response_length])
        output = list(turn.training.get("tokens", [])[-response_length:])

        if turn_index == 0:
            tokens.extend(prompt)
            token_mask.extend([0] * len(prompt))
            if retain_log_probs:
                token_log_probs.extend([0.0] * len(prompt))
            if token_topk is not None:
                token_topk.extend(([], []) for _ in prompt)
            leading_prompt_length = len(prompt)
        else:
            previous_call = previous_prompt + previous_output
            common_prefix = common_prefix_length(previous_call, prompt)
            dropped = len(previous_prompt) - common_prefix
            history_end = common_prefix + len(previous_output)
            if common_prefix == len(previous_call):
                extend_with_masked_context(
                    tokens, token_mask, token_log_probs, prompt[common_prefix:], retain_log_probs, token_topk
                )
            elif 0 < dropped <= scaffold_tolerance and prompt[common_prefix:history_end] == previous_output:
                # The history dropped the end of the previous prompt and repeats its output: keep both, so the
                # output trains.
                aligned_start = offset + len(previous_prompt)
                offset += dropped
                extend_with_masked_context(
                    tokens, token_mask, token_log_probs, prompt[history_end:], retain_log_probs, token_topk
                )
            else:
                realign_start = min(len(previous_prompt), common_prefix)
                if offset + realign_start < aligned_start or not drift_is_realignable(
                    common_prefix, len(previous_prompt), len(previous_call), realign_threshold, scaffold_tolerance
                ):
                    # A genuine fork: the divergence reaches further back than the tolerance, into scaffold that an
                    # earlier prompt dropped, or past the realign budget.
                    return None
                realign_latest_response(
                    tokens,
                    token_mask,
                    token_log_probs,
                    prompt[realign_start:],
                    offset + realign_start,
                    retain_log_probs,
                    token_topk,
                )

        previous_prompt, previous_output = prompt, output
        tokens.extend(output)
        token_mask.extend(turn.training.get("loss_mask", []))
        if retain_log_probs:
            token_log_probs.extend(turn.training.get("rollout_log_probs", []))
        if token_topk is not None:
            token_topk.extend(
                zip(turn.training.get("topk_indices", []), turn.training.get("topk_log_probs", []), strict=True)
            )

    loss_mask = token_mask[leading_prompt_length:]
    if not loss_mask or sum(loss_mask) == 0:
        return None
    response_topk = [] if token_topk is None else token_topk[leading_prompt_length:]
    return (
        make_trajectory(items, reward)
        .with_metadata(source_agent_record_id=source_agent_record_id)
        .with_training(
            tokens=tokens,
            loss_mask=loss_mask,
            rollout_log_probs=token_log_probs[leading_prompt_length:] if retain_log_probs else [],
            artifact_id=versions.pop(),
            turn_count=len(turns),
            topk_indices=[list(indices) for indices, _ in response_topk],
            topk_log_probs=[list(log_probs) for _, log_probs in response_topk],
        )
    )
