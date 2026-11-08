"""The training block of one exact engine sample, which a record keeps so that a training component trains the sample
as the engine drew it.

Two engines make the block: a training component's rollout engine for every call of a task rollout
(``reef.train.slime_backend.engine_chat``), and the inference service for a served call that asks for it with
``x-reef-record-training`` (``reef.inference.chat``). ``reef.record2sample.samples.make_policy_trajectory`` reads it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

#: The parameters of a sampling distribution, in the order the training block's ``distribution`` names them.
DISTRIBUTION_KEYS = ("temperature", "top_p", "top_k")
#: Engine sampling parameters that change the distribution beyond temperature, top_p and top_k, with the value at
#: which each leaves it unchanged.
NEUTRAL_SAMPLING_VALUES: Mapping[str, float] = {
    "min_p": 0.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
}


def sampling_distribution(params: Mapping[str, object]) -> dict[str, float | int] | None:
    """The distribution that an engine's sampling parameters ``params`` draw from, or None when it is unknown.

    The result holds ``temperature`` (positive), ``top_p`` (in (0, 1]) and ``top_k`` (-1 for no limit, or a positive
    int; a ``top_k`` of 0 or below means no limit, as vLLM reads it). It is None when ``params`` does not set all
    three to such values (an engine default is not known here), samples greedily, or sets another parameter that
    changes the distribution: ``min_p``, a penalty, or a logit bias.
    """
    temperature, top_p, top_k = params.get("temperature"), params.get("top_p"), params.get("top_k")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not temperature > 0:
        return None
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not 0 < top_p <= 1:
        return None
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        return None
    for name, neutral in NEUTRAL_SAMPLING_VALUES.items():
        if params.get(name, neutral) != neutral:
            return None
    if params.get("logit_bias"):
        return None
    return {"temperature": float(temperature), "top_p": float(top_p), "top_k": top_k if top_k > 0 else -1}


def sample_training(
    *,
    prompt_ids: Sequence[int],
    output_ids: Sequence[int],
    rollout_log_probs: Sequence[float],
    runtime_load_id: str,
    finish_reason: str,
    distribution: Mapping[str, float | int] | None,
) -> dict[str, object]:
    """The training block of one exact sample: ``tokens`` (the prompt IDs then the sampled IDs), ``loss_mask`` (1 for
    each sampled token), ``rollout_log_probs`` (the sampled tokens' log-probs), ``prompt_length``,
    ``response_length``, ``runtime_load_id`` (the weights that drew every token), ``finish_reason``, and
    ``distribution`` when it is known (``sampling_distribution``).

    ``ValueError`` when ``rollout_log_probs`` does not hold one value per sampled token.
    """
    if len(rollout_log_probs) != len(output_ids):
        raise ValueError("a sample's training block needs one log-prob per sampled token")
    block: dict[str, object] = {
        "tokens": [*prompt_ids, *output_ids],
        "loss_mask": [1] * len(output_ids),
        "rollout_log_probs": list(rollout_log_probs),
        "prompt_length": len(prompt_ids),
        "response_length": len(output_ids),
        "runtime_load_id": runtime_load_id,
        "finish_reason": finish_reason,
    }
    if distribution is not None:
        block["distribution"] = dict(distribution)
    return block


__all__ = ["DISTRIBUTION_KEYS", "NEUTRAL_SAMPLING_VALUES", "sample_training", "sampling_distribution"]
