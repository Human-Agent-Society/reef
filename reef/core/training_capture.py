"""The training block of one exact engine sample, which a record keeps so that a training component trains the sample
as the engine drew it.

Two engines make the block: a training component's rollout engine for every call of a task rollout
(``reef.train.slime_backend.engine_chat``), and the inference service for a served call that asks for it with
``x-reef-record-training`` (``reef.inference.chat``). ``reef.record2sample.samples.make_policy_trajectory`` reads it.
The block names the ``SamplingDistribution`` of the sample when the call's sampling parameters fix it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from reef.core.values import JsonValue

#: The parameters of a sampling distribution, in the order of the fields of ``SamplingDistribution``.
DISTRIBUTION_KEYS = ("temperature", "top_p", "top_k")
#: Engine sampling parameters that change the distribution beyond temperature, top_p and top_k, with the value at
#: which each leaves it unchanged.
NEUTRAL_SAMPLING_VALUES: Mapping[str, float] = {
    "min_p": 0.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
}


@dataclass(frozen=True)
class SamplingDistribution:
    """The distribution that an engine samples a reply from.

    Values: ``temperature`` is a positive finite float; ``top_p`` is a float in (0, 1]; ``top_k`` is -1, for no
    limit, or a positive int. The constructor refuses other values, bools included.
    Purpose: ``sampling_distribution`` reads it from an engine's sampling parameters, and the training block of an
    exact sample keeps it as ``to_dict`` writes it. ``DelayedSample`` reads it back (``from_dict``), and a training
    component compares it with the distribution that it trains under (``DelayedSampleLimit``).
    """

    temperature: float
    top_p: float
    top_k: int

    def __post_init__(self) -> None:
        temperature, top_p, top_k = self.temperature, self.top_p, self.top_k
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ValueError(f"a sampling distribution's temperature must be a number, got {temperature!r}")
        if not (math.isfinite(temperature) and temperature > 0):
            raise ValueError(f"a sampling distribution needs a positive finite temperature, got {temperature!r}")
        if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not 0 < top_p <= 1:
            raise ValueError(f"a sampling distribution needs a top_p in (0, 1], got {top_p!r}")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not (top_k == -1 or top_k > 0):
            raise ValueError(f"a sampling distribution's top_k must be -1 or a positive int, got {top_k!r}")
        object.__setattr__(self, "temperature", float(temperature))
        object.__setattr__(self, "top_p", float(top_p))

    def to_dict(self) -> dict[str, JsonValue]:
        """``{temperature, top_p, top_k}``, which ``from_dict`` reads back to an equal value."""
        return {"temperature": self.temperature, "top_p": self.top_p, "top_k": self.top_k}

    @classmethod
    def from_dict(cls, value: object) -> SamplingDistribution:
        """The distribution that ``to_dict`` wrote; ``ValueError`` for another value."""
        if not isinstance(value, Mapping) or set(value) != set(DISTRIBUTION_KEYS):
            raise ValueError(f"a sampling distribution is an object with exactly {list(DISTRIBUTION_KEYS)}")
        return cls(value["temperature"], value["top_p"], value["top_k"])


def sampling_distribution(params: Mapping[str, object]) -> SamplingDistribution | None:
    """The distribution that an engine's sampling parameters ``params`` draw from, or None when it is unknown.

    A ``top_k`` of 0 or below means no limit, as vLLM reads it, so it becomes -1. The result is None when ``params``
    does not set all three parameters to the values of a ``SamplingDistribution`` (an engine default is not known
    here), samples greedily, or sets another parameter that changes the distribution: ``min_p``, a penalty, or a logit
    bias.
    """
    for name, neutral in NEUTRAL_SAMPLING_VALUES.items():
        if params.get(name, neutral) != neutral:
            return None
    if params.get("logit_bias"):
        return None
    top_k = params.get("top_k")
    if isinstance(top_k, int) and not isinstance(top_k, bool) and top_k <= 0:
        top_k = -1
    try:
        return SamplingDistribution.from_dict(
            {"temperature": params.get("temperature"), "top_p": params.get("top_p"), "top_k": top_k}
        )
    except ValueError:
        return None


def sample_training(
    *,
    prompt_ids: Sequence[int],
    output_ids: Sequence[int],
    rollout_log_probs: Sequence[float],
    runtime_load_id: str,
    finish_reason: str,
    distribution: SamplingDistribution | None,
) -> dict[str, JsonValue]:
    """The training block of one exact sample: ``tokens`` (the prompt IDs then the sampled IDs), ``loss_mask`` (1 for
    each sampled token), ``rollout_log_probs`` (the sampled tokens' log-probs), ``prompt_length``,
    ``response_length``, ``runtime_load_id`` (the weights that drew every token), ``finish_reason``, and
    ``distribution`` when it is known (``sampling_distribution``).

    ``ValueError`` when ``rollout_log_probs`` does not hold one value per sampled token.
    """
    if len(rollout_log_probs) != len(output_ids):
        raise ValueError("a sample's training block needs one log-prob per sampled token")
    tokens: list[JsonValue] = [*prompt_ids, *output_ids]
    loss_mask: list[JsonValue] = [1] * len(output_ids)
    log_probs: list[JsonValue] = list(rollout_log_probs)
    block: dict[str, JsonValue] = {
        "tokens": tokens,
        "loss_mask": loss_mask,
        "rollout_log_probs": log_probs,
        "prompt_length": len(prompt_ids),
        "response_length": len(output_ids),
        "runtime_load_id": runtime_load_id,
        "finish_reason": finish_reason,
    }
    if distribution is not None:
        block["distribution"] = distribution.to_dict()
    return block


__all__ = [
    "DISTRIBUTION_KEYS",
    "NEUTRAL_SAMPLING_VALUES",
    "SamplingDistribution",
    "sample_training",
    "sampling_distribution",
]
