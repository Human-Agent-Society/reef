"""The ``sft`` loss family: Slime's ``sft_loss`` on the default policy row.

The built-in SFT recipe (``reef.recipe.sft``) registers this family by
reference. Thus only the training driver and the workers import this module.
"""

from __future__ import annotations

from argparse import Namespace
from typing import Any

from reef.core.job_data import TrajectorySample
from reef.core.trajectories import source_record_id, trajectory_reward
from reef.train.slime_backend.algorithm import SlimeAlgorithm, register_loss_family


@register_loss_family
class SftAlgorithm(SlimeAlgorithm):
    """Train each response token that the loss mask selects, with the same weight. Rollout log probs are not necessary."""

    loss_family = "sft"
    loss_type = "sft_loss"
    advantages = "forbidden"
    forbidden_advantages_message = (
        "sft ignores advantages: Slime's sft_loss trains each response token with the same weight. "
        "It would silently discard the advantages of the payload. For a reward-weighted objective, "
        "register a loss family that reads advantages with register_loss_family."
    )

    def validate_specific_args(self, args: Namespace, source: str) -> None:
        return None

    def shape_sample_row(self, sample: TrajectorySample) -> list[Any]:
        """The default row, with reward 0 for a trajectory that records none: ``sft_loss`` reads no reward.

        Uploaded demonstrations usually carry no reward. A recorded reward must still be finite.
        """
        reward = trajectory_reward(sample) if "reward" in sample.metadata else 0.0
        return [
            source_record_id(sample),
            list(sample.training.get("tokens", [])),
            list(sample.training.get("loss_mask", [])),
            list(sample.training.get("rollout_log_probs", [])),
            reward,
        ]


__all__ = ["SftAlgorithm"]
