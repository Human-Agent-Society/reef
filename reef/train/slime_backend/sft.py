"""The ``sft`` loss family: Slime's ``sft_loss`` on the default policy row.

The built-in SFT recipe (``reef.recipe.sft``) registers this family by
reference. Thus only the training driver and the workers import this module.
"""

from __future__ import annotations

from argparse import Namespace

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


__all__ = ["SftAlgorithm"]
