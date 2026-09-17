"""Slime implementation of SFT: the stock ``sft_loss`` over the sample's response tokens."""

from __future__ import annotations

from argparse import Namespace

from reef.train.slime_backend.algorithm import SlimeAlgorithm, register_loss_family


@register_loss_family
class SftAlgorithm(SlimeAlgorithm):
    """Slime's stock ``sft_loss`` over the default policy row: every masked token, unweighted."""

    loss_family = "sft"
    loss_type = "sft_loss"
    advantages = "forbidden"
    forbidden_advantages_message = (
        "sft ignores advantages: Slime's sft_loss_function trains every token unweighted, so the "
        "payload's advantages would be silently discarded. A reward-weighted objective needs a loss "
        "family whose loss consumes them (sao, or a custom family registered with register_loss_family)."
    )

    def validate_specific_args(self, args: Namespace, source: str) -> None:
        return None
