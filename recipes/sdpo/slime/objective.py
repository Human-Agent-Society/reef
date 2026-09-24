"""SDPO worker hooks; the shared backend owns the teacher and divergence kernels."""

from argparse import Namespace
from collections.abc import Callable
from typing import Any

from reef.train.slime_backend.algorithm import objective


@objective("custom_loss_function_path")
def sdpo_loss(
    args: Namespace, batch: dict[str, Any], logits: Any, sum_of_sample_mean: Callable[[Any], Any]
) -> tuple[Any, dict[str, Any]]:
    """Distil the feedback-conditioned teacher at the student's original response positions."""
    import torch

    from reef.train.slime_backend.distill.objective import distill_loss

    factors = torch.cat(
        [
            logits.new_full((length,), weight, dtype=torch.float32)
            for length, weight in zip(batch["response_lengths"], batch["distill_sample_weights"], strict=True)
        ]
    )

    def masked_sample_sum(values):
        return sum_of_sample_mean(values * factors)

    return distill_loss(args, batch, logits, masked_sample_sum)


@objective("reef_actor_pre_train_hook_path")
def sdpo_actor_pre_train(actor: Any, rollout_data: dict[str, Any]) -> None:
    """Score the complete batch before its single optimizer step."""
    from reef.train.slime_backend.distill.objective import distill_actor_pre_train

    distill_actor_pre_train(actor, rollout_data)
