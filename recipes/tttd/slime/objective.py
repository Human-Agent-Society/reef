"""Torch-owned tensor implementation for TTT-Discover."""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable
from typing import Any

import torch

from reef.train.slime_backend.algorithm import objective
from reef.train.slime_backend.importance_sampling.objective import importance_sampling_loss


@objective("custom_loss_function_path")
def tttd_loss(
    args: Namespace,
    batch: dict[str, Any],
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Exact sum-reduced importance-sampling objective used by TTT-Discover.

    The reference passes sampled log-probabilities and token advantages to
    Tinker's ``importance_sampling`` loss, which computes
    ``-(exp(logp-logq) * advantage).sum()`` over every submitted token. This
    is Slime's ``importance_sampling`` loss
    (``reef.train.slime_backend.importance_sampling``) on TTTD's advantages,
    with the same step token sum as Tinker.

    The response loss mask is intentionally not used for this policy term. In
    the reference, that mask gates the frozen-base KL before ``remove_mask``
    sends the Datum to Tinker's built-in loss, so forced two-phase prefill
    tokens retain their trajectory advantage in the policy objective.
    """
    return importance_sampling_loss(args, batch, logits, sum_of_sample_mean)


@objective("custom_advantage_function_path")
def tttd_advantages(args, rollout_data) -> None:
    """Apply the frozen-base, centered token KL from the reference algorithm.

    Reef has already supplied one adaptive-entropic advantage per trajectory,
    expanded over response tokens by the bridge. Slime computes frozen-base
    log-probabilities immediately before this callback. The update below is
    the direct distributed equivalent of
    ``test-time-training/discover@6c40e82``'s ``incorporate_kl_penalty``.
    """
    import torch.distributed as dist
    from megatron.core import mpu

    if mpu.get_context_parallel_world_size() != 1:
        raise ValueError("the TTTD frozen-base KL adapter currently requires context_parallel_size=1")
    if not mpu.is_pipeline_last_stage():
        return

    advantages: list[torch.Tensor] | None = rollout_data.get("advantages")
    rollout_log_probs: list[torch.Tensor] | None = rollout_data.get("rollout_log_probs")
    base_log_probs: list[torch.Tensor] | None = rollout_data.get("ref_log_probs")
    loss_masks: list[torch.Tensor] | None = rollout_data.get("loss_masks")
    if not advantages or not rollout_log_probs or not base_log_probs or not loss_masks:
        raise ValueError("TTTD base KL requires external advantages, rollout_log_probs, ref_log_probs, and loss_masks")
    if not (len(advantages) == len(rollout_log_probs) == len(base_log_probs) == len(loss_masks)):
        raise ValueError("TTTD base KL fields must contain the same number of samples")

    log_probability_differences: list[torch.Tensor] = []
    numerator = advantages[0].new_tensor(0.0, dtype=torch.float32)
    denominator = advantages[0].new_tensor(0.0, dtype=torch.float32)
    for index, (rollout, base, mask) in enumerate(zip(rollout_log_probs, base_log_probs, loss_masks, strict=True)):
        mask = mask.to(device=rollout.device, dtype=torch.float32)
        if rollout.shape != base.shape or rollout.shape != mask.shape:
            raise ValueError(
                f"TTTD base KL sample {index} shape mismatch: "
                f"rollout={tuple(rollout.shape)}, base={tuple(base.shape)}, mask={tuple(mask.shape)}"
            )
        difference = (rollout.float() - base.float()) * mask
        log_probability_differences.append(difference)
        numerator = numerator + difference.sum()
        denominator = denominator + mask.sum()

    data_parallel_group = mpu.get_data_parallel_group()
    if dist.get_world_size(group=data_parallel_group) > 1:
        dist.all_reduce(numerator, group=data_parallel_group)
        dist.all_reduce(denominator, group=data_parallel_group)
    if denominator.item() <= 0:
        raise ValueError("TTTD base KL received an empty loss mask")
    average_difference = numerator / denominator

    coefficient = float(args.kl_coef)
    adjusted: list[torch.Tensor] = []
    for advantage, difference, mask in zip(
        advantages,
        log_probability_differences,
        loss_masks,
        strict=True,
    ):
        float_mask = mask.to(device=advantage.device, dtype=torch.float32)
        adjusted.append(advantage.float() + coefficient * float_mask * (average_difference - difference))

    rollout_data["advantages"] = adjusted
    rollout_data["returns"] = [value.clone() for value in adjusted]
    rollout_data["tttd_base_logp_diff"] = [average_difference.detach().expand_as(value) for value in adjusted]


__all__ = ["tttd_advantages", "tttd_loss"]
