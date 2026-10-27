"""The worker side of ``importance_sampling``: Tinker's un-clipped token loss, summed over the optimizer step."""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable, Mapping
from numbers import Integral

import torch

from reef.train.slime_backend.algorithm import objective


def importance_sampling_surrogate(
    log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    """Return Tinker's un-clipped ``importance_sampling`` token loss."""
    if log_probs.shape != rollout_log_probs.shape or log_probs.shape != advantages.shape:
        raise ValueError(
            "importance-sampling tensors must have identical shapes: "
            f"log_probs={tuple(log_probs.shape)}, "
            f"rollout_log_probs={tuple(rollout_log_probs.shape)}, "
            f"advantages={tuple(advantages.shape)}"
        )
    ratio = torch.exp(log_probs.float() - rollout_log_probs.float())
    return -ratio * advantages.float()


@objective("custom_loss_function_path")
def importance_sampling_loss(
    args: Namespace,
    batch: Mapping[str, object],
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Tinker's ``importance_sampling`` loss: ``-(exp(logp - logq) * advantage).sum()`` over the response tokens, each advantage masked.

    The loss reads the token advantages as the rollout data holds them; it
    reads no loss mask. A family that trains only the masked tokens puts the
    mask in the advantages, as this package's spec does. Slime's stock
    ``policy_loss`` clips the ratio and averages the result, so it is not used.

    Slime's outer Megatron adapter divides a custom loss by the dynamic global
    batch size. The loss multiplies by the same value to cancel only that
    division. After microbatch and data-parallel accumulation, the gradient is
    the token sum of the full step, which is what Tinker applies.
    """
    from slime.backends.megatron_utils.loss import get_log_probs_and_entropy, get_rollout_top_p_logprob_kwargs

    if not args.use_rollout_logprobs:
        raise ValueError("importance_sampling requires --use-rollout-logprobs")

    rollout_log_probs_list = batch.get("rollout_log_probs")
    advantages_list = batch.get("advantages")
    if (
        not isinstance(rollout_log_probs_list, list | tuple)
        or not rollout_log_probs_list
        or not isinstance(advantages_list, list | tuple)
        or not advantages_list
    ):
        raise ValueError("importance_sampling requires rollout_log_probs and advantages")
    step_global_batch_size = batch.get("step_global_batch_size")
    if (
        not isinstance(step_global_batch_size, Integral)
        or isinstance(step_global_batch_size, bool)
        or step_global_batch_size <= 0
    ):
        raise ValueError("importance_sampling requires a positive step_global_batch_size")

    _, outputs = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        with_entropy=False,
        **get_rollout_top_p_logprob_kwargs(args, batch),
    )
    log_probs = torch.cat(outputs["log_probs"], dim=0)
    rollout_log_probs = torch.cat(rollout_log_probs_list, dim=0)
    advantages = torch.cat(advantages_list, dim=0)
    token_loss = importance_sampling_surrogate(log_probs, rollout_log_probs, advantages)
    tinker_loss_sum = token_loss.sum()
    loss = tinker_loss_sum * int(step_global_batch_size)

    if log_probs.numel() == 0:
        loss = loss + 0 * logits.sum()

    ratio = torch.exp(log_probs.detach().float() - rollout_log_probs.float())
    old_policy_kl = rollout_log_probs.float() - log_probs.detach().float()
    return loss, {
        "loss": loss.detach().clone(),
        "pg_loss": loss.detach().clone(),
        "importance_ratio": sum_of_sample_mean(ratio).detach().clone(),
        "ppo_kl": sum_of_sample_mean(old_policy_kl).detach().clone(),
    }
