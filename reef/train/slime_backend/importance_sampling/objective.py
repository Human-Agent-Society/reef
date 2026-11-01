"""The worker side of the ``importance_sampling`` family: its loss, in torch."""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable
from typing import Any

import torch

from reef.train.slime_backend.algorithm import objective


def truncated_importance_sampling(
    log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    cap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The token losses ``-A * sg[min(p / q, cap)] * log p``, and the untruncated ratios ``p / q``.

    The three tensors hold one value per response token, in the same order and of one dtype, in which the losses
    are. Raise ``ValueError`` for tensors of different shapes.
    """
    if log_probs.shape != rollout_log_probs.shape or log_probs.shape != advantages.shape:
        raise ValueError(
            "importance sampling needs one log-prob, rollout log-prob and advantage per token: "
            f"log_probs={tuple(log_probs.shape)}, rollout_log_probs={tuple(rollout_log_probs.shape)}, "
            f"advantages={tuple(advantages.shape)}"
        )
    ratio = torch.exp(log_probs.detach() - rollout_log_probs)
    return -advantages * ratio.clamp(max=cap) * log_probs, ratio


@objective("custom_loss_function_path")
def importance_sampling_loss(
    args: Namespace,
    batch: dict[str, Any],
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Truncated importance sampling on Reef's advantages, reduced by Slime's per-sample masked mean.

    ``q`` is the rollout engine's log-probability of each sampled token, which the component's task rollouts
    record (``rollout_log_probs``); a batch without them raises ``ValueError``, since the actor's recomputed
    log-probs are not the sampler's.
    """
    from slime.backends.megatron_utils.loss import get_log_probs_and_entropy, get_rollout_top_p_logprob_kwargs

    rollout_log_probs = batch.get("rollout_log_probs")
    advantages = batch.get("advantages")
    if not rollout_log_probs or not advantages:
        raise ValueError(
            "the importance_sampling family trains samples with recorded rollout log-probs and Reef's advantages: "
            "the component's own task rollouts"
        )
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
    q = torch.cat(rollout_log_probs, dim=0).to(log_probs)
    token_loss, ratio = truncated_importance_sampling(
        log_probs, q, torch.cat(advantages, dim=0).to(log_probs), float(args.tis_clip)
    )
    loss = sum_of_sample_mean(token_loss)
    if log_probs.numel() == 0:
        loss = loss + 0 * logits.sum()
    truncated = (ratio > float(args.tis_clip)).float()
    return loss, {
        "loss": loss.detach().clone(),
        "pg_loss": loss.detach().clone(),
        "importance_ratio": sum_of_sample_mean(ratio).detach().clone(),
        "truncated_fraction": sum_of_sample_mean(truncated).detach().clone(),
        "ppo_kl": sum_of_sample_mean(q - log_probs.detach()).detach().clone(),
    }


__all__ = ["importance_sampling_loss", "truncated_importance_sampling"]
