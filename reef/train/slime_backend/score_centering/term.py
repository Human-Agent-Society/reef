"""The worker side of score centering: the correction term and the wrapper that adds it to a family's loss.

:func:`centering_term` computes the per-token term on replicated log-probs;
:func:`score_centering_term` gathers the trainer's log-probs at the sampler's
head ids across the tensor-parallel vocab shards
(:func:`~reef.train.slime_backend.vocab_parallel.gather_log_probs_at_ids`)
and reduces the term with the same per-sample mean the
family's loss uses. :func:`add_score_centering` adds it to that loss; the
worker hook installs it around Slime's ``policy_loss_function`` or, for a
custom loss, through :func:`score_centered_custom_loss`. The kernel takes
plain tensors, so the CPU tests check it against a full-vocabulary reference.
Megatron and Slime are imported where the term runs.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from reef.train.slime_backend.algorithm import PolicyGradientWeight, resolve_args_loss_family
from reef.train.slime_backend.score_centering import (
    TOPK_INDICES_KEY,
    TOPK_LOG_PROBS_KEY,
    ScoreCenteringSettings,
    settings_from_args,
)
from reef.train.slime_backend.vocab_parallel import gather_log_probs_at_ids


@dataclass(frozen=True)
class CenteringTerm:
    """The per-token term and the per-token quantities the step reports, all ``[R]``.

    ``correction`` is the L1 norm of the centering coefficients
    ``q_v f(p_v / q_v) - alpha p_v``; ``tail_clipped`` marks positions where
    either tail mass fell below ``min_tail_mass``.
    """

    term: torch.Tensor
    correction: torch.Tensor
    sampler_head_mass: torch.Tensor
    trainer_head_mass: torch.Tensor
    tail_ratio: torch.Tensor
    tail_clipped: torch.Tensor


def importance_weight(ratio: torch.Tensor, weight: PolicyGradientWeight) -> torch.Tensor:
    """``f(r)`` of a declared policy-gradient weight, elementwise."""
    if weight.kind == "truncated":
        return ratio.clamp(max=weight.upper)
    if weight.kind == "masked":
        inside = (ratio > weight.lower) & (ratio < weight.upper)
        return torch.where(inside, ratio, torch.zeros_like(ratio))
    return torch.ones_like(ratio)


def centering_term(
    trainer_head_log_probs: torch.Tensor,
    sampler_head_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    weight: PolicyGradientWeight,
    min_tail_mass: float,
) -> CenteringTerm:
    """``A * sum_head sg[q_v f(p_v / q_v) - alpha p_v] * log p_v`` per position.

    ``trainer_head_log_probs`` (``[R, K]``, differentiable) are the trainer's
    full-vocabulary log-probs at the sampler's head ids, and
    ``sampler_head_log_probs`` the recorded ones. Every coefficient is
    detached. Added to a loss whose gradient is ``-A f(r_y) grad log p_y``,
    the sum has the gradient ``-A (f(r_y) s_y - sum_head (q_v f(r_v) -
    alpha p_v) s_v)``: the loss's weighted score, centered under the sampler.
    """
    dtype = torch.promote_types(trainer_head_log_probs.dtype, torch.float32)
    trainer_head = trainer_head_log_probs.to(dtype)
    with torch.no_grad():
        sampler_log_probs = sampler_head_log_probs.to(dtype)
        sampler_head_probs = sampler_log_probs.exp()
        trainer_head_probs = trainer_head.detach().exp()
        sampler_head_mass = sampler_head_probs.sum(dim=-1)
        trainer_head_mass = trainer_head_probs.sum(dim=-1)
        sampler_tail = 1.0 - sampler_head_mass
        trainer_tail = 1.0 - trainer_head_mass
        tail_clipped = (sampler_tail < min_tail_mass) | (trainer_tail < min_tail_mass)
        tail_ratio = sampler_tail.clamp(min=min_tail_mass) / trainer_tail.clamp(min=min_tail_mass)
        # A tail token's ratio p / q is 1 / rho under the q = rho p approximation.
        alpha = tail_ratio * importance_weight(1.0 / tail_ratio, weight)
        head_weight = importance_weight((trainer_head.detach() - sampler_log_probs).exp(), weight)
        coefficients = sampler_head_probs * head_weight - alpha[:, None] * trainer_head_probs
    return CenteringTerm(
        term=advantages.to(dtype) * (coefficients * trainer_head).sum(dim=-1),
        correction=coefficients.abs().sum(dim=-1),
        sampler_head_mass=sampler_head_mass,
        trainer_head_mass=trainer_head_mass,
        tail_ratio=tail_ratio,
        tail_clipped=tail_clipped.to(dtype),
    )


def score_centering_term(
    args: Namespace,
    batch: dict[str, Any],
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
    weight: PolicyGradientWeight,
    settings: ScoreCenteringSettings,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The reduced term of one micro-batch and its metrics, as sums of per-sample means."""
    from megatron.core import mpu
    from slime.backends.megatron_utils.loss import get_responses

    if mpu.get_context_parallel_world_size() > 1:
        raise NotImplementedError("score centering supports context parallel = 1 only")
    for key in ("advantages", TOPK_INDICES_KEY, TOPK_LOG_PROBS_KEY):
        if batch.get(key) is None:
            raise ValueError(f"score centering needs {key} in the batch")
    tp_group = mpu.get_tensor_model_parallel_group()
    tp_world = dist.get_world_size(group=tp_group) if dist.is_initialized() else 1
    tp_rank = dist.get_rank(group=tp_group) if dist.is_initialized() else 0

    per_sample: list[CenteringTerm] = []
    responses = get_responses(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
    )
    for index, ((rows, _), head_ids, head_log_probs, advantages) in enumerate(
        zip(responses, batch[TOPK_INDICES_KEY], batch[TOPK_LOG_PROBS_KEY], batch["advantages"], strict=True)
    ):
        device = rows.device
        head_ids = head_ids.to(device=device, dtype=torch.long)
        if head_ids.size(0) != rows.size(0) or head_ids.size(-1) != settings.top_k:
            raise ValueError(
                f"score centering sample {index} has a {tuple(head_ids.shape)} head for a {rows.size(0)}-token "
                f"response and top_k={settings.top_k}"
            )
        vocab_size = rows.size(-1) * tp_world
        if head_ids.numel() and int(head_ids.max()) >= vocab_size:
            raise ValueError(f"score centering sample {index} names a token id outside the {vocab_size}-token vocab")
        per_sample.append(
            centering_term(
                gather_log_probs_at_ids(rows, head_ids, tp_group, tp_world, tp_rank),
                head_log_probs.to(device=device),
                advantages.to(device=device),
                weight,
                settings.min_tail_mass,
            )
        )
    term = sum_of_sample_mean(torch.cat([sample.term for sample in per_sample], dim=0))
    reported = {
        "correction_l1": [sample.correction for sample in per_sample],
        "sampler_head_mass": [sample.sampler_head_mass for sample in per_sample],
        "trainer_head_mass": [sample.trainer_head_mass for sample in per_sample],
        "tail_ratio": [sample.tail_ratio for sample in per_sample],
        "tail_clipped": [sample.tail_clipped for sample in per_sample],
    }
    metrics = {"score_centering_term": term.detach().clone()}
    for name, values in reported.items():
        metrics[f"score_centering_{name}"] = sum_of_sample_mean(torch.cat(values, dim=0))
    return term, metrics


def add_score_centering(
    args: Namespace,
    batch: dict[str, Any],
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
    loss: torch.Tensor,
    log: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The family's micro-batch ``loss`` and ``log`` with the term added, reduced like the loss."""
    settings = settings_from_args(args)
    if settings is None:
        raise RuntimeError("score centering was installed but --score-centering is off")
    weight = resolve_args_loss_family(args).policy_gradient_weight(args)
    if weight is None:
        raise RuntimeError(f"loss family {args.loss_family!r} declares no policy-gradient weight to center")
    term, metrics = score_centering_term(args, batch, logits, sum_of_sample_mean, weight, settings)
    total = loss + term
    combined = {**log, **metrics}
    if "loss" in log:
        combined["loss"] = total.detach().clone()
    return total, combined


def score_centered_custom_loss(
    args: Namespace,
    batch: dict[str, Any],
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """``--custom-loss-function-path`` of a custom-loss family with score centering: its loss plus the term."""
    from slime.utils.misc import load_function

    loss, log = load_function(args.reef_score_centering_base_loss_path)(args, batch, logits, sum_of_sample_mean)
    return add_score_centering(args, batch, logits, sum_of_sample_mean, loss, log)


__all__ = [
    "CenteringTerm",
    "add_score_centering",
    "centering_term",
    "importance_weight",
    "score_centered_custom_loss",
    "score_centering_term",
]
