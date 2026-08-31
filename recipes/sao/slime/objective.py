"""Tensor implementation of the SAO objective (arXiv:2607.07508).

Loaded lazily by the package wrappers in
``recipes.sao.slime`` so importing the package never
requires torch. Slime's Megatron workers resolve the hook functions by path
(``--custom-advantage-function-path`` / ``--custom-pg-loss-function-path``),
which keeps the vendored tree free of SAO-specific branches. CPU parity tests
compare these tensor transcriptions against an independent pure-Python oracle
(``tests/reef_service/reference_algorithms/sao.py``).
"""

from __future__ import annotations

from argparse import Namespace
from typing import Any

import torch

from reef.train.slime_backend.algorithm import objective


@torch.compile(dynamic=True)
def compute_sao_loss(
    ppo_kl: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SAO policy loss (arXiv:2607.07508, Eqs. 1-3).

    ``ppo_kl = logπrollout − logπθ`` (the caller passes rollout log-probs as the
    old policy), so ``ratio = exp(−ppo_kl) = exp(logπθ − logπrollout)`` is the DIS
    ratio ``r_t`` of Eq. 2. The double-sided calibration ``f(r_t; ε_l, ε_h)``
    (Eq. 3) keeps the ratio strictly inside ``(1 − ε_l, 1 + ε_h)`` and hard-masks
    everything else to zero — a token outside the trust region contributes no
    policy gradient. ``f`` is applied under stop-gradient, so the surrogate
    ``−sg(f(r_t))·Â_t·logπθ`` yields the exact SAO gradient
    ``sg(f(r_t))·Â_t·∇logπθ`` (Eq. 1) on descent.

    Bounds follow the same delta-from-1 convention as Slime's
    ``compute_policy_loss``: the lower bound is ``1 − eps_clip`` and the upper
    bound is ``1 + eps_clip_high``. Both are strict and covered by an
    independent pure-Python parity oracle. Returns ``(pg_losses, clipfrac)``,
    where ``clipfrac`` is the per-token masked indicator.
    """
    ratio = (-ppo_kl).exp()
    keep = (ratio > 1.0 - eps_clip) & (ratio < 1.0 + eps_clip_high)
    calibration = torch.where(keep, ratio, torch.zeros_like(ratio)).detach()
    pg_losses = -calibration * advantages * log_probs
    clipfrac = (~keep).float()
    return pg_losses, clipfrac


@objective("custom_pg_loss_function_path")
def sao_loss(
    args: Namespace,
    ppo_kl: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapt :func:`compute_sao_loss` to Slime's generic pg-loss hook.

    Slime's ``policy_loss_function`` calls ``--custom-pg-loss-function-path``
    with ``(args, ppo_kl, log_probs, advantages)``; the clip bounds ride on
    ``args`` exactly as they do for the built-in objectives.
    """
    return compute_sao_loss(ppo_kl, log_probs, advantages, args.eps_clip, args.eps_clip_high)


def get_skip_observation_advantages_and_returns(
    total_len: int,
    response_len: int,
    values: torch.Tensor,
    reward: float,
    action_mask: torch.Tensor,
    gamma: float,
    lambd: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Skip-observation token-level GAE for SAO (arXiv:2607.07508, Eqs. 4-5).

    ``action_mask[t] == 1`` marks a model-generated (action) token; ``0`` marks
    an environment/observation token. The TD recurrence runs over action tokens
    only, so the residual for the last token of an action bridges directly to
    the first token of the *next* action, skipping the observation span between
    them. The value model is therefore never regressed against an environment
    state it did not produce. CPU parity tests compare this tensor
    transcription against an independent pure-Python oracle.

    The scalar rollout ``reward`` lands on the last action token of the *full*
    response, placed after the context-parallel all-gather so it is not lost or
    duplicated when a rank holds only a slice. ``advantages``/``returns`` are
    zero on observation tokens (never trained), and ``returns = advantages +
    values`` on action tokens.
    """
    from megatron.core import mpu

    cp_size = mpu.get_context_parallel_world_size()
    if cp_size > 1:
        from slime.backends.megatron_utils.cp_utils import all_gather_with_cp

        full_values = all_gather_with_cp(values, total_len, response_len)
        full_action_mask = all_gather_with_cp(action_mask, total_len, response_len)
    else:
        full_values = values
        full_action_mask = action_mask

    # The scalar reward is a terminal reward on the last action token of the
    # whole response; build per-token rewards on the gathered sequence.
    full_rewards = torch.zeros_like(full_values)
    action_positions = torch.nonzero(full_action_mask, as_tuple=False)
    if action_positions.numel() > 0:
        full_rewards[int(action_positions[-1].item())] = float(reward)

    advantages = torch.zeros_like(full_values)
    returns = torch.zeros_like(full_values)
    last_gae = 0.0
    # Value of the next action token (a_i+1,0); 0.0 for the final action token
    # because there is no subsequent action to bridge to.
    next_value = 0.0
    for t in reversed(range(response_len)):
        if full_action_mask[t] == 0:
            continue
        delta = full_rewards[t] + gamma * next_value - full_values[t]
        last_gae = delta + gamma * lambd * last_gae
        advantages[t] = last_gae
        returns[t] = last_gae + full_values[t]
        next_value = full_values[t]

    if cp_size > 1:
        from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

        advantages = slice_log_prob_with_cp(advantages, total_len, response_len)
        returns = slice_log_prob_with_cp(returns, total_len, response_len)

    return advantages.detach(), returns


@objective("custom_advantage_function_path")
def sao_advantages(args: Namespace, rollout_data: dict[str, Any]) -> None:
    """Populate SAO advantages/returns via Slime's custom-advantage hook.

    Single-Rollout Asynchronous Optimization (arXiv:2607.07508):
    skip-observation GAE over action tokens with the critic's values. The
    scalar rollout reward lands on the last action token; observation tokens
    are excluded from the recurrence so the value model never has to predict
    an environment state it did not generate. Sets
    ``rollout_data["advantages"]`` and ``rollout_data["returns"]`` in place,
    as ``--custom-advantage-function-path`` requires.
    """
    values = rollout_data.get("values")
    if values is None:
        raise ValueError("sao advantage computation requires critic values in rollout_data")
    rewards = rollout_data["rewards"]
    response_lengths = rollout_data["response_lengths"]
    total_lengths = rollout_data["total_lengths"]
    action_masks = rollout_data.get("action_masks")
    if action_masks is None:
        # A rollout without an explicit action mask is a pure model
        # trajectory: every response token is an action token.
        action_masks = [torch.ones_like(value) for value in values]
    use_length_adaptive = getattr(args, "sao_length_adaptive_lambda", False)
    advantages: list[torch.Tensor] = []
    returns: list[torch.Tensor] = []
    for i in range(len(values)):
        if use_length_adaptive:
            # Length-adaptive lambda (Section 4): longer responses use a
            # lambda closer to 1 (less bias). Keyed on response_lengths[i],
            # a context-parallel-independent scalar, so the value matches
            # across CP ranks (a local mask sum would not).
            length = int(response_lengths[i])
            alpha = getattr(args, "sao_lambda_alpha", 1.5)
            sample_lambda = 1.0 - 1.0 / (alpha * length) if length > 0 else 0.0
            sample_lambda = min(max(sample_lambda, 0.0), 1.0)
        else:
            sample_lambda = args.lambd
        adv_i, ret_i = get_skip_observation_advantages_and_returns(
            total_lengths[i],
            response_lengths[i],
            values[i],
            float(rewards[i]),
            action_masks[i],
            args.gamma,
            sample_lambda,
        )
        advantages.append(adv_i)
        returns.append(ret_i)
    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns
