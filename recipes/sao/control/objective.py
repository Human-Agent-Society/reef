"""Torch objective for the GRPO(+DIS) control arm.

The only hook this family registers is the per-token DIS policy-gradient
primitive, shared verbatim with the SAO arm so the two objectives differ in
nothing but the advantage estimate. There is deliberately no advantage hook:
without a critic the actor takes Slime's stock ``--advantage-estimator=grpo``
group-normalized advantages from its pre-train pass.
"""

from __future__ import annotations

from argparse import Namespace

import torch

from recipes.sao.slime.objective import compute_sao_loss
from reef.train.slime_backend.algorithm import objective


@objective("custom_pg_loss_function_path")
def control_dis_loss(
    args: Namespace,
    ppo_kl: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return compute_sao_loss(ppo_kl, log_probs, advantages, args.eps_clip, args.eps_clip_high)
