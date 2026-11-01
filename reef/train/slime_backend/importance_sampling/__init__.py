"""The ``importance_sampling`` loss family on Slime: truncated importance sampling on Reef's advantages.

Tinker has a built-in loss of this name, which the generated-task objective
(``reef.recipe.generated_tasks.TaskGroupObjective``) trains with: each
rollout's advantage is its reward centered and scaled within its task's group.
This family is the same objective's loss on Slime's own rollouts, so a task job
trains group relative advantages on either backend.

The loss of one response token ``t`` is ``-A_t * sg[min(p_t / q_t, c)] * log p_t``,
where ``p`` is the trained policy, ``q`` the rollout engine's probability of the
sampled token (``rollout_log_probs``) and ``c`` Slime's ``--tis-clip``. Slime's
per-sample masked mean reduces it, as it reduces its own policy loss. It differs
from Tinker's loss in two ways, both deliberate: the ratio is truncated, and the
reduction is the per-sample mean, not the token sum. The truncated weight is
what score centering corrects (``--score-centering``): the term is zero for the
untruncated ratio, which Tinker's loss uses.

Only samples with recorded rollout log-probs train: those of the component's own
task rollouts. Reef computes the advantages, so Slime's advantage pass stays
off (``--disable-compute-advantages-and-returns``).
"""

from __future__ import annotations

import math
from argparse import Namespace

from reef.train.slime_backend.algorithm import (
    PolicyGradientWeight,
    SlimeAlgorithm,
    is_finite_number,
    register_loss_family,
)


@register_loss_family
class ImportanceSamplingAlgorithm(SlimeAlgorithm):
    """The ``importance_sampling`` family's spec: a custom loss on Reef's advantages, truncated at ``--tis-clip``."""

    loss_family = "importance_sampling"
    loss_type = "custom_loss"
    advantages = "required"
    required_objective_hooks = ("custom_loss_function_path",)

    def validate_specific_args(self, args: Namespace, source: str) -> None:
        cap = args.tis_clip
        if not is_finite_number(cap) or not 0 < float(cap) < math.inf:
            raise RuntimeError(f"{source} truncates the importance ratio at --tis-clip, which must be finite and > 0")

    def policy_gradient_weight(self, args: Namespace) -> PolicyGradientWeight:
        """``min(p / q, c)`` with ``c`` Slime's ``--tis-clip``."""
        return PolicyGradientWeight("truncated", upper=float(args.tis_clip))


__all__ = ["ImportanceSamplingAlgorithm"]
