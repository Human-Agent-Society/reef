"""Slime implementation of standard PPO/RLHF reference-reward training."""

from __future__ import annotations

import math
from argparse import Namespace
from numbers import Real

from reef.train.slime_backend.algorithm import SlimeAlgorithm, TrainResult, register_loss_family


@register_loss_family
class PpoRlhfAlgorithm(SlimeAlgorithm):
    """Keep Slime's native PPO loss and reference-reward advantage path."""

    loss_family = "ppo_rlhf_reference_reward"
    loss_type = "policy_loss"
    requires_rollout_logprobs = True
    advantages = "forbidden"
    allows_slime_advantage_computation = True
    forbidden_advantages_message = (
        "ppo_rlhf_reference_reward advantages are computed by the Slime value model; "
        "the Reef payload must omit them"
    )

    def bind(self, config=None, *, critic_steps_per_actor=None, critic_only_steps=0):
        if config is not None:
            raise TypeError("ppo_rlhf_reference_reward does not accept bridge algorithm config")
        steps = 1 if critic_steps_per_actor is None else critic_steps_per_actor
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("critic_steps_per_actor must be a positive integer")
        if isinstance(critic_only_steps, bool) or not isinstance(critic_only_steps, int) or critic_only_steps < 0:
            raise ValueError("critic_only_steps must be a non-negative integer")
        bound = type(self)()
        bound._critic_steps_per_actor = steps
        bound._critic_only_steps = critic_only_steps
        return bound

    def configure_backend_args(self, args: Namespace) -> None:
        args.compute_advantages_and_returns = True
        args.advantage_estimator = "ppo"

    def validate_specific_args(self, args: Namespace, source: str) -> None:
        if not getattr(args, "use_critic", False):
            raise RuntimeError(
                f"{source} requires a value model; pass --use-critic so the PPO advantage pass has a critic"
            )
        if not getattr(args, "compute_advantages_and_returns", False):
            raise RuntimeError(f"{source} requires Slime advantage computation")
        if getattr(args, "advantage_estimator", None) != "ppo":
            raise RuntimeError(f"{source} requires advantage_estimator=ppo")
        if getattr(args, "use_kl_loss", False) or getattr(args, "use_opd", False):
            raise RuntimeError(
                f"{source} owns reference-policy KL on the reward/advantage path; "
                "loss-side KL and OPD cannot be combined with it"
            )
        kl_coef = getattr(args, "kl_coef", None)
        if (
            not isinstance(kl_coef, Real)
            or isinstance(kl_coef, bool)
            or not math.isfinite(float(kl_coef))
            or kl_coef <= 0
        ):
            raise RuntimeError(
                f"{source} requires a positive finite --kl-coef for reference-policy reward shaping"
            )

    def train(
        self,
        rollout_id,
        rollout_data_refs,
        *,
        actor_group,
        critic_group,
        resolve,
    ) -> TrainResult:
        if critic_group is None:
            raise RuntimeError("ppo_rlhf_reference_reward requires a critic group")
        critic_values = None
        critic_steps = getattr(self, "_critic_steps_per_actor", 1)
        for _ in range(critic_steps):
            critic_values = resolve(critic_group.async_train(rollout_id, rollout_data_refs))
        train_actor = rollout_id >= getattr(self, "_critic_only_steps", 0)
        actor_results = []
        if train_actor:
            actor_results = list(
                resolve(actor_group.async_train(rollout_id, rollout_data_refs, external_data=critic_values)) or ()
            )
        return TrainResult(
            actor_results,
            {
                "ppo_rlhf/critic_updates": critic_steps,
                "ppo_rlhf/actor_trained": int(train_actor),
            },
        )


__all__ = ["PpoRlhfAlgorithm"]
