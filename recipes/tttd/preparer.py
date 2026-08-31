"""TTT-Discover step preparer: grouped adaptive-entropic advantages."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from reef.train.algos.base import StepPreparer, register_step_preparer
from reef.train.algos.helpers import next_steps
from reef.train.algos.signals import StepScheduling, StepSignal
from reef.train.types import GroupedPolicyBatch, TrainingBatch


@register_step_preparer
class TttdPreparer(StepPreparer):
    name = "tttd"

    @staticmethod
    def adaptive_entropic_advantages(rewards: list[float]) -> tuple[tuple[float, ...], float]:
        """Return the reference TTT-Discover LOO advantages and solved beta.

        This is a direct port of
        ``test-time-training/discover@6c40e82/ttt_discover/rl/train.py`` for
        ``adv_estimator == "entropic_adaptive_beta"``. Beta is selected so
        ``KL(softmax(beta * rewards) || uniform) == log(2)`` using the same
        expansion bound, 60 bisection iterations, stable reward shift, and
        leave-one-out normalizer.
        """
        if not rewards:
            raise ValueError("an adaptive-entropic group cannot be empty")

        delta = math.log(2)
        beta_max = 1e6
        iterations = 60
        epsilon = 1e-12

        group_size = len(rewards)
        max_reward = max(rewards)

        beta: float | None
        if group_size < 2:
            beta = 0.0
        else:
            log_group_size = math.log(group_size)

            def estimated_kl(beta_scalar: float) -> float:
                logits = [beta_scalar * (r - max_reward) for r in rewards]
                max_logit = max(logits)
                exp_shifted = [math.exp(lg - max_logit) for lg in logits]
                sum_exp = sum(exp_shifted)
                log_sum_exp = math.log(sum_exp) + max_logit
                kl = 0.0
                for i in range(group_size):
                    prob = exp_shifted[i] / sum_exp
                    log_prob = logits[i] - log_sum_exp
                    kl += prob * (log_prob + log_group_size)
                return kl

            low, high = 0.0, 1.0
            if estimated_kl(high) < delta:
                while high < beta_max and estimated_kl(high) < delta:
                    high *= 2.0
                beta = high if estimated_kl(high) < delta else None
            else:
                beta = None

            if beta is None:
                for _ in range(iterations):
                    midpoint = 0.5 * (low + high)
                    if estimated_kl(midpoint) < delta:
                        low = midpoint
                    else:
                        high = midpoint
                beta = high

        if beta is None:
            raise RuntimeError("adaptive beta search did not produce a value")
        exponentials = [math.exp(beta * (r - max_reward)) for r in rewards]
        if group_size == 1:
            normalizers = exponentials
        else:
            total = sum(exponentials)
            normalizers = [(total - e) / (group_size - 1) for e in exponentials]
        advantages = tuple(e / (n + epsilon) - 1.0 for e, n in zip(exponentials, normalizers, strict=True))
        return advantages, beta

    def __call__(self, batch: TrainingBatch, state: Mapping[str, Any]) -> StepSignal:
        if not isinstance(batch, GroupedPolicyBatch):
            raise TypeError(f"{self.name} requires GroupedPolicyBatch, got {type(batch).__name__}")
        advantages: list[float] = []
        betas: list[float] = []
        for comparison_set in batch.comparison_sets:
            group_advantages, beta = self.adaptive_entropic_advantages([sample.reward for sample in comparison_set])
            advantages.extend(group_advantages)
            betas.append(beta)
        steps = next_steps(state)
        normalized = tuple(advantages)
        # The processor drops constant-reward groups but keeps one when every
        # group is constant; flag that batch (its advantages carry no signal).
        constant_groups = all(len({sample.reward for sample in group}) == 1 for group in batch.comparison_sets)
        return StepSignal(
            "train",
            self.name,
            {"steps": steps},
            {
                "advantages": normalized,
                "adaptive_betas": tuple(betas),
                "constant_groups_retained": int(constant_groups),
                "steps": steps,
            },
            normalized,
            StepScheduling(unit="sample", batch_size="actual"),
        )
