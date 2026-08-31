"""SAO train scheduling: critic warmup cadence and actor train plan."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_CRITIC_STEPS_PER_ACTOR = 2


@dataclass(frozen=True)
class SaoStepPlan:
    critic_updates: int
    train_actor: bool


@dataclass(frozen=True)
class SaoSchedule:
    critic_steps_per_actor: int = DEFAULT_CRITIC_STEPS_PER_ACTOR
    critic_only_steps: int = 0

    def __post_init__(self) -> None:
        if self.critic_steps_per_actor < 1:
            raise ValueError("critic_steps_per_actor (K) must be >= 1")
        if self.critic_only_steps < 0:
            raise ValueError("critic_only_steps must be >= 0")

    def plan(self, rollout_index: int) -> SaoStepPlan:
        if rollout_index < 0:
            raise ValueError("rollout_index must be >= 0")
        return SaoStepPlan(self.critic_steps_per_actor, rollout_index >= self.critic_only_steps)
