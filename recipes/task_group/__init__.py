"""Task-group advantages for Harbor task jobs: each rollout's reward minus its task's mean reward.

A task job names it as its ``objective``, as the AutoEnvScaling and SPADE examples on Tinker do. ``tinker`` holds the
Tinker objective, on Tinker's built-in importance-sampling loss.
"""

from reef.train.algos.registry import register_objective_ref

register_objective_ref("task_group", "recipes.task_group.tinker:TaskGroupTinkerObjective", backend="tinker")
