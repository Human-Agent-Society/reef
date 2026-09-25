"""SDPO's worker hooks: the backend's distillation loss and teacher pass under this family's names.

Slime's Megatron workers resolve the hooks by path; the base's
``reef.train.slime_backend.distill.objective`` does the work, and these entry
points exist so the family's hooks carry its name, as the loss-family
contract asks. Torch is imported only when a hook runs.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable
from typing import Any

from reef.train.slime_backend.algorithm import objective


@objective("custom_loss_function_path")
def sdpo_loss(
    args: Namespace, batch: dict[str, Any], logits: Any, sum_of_sample_mean: Callable[[Any], Any]
) -> tuple[Any, dict[str, Any]]:
    """``--custom-loss-function-path`` entry point: the weighted per-sample mean token divergence to the self-teacher."""
    from reef.train.slime_backend.distill.objective import distill_loss

    return distill_loss(args, batch, logits, sum_of_sample_mean)


@objective("reef_actor_pre_train_hook_path")
def sdpo_actor_pre_train(actor: Any, rollout_data: dict[str, Any]) -> None:
    """Move the teacher toward the policy, select the student's top-K ids, then score every teacher sequence."""
    from reef.train.slime_backend.distill.objective import distill_actor_pre_train

    distill_actor_pre_train(actor, rollout_data)
