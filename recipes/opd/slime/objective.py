"""OPD worker entry points delegate tensor operations to the shared backend."""

from argparse import Namespace
from collections.abc import Callable
from typing import Any

from reef.train.slime_backend.algorithm import objective


@objective("custom_loss_function_path")
def opd_loss(
    args: Namespace, batch: dict[str, Any], logits: Any, sum_of_sample_mean: Callable[[Any], Any]
) -> tuple[Any, dict[str, Any]]:
    """Minimize the teacher divergence over the student's generated tokens."""
    from reef.train.slime_backend.distill.objective import distill_loss

    return distill_loss(args, batch, logits, sum_of_sample_mean)


@objective("reef_actor_pre_train_hook_path")
def opd_actor_pre_train(actor: Any, rollout_data: dict[str, Any]) -> None:
    """Score the batch with the frozen teacher before the student update."""
    from reef.train.slime_backend.distill.objective import distill_actor_pre_train

    distill_actor_pre_train(actor, rollout_data)
