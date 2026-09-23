"""GRPO(+DIS) control arm for the SAO comparison (arXiv:2607.07508, Table 1).

Same DIS token-level ratio and double-sided mask as SAO, same rollout
log-probabilities as the behaviour proxy, no value model: advantages are
Slime's group-normalized GRPO estimate over ``--n-samples-per-prompt``
rollouts of one prompt. This is the paper's "GRPO (+ DIS)" row, the baseline
SAO is compared against once vanilla GRPO has collapsed.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from numbers import Real
from typing import Any

from recipes.sao.slime import SaoAlgorithm, build_sao_rollout_data, sao_rollout_metrics, sao_sample_row
from reef.train.slime_backend.algorithm import SlimeAlgorithm, TrainResult, register_loss_family
from reef.train.types import TrajectoryItem


@register_loss_family
class SaoGrpoControlAlgorithm(SlimeAlgorithm):
    loss_family = "sao-grpo-dis"
    loss_type = "policy_loss"
    requires_rollout_logprobs = True
    advantages = "forbidden"
    allows_slime_advantage_computation = True
    forbidden_advantages_message = (
        "the GRPO(+DIS) control computes group advantages in the training backend; the Reef payload must omit them"
    )
    # The wire surface is SAO's: the same columns reach the workers, only the
    # advantage estimate differs.
    rollout_data_keys = SaoAlgorithm.rollout_data_keys
    rollout_tensor_dtypes: Mapping[str, str] = dict(SaoAlgorithm.rollout_tensor_dtypes)
    response_aligned_keys = SaoAlgorithm.response_aligned_keys
    external_batch_keys = SaoAlgorithm.external_batch_keys
    rollout_log_skip_keys = SaoAlgorithm.rollout_log_skip_keys
    uses_pg_loss_primitive = True
    required_objective_hooks = ("custom_pg_loss_function_path",)

    def configure_backend_args(self, args: Namespace) -> None:
        # Slime's pre-train pass builds the GRPO advantages; nothing here
        # overrides them.
        args.compute_advantages_and_returns = True

    def validate_specific_args(self, args: Namespace, source: str) -> None:
        if getattr(args, "use_critic", False):
            raise RuntimeError(f"{source} is the critic-free GRPO(+DIS) control; do not pass --use-critic")
        # The adapter routes every pg-primitive family through Slime's cispo
        # callsite (configure_reef_loss_args sets advantage_estimator="cispo");
        # in the pinned Slime that lane's advantages are get_grpo_returns, the
        # same group normalization as "grpo", so both spellings are the control.
        if getattr(args, "advantage_estimator", None) not in ("grpo", "cispo"):
            raise RuntimeError(f"{source} requires --advantage-estimator=grpo")
        group = getattr(args, "n_samples_per_prompt", None)
        if not isinstance(group, int) or isinstance(group, bool) or group < 2:
            raise RuntimeError(f"{source} requires --n-samples-per-prompt >= 2 (a group to normalize over)")
        for name, low, high in (("eps_clip", 0.0, 1.0), ("eps_clip_high", 0.0, None)):
            value = getattr(args, name, None)
            if (
                not isinstance(value, Real)
                or isinstance(value, bool)
                or value < low
                or (high is not None and value >= high)
            ):
                raise RuntimeError(f"{source} requires --{name.replace('_', '-')} in the DIS range")

    def shape_sample_row(self, sample: TrajectoryItem) -> list[Any]:
        return sao_sample_row(sample)

    def build_rollout_data(self, payload: Mapping[str, Any], samples: Sequence) -> dict:
        return build_sao_rollout_data(payload, samples, self)

    def train(
        self,
        rollout_id: int,
        rollout_data_refs: Any,
        *,
        actor_group: Any,
        critic_group: Any,
        resolve: Callable[[Any], Any],
    ) -> TrainResult:
        if critic_group is not None:
            raise RuntimeError("the GRPO(+DIS) control was booted with a critic group; drop --use-critic")
        actor_results = list(resolve(actor_group.async_train(rollout_id, rollout_data_refs)) or ())
        return TrainResult(actor_results, {"control/actor_trained": 1})

    def rollout_metrics(self, rollout_data: dict[str, Any], serving_version: str) -> dict[str, Any]:
        return sao_rollout_metrics(rollout_data, serving_version)
