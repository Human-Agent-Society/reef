"""The ``importance_sampling`` loss family: Tinker's built-in loss of the same name, on Slime.

Slime has this family without a method package, as Tinker has its built-in
loss (``reef.train.slime_backend.loss_families`` registers it). A recipe whose
objective names ``importance_sampling``, such as the generated-task recipe,
therefore trains on either backend.

- this module: the spec, driver side and torch free.
- ``objective``: the worker's loss hook and the token loss that TTTD also uses.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Mapping, Sequence

from reef.train.slime_backend.algorithm import SlimeAlgorithm, register_loss_family
from reef.train.slime_backend.data_builder import build_policy_rollout_data


@register_loss_family
class ImportanceSamplingAlgorithm(SlimeAlgorithm):
    """Tinker's ``importance_sampling`` loss: each sample's advantage on the response tokens that its loss mask selects.

    Values: the one instance that ``@register_loss_family`` makes; it has no
    fields. The step loss is ``-sum_i sum_t exp(log p_it - log q_it) * A_i * m_it``
    over the step's samples ``i`` and response tokens ``t``, with ``p`` the
    trainer's probability, ``q`` the rollout probability, ``A_i`` the sample's
    advantage and ``m_it`` its loss mask. It has no ratio clip and no KL term.
    Purpose: the Slime registry resolves it for a job whose objective names
    ``importance_sampling``. The driver checks the Slime flags with it, and
    the bridge builds the job's rollout data with it.
    """

    loss_family = "importance_sampling"
    loss_type = "custom_loss"
    requires_rollout_logprobs = True
    advantages = "required"

    def validate_specific_args(self, args: Namespace, source: str) -> None:
        """Raise ``RuntimeError`` when ``args`` sets a KL term, which the loss does not have, or Slime's per-token
        reduction, which would divide each step's loss by its token count."""
        # With a KL term Slime also loads a reference model that this loss never reads.
        if args.kl_coef or args.use_kl_loss:
            raise RuntimeError(
                f"{source} trains importance_sampling, which has no KL penalty: "
                "set --kl-coef 0 and remove --use-kl-loss"
            )
        # The loss cancels Slime's division by the step's sample count; the per-token reduction divides by the
        # step's token count instead, so the step would not match Tinker's.
        if args.calculate_per_token_loss:
            raise RuntimeError(
                f"{source} trains importance_sampling, which sums over tokens as Tinker does: "
                "omit --calculate-per-token-loss"
            )

    def build_rollout_data(self, payload: Mapping[str, object], samples: Sequence[object]) -> dict[str, object]:
        """The default policy rollout data, with ``advantages[i][t] = A_i * loss_masks[i][t]``.

        Tinker's built-in loss gets the same advantages
        (``reef.train.tinker_backend.losses.ImportanceSamplingLoss``), so a
        token outside the loss mask does not train.
        """
        data = build_policy_rollout_data(payload, samples, self)
        if "advantages" in data:
            data["advantages"] = [
                [advantage * mask for advantage, mask in zip(advantages, loss_mask, strict=True)]
                for advantages, loss_mask in zip(data["advantages"], data["loss_masks"], strict=True)
            ]
        return data


__all__ = ["ImportanceSamplingAlgorithm"]
