"""Score centering on the Slime backend: a correction term Reef adds to a compatible family's own loss.

Score centering (Marek and Ryabinin, arXiv:2609.20807) removes the drift an
off-policy policy gradient accumulates when the sampler ``q`` differs from
the trained policy ``p`` (quantized inference, stale weights). A loss whose
per-token gradient is ``A_t * f(p_t / q_t) * grad log p_t`` pulls ``p``
toward ``q`` by ``A_t * E_q[f(p / q) * score]`` at every prefix; the term
subtracts it (the paper's Appendix A, equations 9-14)::

    A_t * sum_{v in head} sg[q_v * f(p_v / q_v) - alpha * p_v] * log p_v

The head is the sampler's recorded top-K ids; the sampler's tail is
approximated as ``rho * p`` with ``rho = (1 - q(head)) / (1 - p(head))`` and
``alpha = rho * f(1 / rho)``. The term is zero when ``q = p``.

It is not a loss family. ``--score-centering`` (with ``--score-centering-top-k``
and ``--score-centering-min-tail-mass``) in ``training.options`` adds the term
to whatever loss the recipe's family computes. The term must mirror that loss
(its advantages, weight ``f``, mask and reduction), so the family declares
``f`` (:meth:`SlimeAlgorithm.policy_gradient_weight`); a family that declares
none is refused.

- this module, torch-free: the term's settings, the driver-side checks and
  the rollout keys, and the payload columns (the sampler's top-K per sample,
  in the order the schedule trains the rows);
- :mod:`.heads`, torch, on the bridge: the check of the recorded heads;
- :mod:`.term`, torch, on the workers: the term and the wrapper that adds it
  to the family's loss.

The sampler's log-probs are those at the point the trainer reads (after
temperature, before top-k, top-p and min-p filters; see the configuration
reference), so records must come from a token-native handler with
``capture_topk`` at least ``--score-centering-top-k``.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

from reef.train.slime_backend.algorithm import SlimeAlgorithm, is_finite_number
from reef.train.types import TrajectoryItem

#: Payload and rollout-data keys of the sampler's top-K: one ``[R, top_k]`` block per sample.
TOPK_INDICES_KEY = "sampler_topk_indices"
TOPK_LOG_PROBS_KEY = "sampler_topk_log_probs"
ROLLOUT_KEYS = (TOPK_INDICES_KEY, TOPK_LOG_PROBS_KEY)


@dataclass(frozen=True)
class ScoreCenteringSettings:
    """The term's own settings.

    ``top_k`` is the number of sampler log-probs per position the term reads;
    a record must carry at least that many (the inference handler's
    ``capture_topk``). The paper's default is 128; a smaller head leaves more
    of the drift uncorrected when the sampler's tail differs from the
    trainer's. ``min_tail_mass`` is the floor both tail masses are clipped to
    before their ratio is taken (the paper uses ``1e-6``).
    """

    top_k: int = 128
    min_tail_mass: float = 1e-6

    def __post_init__(self) -> None:
        if not isinstance(self.top_k, Integral) or isinstance(self.top_k, bool) or self.top_k <= 0:
            raise ValueError("score centering top_k must be a positive integer")
        if not is_finite_number(self.min_tail_mass) or not 0 < self.min_tail_mass < 1:
            raise ValueError("score centering min_tail_mass must be a number in (0, 1)")


def settings_from_args(args: Namespace) -> ScoreCenteringSettings | None:
    """The term's settings from the parsed driver arguments, or ``None`` when it is off."""
    if not args.score_centering:
        return None
    return ScoreCenteringSettings(top_k=args.score_centering_top_k, min_tail_mass=args.score_centering_min_tail_mass)


def configure_score_centering(args: Namespace, spec: SlimeAlgorithm) -> None:
    """Refuse configurations the term cannot correct, and declare its rollout keys (driver side).

    Runs after the family's own wire declarations are on ``args``.
    """
    if settings_from_args(args) is None:
        return
    source = f"--score-centering with loss family {spec.loss_family!r}"
    if spec.policy_gradient_weight(args) is None:
        raise RuntimeError(
            f"{source}: the family does not declare the weight of its policy-gradient loss "
            "(SlimeAlgorithm.policy_gradient_weight), so no correction can match it"
        )
    if int(args.context_parallel_size or 1) != 1:
        raise RuntimeError(f"{source} supports --context-parallel-size 1 only: the top-k rows are not CP-sliced")
    if spec.loss_type == "policy_loss":
        # Slime's policy loss reweights, masks or re-reduces the policy-gradient
        # term under these options, beyond the weight the family declares, so the
        # correction would center the wrong score.
        altering = {
            "--use-tis": args.use_tis,
            "--get-mismatch-metrics": args.get_mismatch_metrics,
            "--use-opsm": args.use_opsm,
            "--custom-pg-loss-reducer-function-path": args.custom_pg_loss_reducer_function_path,
        }
        flags = [flag for flag, value in altering.items() if value]
        if flags:
            raise RuntimeError(
                f"{source} cannot be combined with {', '.join(flags)}: Slime changes the policy-gradient term with it"
            )
    args.custom_rollout_data_keys = tuple(dict.fromkeys((*(args.custom_rollout_data_keys or ()), *ROLLOUT_KEYS)))
    args.reef_rollout_tensor_dtypes = {
        **(args.reef_rollout_tensor_dtypes or {}),
        TOPK_INDICES_KEY: "long",
        TOPK_LOG_PROBS_KEY: "float32",
    }
    args.reef_external_batch_keys = tuple(dict.fromkeys((*args.reef_external_batch_keys, *ROLLOUT_KEYS)))
    args.reef_rollout_log_skip_keys = tuple(dict.fromkeys((*args.reef_rollout_log_skip_keys, *ROLLOUT_KEYS)))


def sampler_topk_columns(samples: Sequence[TrajectoryItem], row_indices: Sequence[int]) -> dict[str, list[Any]]:
    """The payload's top-K columns: each wire row's recorded ``topk_indices`` and ``topk_log_probs``."""
    return {
        TOPK_INDICES_KEY: [list(samples[row].training.get("topk_indices", [])) for row in row_indices],
        TOPK_LOG_PROBS_KEY: [list(samples[row].training.get("topk_log_probs", [])) for row in row_indices],
    }


__all__ = [
    "ROLLOUT_KEYS",
    "TOPK_INDICES_KEY",
    "TOPK_LOG_PROBS_KEY",
    "ScoreCenteringSettings",
    "configure_score_centering",
    "sampler_topk_columns",
    "settings_from_args",
]
