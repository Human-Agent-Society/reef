"""OpenClaw-RL's paper objective (Eq. 1): top-K select loss, ported verbatim."""

from __future__ import annotations

import argparse
import math
from argparse import Namespace
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

from reef.core.job_data import TrajectoryItem
from reef.core.trajectories import source_record_id, trajectory_reward
from reef.train.slime_backend.algorithm import SlimeAlgorithm, register_loss_family
from reef.train.slime_backend.data_builder import build_policy_rollout_data

_HINT_SELECTIONS = ("shortest", "token_optimal", "sequence_optimal")
_SUBSET_MODES = ("student", "overlap", "teacher")
ROW_SHAPE = "[source_id, tokens, loss_mask, rollout_log_probs, reward, topk_indices, topk_log_probs, teacher_cands]"


@dataclass(frozen=True)
class OpenclawrlSettings:
    """Top-K objective knobs parsed from the ``--openclawrl-*`` flag family.

    Upstream ships these through ray's runtime-env; carrying them on the
    driver's argv and stamping them onto ``args`` means a Megatron actor can
    never silently run a different value than the one configured.
    """

    w_rl: float = 1.0
    w_opd: float = 1.0
    adv_diff_clip: float = 1.0
    hint_selection: str = "sequence_optimal"
    subset_mode: str = "student"

    def __post_init__(self) -> None:
        for name, value in (("w_rl", self.w_rl), ("w_opd", self.w_opd), ("adv_diff_clip", self.adv_diff_clip)):
            if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"openclawrl {name} must be a finite number")
        if self.hint_selection not in _HINT_SELECTIONS:
            raise ValueError(f"openclawrl hint_selection must be one of: {', '.join(_HINT_SELECTIONS)}")
        if self.subset_mode not in _SUBSET_MODES:
            raise ValueError(f"openclawrl subset_mode must be one of: {', '.join(_SUBSET_MODES)}")


def openclawrl_sample_row(sample: TrajectoryItem) -> list[Any]:
    """Shape one Reef sample into this family's 8-column wire row.

    Outbound mirror of :func:`build_openclawrl_rollout_data`.
    """
    return [
        source_record_id(sample),
        list(sample.training.get("tokens", [])),
        list(sample.training.get("loss_mask", [])),
        list(sample.training.get("rollout_log_probs", [])),
        trajectory_reward(sample),
        [list(row) for row in sample.training.get("topk_indices", [])],
        [list(row) for row in sample.training.get("topk_log_probs", [])],
        [dict(cand) for cand in sample.training.get("extras", {}).get("teacher_cands", ())],
    ]


def build_openclawrl_rollout_data(
    payload: Mapping[str, Any],
    samples: Sequence,
    spec: SlimeAlgorithm,
) -> dict:
    """Validate and convert the family's rows into Slime's external rollout payload.

    The three extra channels ride the family's wire row as columns 5-7 after the
    shared policy 5-tuple (``openclawrl_sample_row``); the builder unpacks them
    back into Slime's per-sample rollout keys:

    * ``topk_indices`` / ``topk_log_probs`` — the student's generation-time
      top-K vocab ids and log-probs per response token (``ell_old`` on S^q),
      captured by the serving backend at sampling time.
    * ``teacher_cands`` (carried in ``sample.training["extras"]``) — per
      candidate (accepted hints, shortest-first, up to the recipe's
      ``prm_max_hint_candidates``; or the un-enhanced anchor upstream ships for
      every RL-only turn), the candidate TOKEN SEQUENCE (upstream's
      ``teacher_tokens_candidates``: hint-enhanced prompt ids plus the native
      response ids verbatim). The train actor's Megatron teacher pass computes
      the ``prm_teacher_*_cand`` tensors from these before the loss runs;
      nothing numeric about the teacher rides the wire.

    Every sample carries at least one candidate — the processor retires
    candidate-less turns, matching upstream, so the builder treats an empty
    candidate list as a contract violation rather than something to repair.
    """
    base_rows: list[list[Any]] = []
    student_idx_rows: list[Any] = []
    student_lp_rows: list[Any] = []
    cand_rows: list[Any] = []
    for index, row in enumerate(samples):
        if not isinstance(row, Sequence) or isinstance(row, str | bytes) or len(row) != 8:
            raise ValueError(f"sample {index} must be {ROW_SHAPE}")
        base_rows.append(list(row[:5]))
        student_idx_rows.append(row[5])
        student_lp_rows.append(row[6])
        cand_rows.append(row[7])

    data = build_policy_rollout_data({**payload, "samples": base_rows}, base_rows, spec)
    response_lengths = data["response_lengths"]

    topk_indices: list[list[list[int]]] = []
    topk_log_probs: list[list[list[float]]] = []
    teacher_tokens_cand: list[list[list[int]]] = []

    for index, (s_idx, s_lp, cands, response_length) in enumerate(
        zip(student_idx_rows, student_lp_rows, cand_rows, response_lengths, strict=True)
    ):
        if len(s_idx) != response_length or len(s_lp) != response_length:
            raise ValueError(f"sample {index} top-K rows must cover the {response_length}-token response")
        k = len(s_idx[0]) if s_idx else 0
        if any(len(row) != k for row in s_idx) or any(len(row) != k for row in s_lp):
            raise ValueError(f"sample {index} top-K rows must be rectangular")
        topk_indices.append([[int(v) for v in row] for row in s_idx])
        topk_log_probs.append([[float(v) for v in row] for row in s_lp])

        if not cands:
            raise ValueError(f"sample {index} carries no teacher candidate; the processor must retire such turns")
        sample_tokens: list[list[int]] = []
        for cand in cands:
            tokens = cand.get("teacher_tokens")
            if not tokens or len(tokens) <= response_length:
                raise ValueError(
                    f"sample {index} candidate must be a token sequence longer than its {response_length}-token response"
                )
            sample_tokens.append([int(v) for v in tokens])
        teacher_tokens_cand.append(sample_tokens)

    data["topk_indices"] = topk_indices
    data["topk_log_probs"] = topk_log_probs
    # The numeric prm_teacher_*_cand tensors are filled trainer-side by the
    # Megatron teacher pass (recipes/openclawrl/slime/teacher.py);
    # the wire carries only the candidate sequences it forwards.
    data["teacher_tokens_cand"] = teacher_tokens_cand
    return data


@register_loss_family
class OpenclawrlAlgorithm(SlimeAlgorithm):
    loss_family = "openclawrl"
    loss_type = "custom_loss"
    requires_rollout_logprobs = False
    advantages = "required"
    # The worker's pre-train pass recomputes old-actor log-probabilities for
    # the GRPO branch. The objective's custom-advantage hook preserves the
    # externally supplied advantages while allowing that recomputation to run.
    allows_slime_advantage_computation = True
    required_objective_hooks = (
        "custom_loss_function_path",
        "custom_advantage_function_path",
        "reef_actor_init_hook_path",
        "reef_actor_pre_train_hook_path",
    )

    # Wire surface the adapter layer serves generically: per-sample payload
    # keys, their tensor dtypes, the keys Megatron's get_batch forwards, and
    # the non-scalar keys hidden from Slime's rollout logger.
    # ``teacher_tokens_cand`` is ragged (K_i variable-length candidate token
    # lists per sample), so it rides as plain lists with no declared dtype.
    rollout_data_keys = (
        "topk_log_probs",
        "topk_indices",
        "prm_teacher_topk_log_probs_cand",
        "prm_teacher_topk_indices_cand",
        "prm_teacher_native_topk_indices_cand",
        "teacher_tokens_cand",
    )
    rollout_tensor_dtypes: Mapping[str, str] = {
        "topk_log_probs": "float32",
        "topk_indices": "long",
        "prm_teacher_topk_log_probs_cand": "float32",
        "prm_teacher_topk_indices_cand": "long",
        "prm_teacher_native_topk_indices_cand": "long",
    }
    external_batch_keys = tuple(rollout_tensor_dtypes)
    rollout_log_skip_keys = tuple(rollout_tensor_dtypes)

    def validate_specific_args(self, args: Namespace, source: str) -> None:
        # The OPD branch gathers ell_old on S^q from the actor forward that
        # also produces ell_cur, which is exact only while the old actor IS
        # the current actor. Upstream holds the same invariant with an assert
        # in the loss; here the configurations that would break it are
        # refused up front rather than silently pinning rho to 1.
        if int(getattr(args, "num_steps_per_rollout", 1) or 1) != 1:
            raise RuntimeError(
                f"{source} requires --num-steps-per-rollout=1: the OPD branch's "
                "old-policy log-probs come from the actor forward, so a second optimizer "
                "step per rollout would make them the wrong policy's."
            )
        if getattr(args, "keep_old_actor", False):
            raise RuntimeError(
                f"{source} does not support --keep-old-actor: the OPD branch reads "
                "old-policy log-probs from the current actor forward."
            )

    def parse_specific_options(self, arguments: Sequence[str]) -> tuple[OpenclawrlSettings, list[str]]:
        # Canonical flags carry the package-name prefix (--openclawrl-*); the
        # historical --openclaw-topk-* spellings remain as deprecated aliases.
        parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, argument_default=argparse.SUPPRESS)
        parser.add_argument("--openclawrl-w-rl", "--openclaw-topk-w-rl", dest="w_rl", type=float)
        parser.add_argument("--openclawrl-w-opd", "--openclaw-topk-w-opd", dest="w_opd", type=float)
        parser.add_argument(
            "--openclawrl-adv-diff-clip",
            "--openclaw-topk-adv-diff-clip",
            dest="adv_diff_clip",
            type=float,
        )
        parser.add_argument(
            "--openclawrl-hint-selection",
            "--openclaw-topk-hint-selection",
            dest="hint_selection",
            choices=list(_HINT_SELECTIONS),
        )
        parser.add_argument(
            "--openclawrl-subset-mode",
            "--openclaw-topk-subset-mode",
            dest="subset_mode",
            choices=list(_SUBSET_MODES),
        )
        options, remaining = parser.parse_known_args(list(arguments))
        return OpenclawrlSettings(**vars(options)), remaining

    def apply_driver_options(self, args: Namespace, options: object | None) -> None:
        super().apply_driver_options(args, options)
        settings = options if isinstance(options, OpenclawrlSettings) else OpenclawrlSettings()
        args.openclawrl_w_rl = settings.w_rl
        args.openclawrl_w_opd = settings.w_opd
        args.openclawrl_adv_diff_clip = settings.adv_diff_clip
        args.openclawrl_hint_selection = settings.hint_selection
        args.openclawrl_subset_mode = settings.subset_mode

    def bind(self, config=None, *, critic_steps_per_actor=1, critic_only_steps=0):
        # The component forwards whatever parse_specific_options returned; the
        # knobs themselves travel on args, so the bound instance stays
        # stateless (same pattern as sao).
        if config is not None and not isinstance(config, OpenclawrlSettings):
            raise TypeError("openclawrl algorithm config must be OpenclawrlSettings")
        return self

    def configure_backend_args(self, args: Namespace) -> None:
        # The top-K objective needs an old-actor log-probability pass. Its
        # registered custom-advantage hook preserves the values supplied by
        # Reef.
        args.compute_advantages_and_returns = True

    def build_rollout_data(self, payload, samples):
        return build_openclawrl_rollout_data(payload, samples, self)

    def shape_sample_row(self, sample):
        return openclawrl_sample_row(sample)


__all__ = ["OpenclawrlAlgorithm", "OpenclawrlSettings", "build_openclawrl_rollout_data", "openclawrl_sample_row"]
