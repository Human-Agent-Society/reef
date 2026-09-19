"""The driver side of a distillation family: its settings, its wire row and the base every recipe subclasses.

A recipe's family is a thin subclass of :class:`DistillAlgorithm`: it names
itself (``loss_family``), carries its defaults in a :class:`DistillSettings`
subclass, and its ``objective.py`` forwards the worker hooks to
:mod:`.objective`. Everything else, the wire row, the flags, the validation,
lives here, so SDFT, SDPO and OPD differ only in their prefix, their
defaults and how their processors build the teacher's prompt.

The wire row is the policy row plus the sample's ``teacher_tokens``: the
teacher's prompt ids followed by the student's response ids verbatim (for
a teacher that reads no privileged prefix, the student's own sequence).
Alignment is exact by construction and checked here: a sequence whose tail
is not the student's response would make the teacher pass score the wrong
positions. Torch-free: the driver imports this module, the workers the
other two.
"""

from __future__ import annotations

import argparse
import math
from argparse import Namespace
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

from reef.core.trajectories import source_record_id, trajectory_reward
from reef.train.slime_backend.algorithm import SlimeAlgorithm
from reef.train.slime_backend.data_builder import build_policy_rollout_data
from reef.train.types import TrajectoryItem

#: ``self``: the student's own weights read a privileged prefix (the current
#: weights at update rate 1, an EMA of them below 1, a frozen snapshot at 0).
#: ``separate``: another model with the same tokenizer scores the student's
#: sequence; its Megatron checkpoint is named by ``teacher_checkpoint``.
TEACHER_SOURCES = ("self", "separate")
#: ``forward`` is KL(teacher || student), the SDFT reference's default and
#: what its paper's results used (GKD-style); ``reverse`` is KL(student ||
#: teacher); ``jsd`` is the generalized Jensen-Shannon divergence with
#: ``jsd_beta`` as the teacher's mixture weight.
DIVERGENCES = ("forward", "reverse", "jsd")


@dataclass(frozen=True)
class DistillSettings:
    """Driver options of a distillation family, parsed from its ``--<family>-*`` flags.

    A recipe's family subclasses this with its own defaults. ``top_k`` selects
    the teacher's representation: 0 keeps the teacher's whole next-token
    distribution at every response position (exact divergences, one
    ``[R, V_local]`` block per sample on the host); a positive value keeps
    the teacher's top-K log-probs and its log-prob at the sampled token
    (divergences restricted to those entries, the reverse one estimated at
    the sampled token).
    """

    teacher: str = "self"
    divergence: str = "forward"
    top_k: int = 0
    teacher_update_rate: float = 0.01
    teacher_checkpoint: str = ""
    importance_sampling_cap: float = 2.0
    skip_response_tokens: int = 0
    jsd_beta: float = 0.5

    def __post_init__(self) -> None:
        if self.teacher not in TEACHER_SOURCES:
            raise ValueError(f"distill teacher must be one of: {', '.join(TEACHER_SOURCES)}")
        if self.divergence not in DIVERGENCES:
            raise ValueError(f"distill divergence must be one of: {', '.join(DIVERGENCES)}")
        if not _is_integer(self.top_k) or self.top_k < 0:
            raise ValueError("distill top_k must be a non-negative integer (0 keeps the whole distribution)")
        if not _is_finite(self.teacher_update_rate) or not 0 <= self.teacher_update_rate <= 1:
            raise ValueError("distill teacher_update_rate must be a number in [0, 1]")
        if not isinstance(self.teacher_checkpoint, str):
            raise ValueError("distill teacher_checkpoint must be a path string")
        if self.teacher == "separate" and not self.teacher_checkpoint.strip():
            raise ValueError("distill teacher 'separate' needs teacher_checkpoint, the teacher's Megatron checkpoint")
        if not _is_finite(self.importance_sampling_cap) or self.importance_sampling_cap < 0:
            raise ValueError(
                "distill importance_sampling_cap must be a finite number >= 0 (0 disables the correction)"
            )
        if not _is_integer(self.skip_response_tokens) or self.skip_response_tokens < 0:
            raise ValueError("distill skip_response_tokens must be a non-negative integer")
        if not _is_finite(self.jsd_beta) or not 0 < self.jsd_beta < 1:
            raise ValueError("distill jsd_beta must be a number in (0, 1)")

    @property
    def exact(self) -> bool:
        """Whether the teacher's whole distribution is kept (``top_k == 0``)."""
        return self.top_k == 0


def _is_integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _is_finite(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


ROW_SHAPE = "[source_id, tokens, loss_mask, rollout_log_probs, reward, teacher_tokens]"


def distill_sample_row(sample: TrajectoryItem) -> list[Any]:
    """Shape one Reef sample into the family's 6-element wire row."""
    return [
        source_record_id(sample),
        list(sample.training.get("tokens", [])),
        list(sample.training.get("loss_mask", [])),
        list(sample.training.get("rollout_log_probs", [])),
        trajectory_reward(sample),
        list(sample.training.get("teacher_tokens", [])),
    ]


def build_distill_rollout_data(payload: Mapping[str, Any], samples: Sequence, spec: SlimeAlgorithm) -> dict:
    """Validate and convert the family's rows into Slime's external rollout payload."""
    name = spec.loss_family
    base_rows: list[list[Any]] = []
    teacher_rows: list[Any] = []
    for index, row in enumerate(samples):
        if not isinstance(row, Sequence) or isinstance(row, str | bytes) or len(row) != 6:
            raise ValueError(f"{name} sample {index} must be {ROW_SHAPE}")
        base_rows.append(list(row[:5]))
        teacher_rows.append(row[5])

    data = build_policy_rollout_data({**dict(payload), "samples": base_rows}, base_rows, spec)
    teacher_tokens: list[list[int]] = []
    for index, (row_teacher, tokens, response_length) in enumerate(
        zip(teacher_rows, data["tokens"], data["response_lengths"], strict=True)
    ):
        if (
            not isinstance(row_teacher, Sequence)
            or isinstance(row_teacher, str | bytes)
            or any(not isinstance(value, Integral) or isinstance(value, bool) for value in row_teacher)
        ):
            raise ValueError(f"{name} sample {index} teacher_tokens must be a sequence of integers")
        ids = [int(value) for value in row_teacher]
        if len(ids) <= response_length:
            raise ValueError(
                f"{name} sample {index} teacher sequence must carry a prompt before its {response_length}-token response"
            )
        if ids[-response_length:] != tokens[-response_length:]:
            raise ValueError(f"{name} sample {index} teacher sequence must end with the student's response ids")
        teacher_tokens.append(ids)
    data["teacher_tokens"] = teacher_tokens
    return data


#: The batch keys the teacher pass fills; the loss reads the ones its representation needs.
TEACHER_BATCH_KEYS = (
    "distill_teacher_log_probs",
    "distill_teacher_topk_ids",
    "distill_teacher_topk_log_probs",
    "distill_teacher_sampled_log_probs",
)


class DistillAlgorithm(SlimeAlgorithm):
    """A per-token divergence between the student and a teacher on the student's own samples.

    Before each step the pre-train hook scores every sample's teacher sequence
    with the teacher (the student's own weights reading a privileged prefix,
    or a separate model); the loss then puts the student's distribution at the
    same positions, from the training forward over the plain request, against
    it. The family's settings travel on ``args`` under ``distill_*`` names,
    whatever the recipe's flag prefix, so the hooks read one contract.
    """

    loss_type = "custom_loss"
    # The truncated importance-sampling weight compares the policy against the
    # rollout engine's log-probs; Reef ships them per row.
    requires_rollout_logprobs = True
    advantages = "forbidden"
    forbidden_advantages_message = (
        "a distillation family distils its teacher's distribution; the Reef payload must omit advantages"
    )
    rollout_data_keys = ("teacher_tokens",)
    rollout_tensor_dtypes: Mapping[str, str] = {"teacher_tokens": "long"}
    external_batch_keys = ("rollout_log_probs", *TEACHER_BATCH_KEYS)
    rollout_log_skip_keys = ("teacher_tokens", *TEACHER_BATCH_KEYS)
    required_objective_hooks = ("custom_loss_function_path", "reef_actor_pre_train_hook_path")
    #: The recipe's settings type, with its defaults.
    settings_type: type[DistillSettings] = DistillSettings

    # --- stage 1: configure ---

    def validate_specific_args(self, args: Namespace, source: str) -> None:
        # The teacher is scored once before the step from the weights the step
        # starts with; a second optimizer step per rollout would distil
        # distributions the first step already moved away from.
        # Slime leaves the option None for its default of one step.
        if int(args.num_steps_per_rollout or 1) != 1:
            raise RuntimeError(
                f"{source} requires --num-steps-per-rollout=1: the teacher's distributions are computed once "
                "before the step"
            )

    def parse_specific_options(self, arguments: Sequence[str]) -> tuple[DistillSettings, list[str]]:
        prefix = f"--{self.loss_family}-"
        defaults = self.settings_type()
        parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, argument_default=argparse.SUPPRESS)
        parser.add_argument(
            f"{prefix}teacher",
            dest="teacher",
            choices=list(TEACHER_SOURCES),
            help=(
                "Who scores the student's samples: 'self' is the student's own weights reading the privileged "
                "prefix, 'separate' another model with the same tokenizer (see teacher-checkpoint). "
                f"Default {defaults.teacher}."
            ),
        )
        parser.add_argument(
            f"{prefix}divergence",
            dest="divergence",
            choices=list(DIVERGENCES),
            help=(
                "The per-token divergence: 'forward' is KL(teacher || student), 'reverse' is KL(student || "
                f"teacher), 'jsd' the generalized Jensen-Shannon divergence. Default {defaults.divergence}."
            ),
        )
        parser.add_argument(
            f"{prefix}top-k",
            dest="top_k",
            type=int,
            help=(
                "Keep the teacher's top-K log-probs and its log-prob at the sampled token instead of its whole "
                f"distribution; 0 keeps the whole distribution. Default {defaults.top_k}."
            ),
        )
        parser.add_argument(
            f"{prefix}teacher-update-rate",
            dest="teacher_update_rate",
            type=float,
            help=(
                "For a 'self' teacher: the fraction of the current policy mixed into the teacher's weights after "
                "every step (the SDFT reference's ref_model_mixup_alpha). 1 makes the current policy the teacher, "
                f"0 freezes the initial weights. Default {defaults.teacher_update_rate}."
            ),
        )
        parser.add_argument(
            f"{prefix}teacher-checkpoint",
            dest="teacher_checkpoint",
            help="For a 'separate' teacher: its Megatron checkpoint, loaded beside the actor's weights.",
        )
        parser.add_argument(
            f"{prefix}importance-sampling-cap",
            dest="importance_sampling_cap",
            type=float,
            help=(
                "Cap of the truncated importance-sampling weight between the policy and the rollout engine's "
                f"log-probs, averaged over the response. 0 disables it. Default {defaults.importance_sampling_cap}."
            ),
        )
        parser.add_argument(
            f"{prefix}skip-response-tokens",
            dest="skip_response_tokens",
            type=int,
            help=(
                "Response tokens at the start of every sample left out of the loss and its denominator. "
                f"Default {defaults.skip_response_tokens}."
            ),
        )
        parser.add_argument(
            f"{prefix}jsd-beta",
            dest="jsd_beta",
            type=float,
            help=f"For 'jsd': the teacher's weight in the mixture. Default {defaults.jsd_beta}.",
        )
        options, remaining = parser.parse_known_args(list(arguments))
        return self.settings_type(**vars(options)), remaining

    def apply_driver_options(self, args: Namespace, options: object | None) -> None:
        super().apply_driver_options(args, options)
        settings = options if isinstance(options, DistillSettings) else self.settings_type()
        args.distill_teacher = settings.teacher
        args.distill_divergence = settings.divergence
        args.distill_top_k = settings.top_k
        args.distill_teacher_update_rate = settings.teacher_update_rate
        args.distill_teacher_checkpoint = settings.teacher_checkpoint
        args.distill_importance_sampling_cap = settings.importance_sampling_cap
        args.distill_skip_response_tokens = settings.skip_response_tokens
        args.distill_jsd_beta = settings.jsd_beta

    def bind(self, config=None, *, critic_steps_per_actor=None, critic_only_steps=0):
        # The settings travel on args; the bound instance stays stateless.
        if config is not None and not isinstance(config, DistillSettings):
            raise TypeError(f"{self.loss_family} bridge algorithm config must be {self.settings_type.__name__}")
        return self

    # --- stage 2: shape row ---

    def shape_sample_row(self, sample: TrajectoryItem) -> list[Any]:
        return distill_sample_row(sample)

    # --- stage 3: build batch ---

    def build_rollout_data(self, payload: Mapping[str, Any], samples: Sequence) -> dict:
        return build_distill_rollout_data(payload, samples, self)


def settings_from_args(args: Namespace) -> DistillSettings:
    """The family's settings as the driver stamped them on ``args`` (worker side)."""
    return DistillSettings(
        teacher=args.distill_teacher,
        divergence=args.distill_divergence,
        top_k=args.distill_top_k,
        teacher_update_rate=args.distill_teacher_update_rate,
        teacher_checkpoint=args.distill_teacher_checkpoint,
        importance_sampling_cap=args.distill_importance_sampling_cap,
        skip_response_tokens=args.distill_skip_response_tokens,
        jsd_beta=args.distill_jsd_beta,
    )


__all__ = [
    "DIVERGENCES",
    "TEACHER_BATCH_KEYS",
    "TEACHER_SOURCES",
    "DistillAlgorithm",
    "DistillSettings",
    "build_distill_rollout_data",
    "distill_sample_row",
    "settings_from_args",
]
