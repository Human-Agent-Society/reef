"""The driver side of a distillation family: its settings, its wire row and the base every recipe subclasses.

A recipe's family is a thin subclass of :class:`DistillAlgorithm`: it names
itself (``loss_family``), carries its defaults in a :class:`DistillSettings`
subclass, and its ``objective.py`` forwards the worker hooks to
:mod:`.objective`. Everything else, the wire row, the flags, the validation,
lives here, so SDFT, SDPO and OPD differ only in their prefix, their
defaults and how their processors build the teacher's prompt.

The wire row is the policy row plus the sample's ``teacher_tokens``, the
teacher's prompt ids followed by the student's response ids verbatim (for
a teacher that reads no privileged prefix, the student's own sequence), and
its ``sample_weight``, the factor the loss puts on that sample's mean token
divergence (1 for an ordinary sample; SDPO gives a rollout whose teacher
read no privileged information 0, so it stays in the step's mean without
a target). Alignment is exact by construction and checked here: a sequence
whose tail is not the student's response would make the teacher pass score
the wrong positions. Torch-free: the driver imports this module, the
workers the other two.
"""

from __future__ import annotations

import argparse
import math
from argparse import Namespace
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from numbers import Integral, Real
from typing import Any
from urllib.parse import urlsplit

from reef.core.trajectories import source_record_id, trajectory_reward
from reef.train.slime_backend.algorithm import PolicyGradientWeight, SlimeAlgorithm
from reef.train.slime_backend.data_builder import build_policy_rollout_data
from reef.train.types import TrajectoryItem

#: ``self``: the student's own weights read a privileged prefix (the current
#: weights at update rate 1, an EMA of them below 1, a frozen snapshot at 0).
#: ``separate``: another model with the same tokenizer scores the student's
#: sequence; its checkpoint is named by ``teacher_checkpoint``.
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
    the teacher's log-probs at K ids per position and at the sampled token.
    ``top_k_source`` says who picks the K ids, the teacher (its own top-K) or
    the current student (its top-K, from one more forward before the step);
    ``top_k_distribution`` says how the two distributions are compared on
    them, ``renormalized`` over the K ids (the reverse KL then estimated at
    the sampled token) or with a ``tail`` bucket for the rest of the
    vocabulary. ``importance_sampling_level`` averages the truncated
    importance weight over the response (``sequence``) or applies it at each
    token (``token``).
    """

    teacher: str = "self"
    divergence: str = "forward"
    top_k: int = 0
    top_k_source: str = "teacher"
    top_k_distribution: str = "renormalized"
    teacher_update_rate: float = 0.01
    teacher_checkpoint: str = ""
    teacher_url: str = ""
    teacher_model_path: str = ""
    teacher_timeout: float = 300.0
    importance_sampling_cap: float = 2.0
    importance_sampling_level: str = "sequence"
    skip_response_tokens: int = 0
    jsd_beta: float = 0.5

    def __post_init__(self) -> None:
        if self.teacher not in TEACHER_SOURCES:
            raise ValueError(f"distill teacher must be one of: {', '.join(TEACHER_SOURCES)}")
        if self.divergence not in DIVERGENCES:
            raise ValueError(f"distill divergence must be one of: {', '.join(DIVERGENCES)}")
        if not _is_integer(self.top_k) or self.top_k < 0:
            raise ValueError("distill top_k must be a non-negative integer (0 keeps the whole distribution)")
        if self.top_k_source not in ("teacher", "student"):
            raise ValueError("distill top_k_source must be teacher or student")
        if self.top_k_distribution not in ("renormalized", "tail"):
            raise ValueError("distill top_k_distribution must be renormalized or tail")
        if self.importance_sampling_level not in ("sequence", "token"):
            raise ValueError("distill importance_sampling_level must be sequence or token")
        if not _is_finite(self.teacher_update_rate) or not 0 <= self.teacher_update_rate <= 1:
            raise ValueError("distill teacher_update_rate must be a number in [0, 1]")
        if not isinstance(self.teacher_checkpoint, str):
            raise ValueError("distill teacher_checkpoint must be a path string")
        if not isinstance(self.teacher_url, str) or not isinstance(self.teacher_model_path, str):
            raise ValueError("teacher_url and teacher_model_path must be strings")
        if not _is_finite(self.teacher_timeout) or self.teacher_timeout <= 0:
            raise ValueError("teacher_timeout must be finite and positive")
        if self.teacher_url:
            address = urlsplit(self.teacher_url)
            if address.scheme not in ("http", "https") or not address.hostname or address.query or address.fragment:
                raise ValueError("teacher_url must be an HTTP(S) base URL")
            if address.username or address.password:
                raise ValueError("teacher_url must not contain credentials")
            if self.teacher != "separate" or self.teacher_checkpoint:
                raise ValueError("teacher_url requires teacher separate and replaces teacher_checkpoint")
            if not self.teacher_model_path.strip():
                raise ValueError("teacher_url requires teacher_model_path for tokenizer validation")
            if self.top_k == 0 or self.top_k_source != "teacher":
                raise ValueError(
                    "engine teacher requires positive top_k and top_k_source teacher; use checkpoint mode"
                )
        elif self.teacher_model_path:
            raise ValueError("teacher_model_path requires teacher_url")
        if self.teacher == "separate" and not self.teacher_checkpoint.strip() and not self.teacher_url:
            raise ValueError("distill teacher 'separate' needs teacher_checkpoint, the teacher's checkpoint directory")
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

    @property
    def score_centering_weight(self) -> PolicyGradientWeight:
        """The sampled reverse-KL weight, refusing losses with a different gradient."""
        if self.exact or self.divergence != "reverse" or self.top_k_distribution != "renormalized":
            raise RuntimeError(
                "score centering for distillation requires sampled reverse KL: set the family's "
                "divergence=reverse, top-k>0 and top-k-distribution=renormalized"
            )
        if self.importance_sampling_cap == 0:
            return PolicyGradientWeight("none")
        if self.importance_sampling_level != "token":
            raise RuntimeError(
                "score centering for distillation requires importance-sampling-level=token "
                "or importance-sampling-cap=0; sequence weights depend on other sampled tokens"
            )
        return PolicyGradientWeight("truncated", upper=self.importance_sampling_cap)


def _is_integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _is_finite(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


ROW_SHAPE = "[source_id, tokens, loss_mask, rollout_log_probs, reward, teacher_tokens, sample_weight]"


def distill_sample_row(sample: TrajectoryItem) -> list[Any]:
    """Shape one Reef sample into the family's 7-element wire row.

    A sample without a ``distill_sample_weight`` in its training data
    carries weight 1: the plain per-sample mean of the divergence.
    """
    return [
        source_record_id(sample),
        list(sample.training.get("tokens", [])),
        list(sample.training.get("loss_mask", [])),
        list(sample.training.get("rollout_log_probs", [])),
        trajectory_reward(sample),
        list(sample.training.get("teacher_tokens", [])),
        float(sample.training.get("distill_sample_weight", 1.0)),
    ]


def build_distill_rollout_data(payload: Mapping[str, Any], samples: Sequence, spec: SlimeAlgorithm) -> dict:
    """Validate and convert the family's rows into Slime's external rollout payload."""
    name = spec.loss_family
    base_rows: list[list[Any]] = []
    teacher_rows: list[Any] = []
    sample_weights: list[float] = []
    for index, row in enumerate(samples):
        if not isinstance(row, Sequence) or isinstance(row, str | bytes) or len(row) != 7:
            raise ValueError(f"{name} sample {index} must be {ROW_SHAPE}")
        if not _is_finite(row[6]) or row[6] < 0:
            raise ValueError(f"{name} sample {index} sample_weight must be a finite number >= 0")
        base_rows.append(list(row[:5]))
        teacher_rows.append(row[5])
        sample_weights.append(float(row[6]))

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
    data["distill_sample_weights"] = sample_weights
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
    rollout_data_keys = ("teacher_tokens", "distill_sample_weights", *TEACHER_BATCH_KEYS[1:])
    rollout_tensor_dtypes: Mapping[str, str] = {
        "teacher_tokens": "long",
        "distill_teacher_topk_ids": "long",
        "distill_teacher_topk_log_probs": "float32",
        "distill_teacher_sampled_log_probs": "float32",
    }
    external_batch_keys = ("rollout_log_probs", "distill_sample_weights", *TEACHER_BATCH_KEYS)
    rollout_log_skip_keys = ("teacher_tokens", "distill_sample_weights", *TEACHER_BATCH_KEYS)
    required_objective_hooks = ("custom_loss_function_path", "reef_actor_pre_train_hook_path")
    #: The recipe's settings type, with its defaults.
    settings_type: type[DistillSettings] = DistillSettings
    _teacher_settings: DistillSettings | None = None

    # --- stage 1: configure ---

    def policy_gradient_weight(self, args: Namespace) -> PolicyGradientWeight:
        return settings_from_args(args).score_centering_weight

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
        # The student's top-K ids come from a forward-only pass before the
        # step; the training forward must put the same distribution at them,
        # so dropout would let the loss compare the teacher against ids the
        # trained student never ranked highest.
        settings = settings_from_args(args)
        if settings.teacher_url:
            if getattr(args, "rollout_temperature", 1.0) != 1.0:
                raise ValueError("engine teacher requires rollout_temperature=1: SGLang input logprobs are untempered")
            from reef.train.slime_backend.distill.engine import validate_teacher_tokenizer

            validate_teacher_tokenizer(args.hf_checkpoint, settings.teacher_model_path)
        student_selects = not settings.exact and settings.top_k_source == "student"
        if student_selects and (args.attention_dropout != 0 or args.hidden_dropout != 0):
            raise RuntimeError(
                f"{source} selects the top-K ids with the student and needs --attention-dropout 0 and "
                "--hidden-dropout 0: the ids come from a forward before the step"
            )

    def parse_specific_options(self, arguments: Sequence[str]) -> tuple[DistillSettings, list[str]]:
        prefix = f"--{self.loss_family}-"
        # Field defaults for the help text: a family that requires an option, such as a separate
        # teacher's checkpoint, has no valid default instance.
        defaults = {field.name: field.default for field in fields(self.settings_type)}
        parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, argument_default=argparse.SUPPRESS)
        parser.add_argument(
            f"{prefix}teacher",
            dest="teacher",
            choices=list(TEACHER_SOURCES),
            help=(
                "Who scores the student's samples: 'self' is the student's own weights reading the privileged "
                "prefix, 'separate' another model with the same tokenizer (see teacher-checkpoint). "
                f"Default {defaults['teacher']}."
            ),
        )
        parser.add_argument(
            f"{prefix}divergence",
            dest="divergence",
            choices=list(DIVERGENCES),
            help=(
                "The per-token divergence: 'forward' is KL(teacher || student), 'reverse' is KL(student || "
                f"teacher), 'jsd' the generalized Jensen-Shannon divergence. Default {defaults['divergence']}."
            ),
        )
        parser.add_argument(
            f"{prefix}top-k",
            dest="top_k",
            type=int,
            help=(
                "Keep the teacher's log-probs at K ids per position (see top-k-source) and at the sampled token "
                f"instead of its whole distribution; 0 keeps the whole distribution. Default {defaults['top_k']}."
            ),
        )
        parser.add_argument(
            f"{prefix}teacher-update-rate",
            dest="teacher_update_rate",
            type=float,
            help=(
                "For a 'self' teacher: the fraction of the current policy mixed into the teacher's weights after "
                "every step (the SDFT reference's ref_model_mixup_alpha). 1 makes the current policy the teacher, "
                f"0 freezes the initial weights. Default {defaults['teacher_update_rate']}."
            ),
        )
        parser.add_argument(
            f"{prefix}teacher-checkpoint",
            dest="teacher_checkpoint",
            help="For a 'separate' teacher: its checkpoint, loaded into the actor's layout beside the student's weights.",
        )
        parser.add_argument(
            f"{prefix}importance-sampling-cap",
            dest="importance_sampling_cap",
            type=float,
            help=(
                "Cap of the truncated importance-sampling weight between the policy and the rollout engine's "
                f"log-probs, averaged over the response. 0 disables it. Default {defaults['importance_sampling_cap']}."
            ),
        )
        parser.add_argument(
            f"{prefix}skip-response-tokens",
            dest="skip_response_tokens",
            type=int,
            help=(
                "Response tokens at the start of every sample left out of the loss and its denominator. "
                f"Default {defaults['skip_response_tokens']}."
            ),
        )
        parser.add_argument(
            f"{prefix}jsd-beta",
            dest="jsd_beta",
            type=float,
            help=f"For 'jsd': the teacher's weight in the mixture. Default {defaults['jsd_beta']}.",
        )
        parser.add_argument(
            f"{prefix}top-k-source",
            dest="top_k_source",
            choices=("teacher", "student"),
            help="Select top-K ids using the teacher or the current student before the optimizer step.",
        )
        parser.add_argument(
            f"{prefix}top-k-distribution",
            dest="top_k_distribution",
            choices=("renormalized", "tail"),
            help="Renormalize the selected K entries, or retain the remaining vocabulary mass as a tail bucket.",
        )
        parser.add_argument(
            f"{prefix}importance-sampling-level",
            dest="importance_sampling_level",
            choices=("sequence", "token"),
            help="Average correction weights per sequence, or apply them independently at each token.",
        )
        parser.add_argument(
            f"{prefix}teacher-url", dest="teacher_url", help="Independent frozen SGLang teacher base URL."
        )
        parser.add_argument(
            f"{prefix}teacher-model-path",
            dest="teacher_model_path",
            help="Teacher HF model directory for tokenizer validation.",
        )
        parser.add_argument(
            f"{prefix}teacher-timeout",
            dest="teacher_timeout",
            type=float,
            help="Teacher scoring HTTP timeout in seconds.",
        )
        options, remaining = parser.parse_known_args(list(arguments))
        return self.settings_type(**vars(options)), remaining

    def apply_driver_options(self, args: Namespace, options: object | None) -> None:
        super().apply_driver_options(args, options)
        settings = options if isinstance(options, DistillSettings) else self.settings_type()
        args.distill_teacher = settings.teacher
        args.distill_divergence = settings.divergence
        args.distill_top_k = settings.top_k
        args.distill_top_k_source = settings.top_k_source
        args.distill_top_k_distribution = settings.top_k_distribution
        args.distill_teacher_update_rate = settings.teacher_update_rate
        args.distill_teacher_checkpoint = settings.teacher_checkpoint
        args.distill_teacher_url = settings.teacher_url
        args.distill_teacher_model_path = settings.teacher_model_path
        args.distill_teacher_timeout = settings.teacher_timeout
        args.distill_importance_sampling_cap = settings.importance_sampling_cap
        args.distill_importance_sampling_level = settings.importance_sampling_level
        args.distill_skip_response_tokens = settings.skip_response_tokens
        args.distill_jsd_beta = settings.jsd_beta

    def bind(
        self,
        config: object | None = None,
        *,
        critic_steps_per_actor: int | None = None,
        critic_only_steps: int = 0,
    ) -> DistillAlgorithm:
        # Checkpoint teachers travel on args; engine scoring is owned by this bridge.
        if config is not None and not isinstance(config, DistillSettings):
            raise TypeError(f"{self.loss_family} bridge algorithm config must be {self.settings_type.__name__}")
        if isinstance(config, DistillSettings) and config.teacher_url:
            bound = self.__class__()
            bound._teacher_settings = config
            return bound
        return self

    def prepare_rollout(self, rollout_data: dict[str, Any]) -> dict[str, Any]:
        """Score an independent teacher before tensorization and the training job marker."""
        if self._teacher_settings is None:
            return {}
        from reef.train.slime_backend.distill.engine import EngineTeacher

        return EngineTeacher(self._teacher_settings).prepare(rollout_data)

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
        top_k_source=args.distill_top_k_source,
        top_k_distribution=args.distill_top_k_distribution,
        teacher_update_rate=args.distill_teacher_update_rate,
        teacher_checkpoint=args.distill_teacher_checkpoint,
        teacher_url=getattr(args, "distill_teacher_url", ""),
        teacher_model_path=getattr(args, "distill_teacher_model_path", ""),
        teacher_timeout=getattr(args, "distill_teacher_timeout", 300.0),
        importance_sampling_cap=args.distill_importance_sampling_cap,
        importance_sampling_level=args.distill_importance_sampling_level,
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
