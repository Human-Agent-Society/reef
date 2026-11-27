"""What the Slime backend states from its options alone, without Slime's parser or a component.

``SlimeFacts`` reads the model and the LoRA rank, alpha and target modules from Slime's options. The component
answers the same members from the value of its parsed arguments (``SlimeFacts.from_parsed``), and it refuses to start
when the options as written give another value (``SlimeFacts.from_arguments``). The facts also read the rollout
options that bound the delayed samples of a task job (``SlimeRolloutOptions``); the training service checks those
against the parsed arguments when it checks the recipe's task jobs. The module imports no Slime, Megatron, Ray or
torch module, so control and inference read it in an environment without them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from reef.artifact.base_root import BASE_FILE, BaseRoot
from reef.artifact.repository import DefaultStart
from reef.core.job_data import DelayedSampleLimit
from reef.core.training_capture import sampling_distribution
from reef.core.weight_layout import ArtifactFormat
from reef.storage.data.job_data import JobDataKind
from reef.train.component import ScenarioRoot, TrainingBackendFacts
from reef.train.slime_backend.objectives import resolve_objective
from reef.train.slime_backend.parameters import SlimeJobParameters, slime_parameters_schema
from reef.train.slime_backend.reef_adapters.arguments import SlimeArguments

#: The name of the Slime training component, which its jobs carry.
SLIME_COMPONENT = "slime"
#: The Hugging Face module names that each Megatron Bridge LoRA target trains.
MEGATRON_TO_HF_TARGETS = {
    "linear_qkv": ("q_proj", "k_proj", "v_proj"),
    "linear_proj": ("o_proj",),
    "linear_fc1": ("gate_proj", "up_proj"),
    "linear_fc2": ("down_proj",),
}
#: The Megatron targets of a LoRA whose options name none.
DEFAULT_MEGATRON_TARGETS = tuple(MEGATRON_TO_HF_TARGETS)


def hf_lora_target_modules(megatron_targets: Sequence[str] | None) -> list[str]:
    """The Hugging Face leaf module names of the Megatron Bridge targets ``megatron_targets``, in their order,
    each once; None names the default targets. A target that the table does not hold keeps its leaf name."""
    converted: list[str] = []
    for target in DEFAULT_MEGATRON_TARGETS if megatron_targets is None else megatron_targets:
        leaf = target.rsplit(".", 1)[-1]
        for hf_target in MEGATRON_TO_HF_TARGETS.get(leaf, (leaf,)):
            if hf_target not in converted:
                converted.append(hf_target)
    return converted


def option_values(arguments: Sequence[str], flag: str) -> list[str] | None:
    """The values of the last ``flag`` in the argument list ``arguments``, or None when it is absent.

    ``--flag=value`` gives ``[value]``. ``--flag`` alone gives the tokens after it up to the next token that starts
    with ``--``, as ``reef.execution.executor.arguments.native_arguments`` writes a list option.
    """
    values: list[str] | None = None
    for index, token in enumerate(arguments):
        if token.startswith(f"{flag}="):
            values = [token.split("=", 1)[1]]
        elif token == flag:
            values = []
            for following in arguments[index + 1 :]:
                if following.startswith("--"):
                    break
                values.append(following)
    return values


def option_number(arguments: Sequence[str], flag: str, kind: type[int] | type[float]) -> int | float | None:
    """The value of the last ``flag`` in ``arguments`` as ``kind``, or None when it is absent (``option_values``).

    ``ValueError``: the option has no value, or one that is not a ``kind``."""
    values = option_values(arguments, flag)
    if values is None:
        return None
    if len(values) != 1:
        raise ValueError(f"{flag} takes one value, got {values!r}")
    try:
        return kind(values[0])
    except ValueError as exc:
        raise ValueError(f"{flag} is a number: {exc}") from exc


@dataclass(frozen=True)
class SlimeRolloutOptions:
    """The rollout options of Slime that bound the delayed samples of a task job.

    Values, each the option that the engines and Slime's parser accept:

    - ``temperature`` is ``--rollout-temperature``, a finite number that is at least 0 (Slime's default 1.0).
    - ``top_p`` is ``--rollout-top-p``, a number in (0, 1] (default 1.0).
    - ``top_k`` is ``--rollout-top-k``: -1, for no limit, or a positive int (default -1).
    - ``batch_size`` is ``--rollout-batch-size``, the prompts of one rollout: a positive int, or None when the options
      give none.
    - ``samples_per_prompt`` is ``--n-samples-per-prompt``, a positive int (default 1).
    - ``global_batch_size`` is the samples of one optimizer step: ``--global-batch-size``, or the value that Slime
      computes from ``--num-steps-per-rollout``. It is a positive int, or None when the options give none.

    The constructor refuses other values, bools included, with a ``ValueError`` that names the field and the option.

    Purpose: ``SlimeFacts`` holds the value of its options, and ``delayed_sample_limit`` computes the limit from it,
    so control (``SlimeFacts.delayed_sample_limit``) and the training service
    (``harbor_rollouts.delayed_sample_limit``) give the same limit.
    """

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    batch_size: int | None = None
    samples_per_prompt: int = 1
    global_batch_size: int | None = None

    def __post_init__(self) -> None:
        temperature, top_p, top_k = self.temperature, self.top_p, self.top_k
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not (math.isfinite(temperature) and temperature >= 0)
        ):
            raise ValueError(
                "SlimeRolloutOptions.temperature (--rollout-temperature) must be a finite number of at least 0, "
                f"got {temperature!r}"
            )
        if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not 0 < top_p <= 1:
            raise ValueError(f"SlimeRolloutOptions.top_p (--rollout-top-p) must be in (0, 1], got {top_p!r}")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not (top_k == -1 or top_k > 0):
            raise ValueError(
                f"SlimeRolloutOptions.top_k (--rollout-top-k) must be -1 or a positive int, got {top_k!r}"
            )
        for name, option, value, may_be_none in (
            ("batch_size", "--rollout-batch-size", self.batch_size, True),
            ("samples_per_prompt", "--n-samples-per-prompt", self.samples_per_prompt, False),
            (
                "global_batch_size",
                "--global-batch-size, or the value of --num-steps-per-rollout",
                self.global_batch_size,
                True,
            ),
        ):
            if value is None and may_be_none:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                domain = "a positive int or None" if may_be_none else "a positive int"
                raise ValueError(f"SlimeRolloutOptions.{name} ({option}) must be {domain}, got {value!r}")

    @classmethod
    def from_arguments(cls, arguments: Sequence[str]) -> Self:
        """The rollout options of the Slime argument list ``arguments``, read as written, with Slime's defaults.

        With ``--num-steps-per-rollout`` and ``--rollout-batch-size``, ``global_batch_size`` is
        ``--rollout-batch-size * --n-samples-per-prompt // --num-steps-per-rollout``, as Slime computes it.
        ``ValueError``: an option is no number of its kind, ``--num-steps-per-rollout`` is below 1, or a value is
        outside the domain of its field.
        """
        temperature = option_number(arguments, "--rollout-temperature", float)
        top_p = option_number(arguments, "--rollout-top-p", float)
        top_k = option_number(arguments, "--rollout-top-k", int)
        batch_size = option_number(arguments, "--rollout-batch-size", int)
        samples_per_prompt = option_number(arguments, "--n-samples-per-prompt", int)
        global_batch_size = option_number(arguments, "--global-batch-size", int)
        steps = option_number(arguments, "--num-steps-per-rollout", int)
        samples = 1 if samples_per_prompt is None else int(samples_per_prompt)
        if steps is not None and steps < 1:
            raise ValueError(f"--num-steps-per-rollout must be a positive int, got {steps}")
        if steps is not None and batch_size is not None:
            # Slime's own rule: the steps of one rollout split its samples.
            global_batch_size = int(batch_size) * samples // int(steps)
        return cls(
            1.0 if temperature is None else float(temperature),
            1.0 if top_p is None else float(top_p),
            -1 if top_k is None else int(top_k),
            None if batch_size is None else int(batch_size),
            samples,
            None if global_batch_size is None else int(global_batch_size),
        )

    @classmethod
    def from_parsed(cls, args: SlimeArguments) -> Self:
        """The rollout options of Slime's parsed arguments ``args``. ``ValueError``: a value is outside the domain of
        its field."""
        return cls(
            float(args.rollout_temperature),
            float(args.rollout_top_p),
            int(args.rollout_top_k),
            args.rollout_batch_size,
            int(args.n_samples_per_prompt),
            args.global_batch_size,
        )

    def delayed_sample_limit(self) -> DelayedSampleLimit:
        """The delayed samples that a task job that carries samples trains with these options.

        The job trains at most ``min(--global-batch-size, (--rollout-batch-size - 1) * --n-samples-per-prompt)``
        delayed samples, which keeps the delayed groups and a task in the first rollout and the samples in its first
        optimizer step. Each sample comes from Slime's rollout distribution (``--rollout-temperature``,
        ``--rollout-top-p``, ``--rollout-top-k``), so that its recorded log-probs are those of the sampler that the
        workers' log-probs stand for. ``ValueError`` for a greedy rollout distribution, for options that give no
        rollout or optimizer step size, or for a rollout of fewer than two prompts.
        """
        distribution = sampling_distribution(
            {"temperature": self.temperature, "top_p": self.top_p, "top_k": self.top_k}
        )
        if distribution is None:
            raise ValueError(
                "a task job that carries samples needs a sampled rollout distribution: a positive "
                "--rollout-temperature and a --rollout-top-p in (0, 1]"
            )
        if self.batch_size is None or self.global_batch_size is None:
            raise ValueError(
                "a task job that carries samples needs --rollout-batch-size, and --global-batch-size or "
                "--num-steps-per-rollout, in Slime's options"
            )
        if self.batch_size < 2:
            raise ValueError(
                "a task job that carries samples needs --rollout-batch-size of at least 2: its first rollout leads "
                "with a delayed group, even an empty one, and needs a task beside it"
            )
        return DelayedSampleLimit(
            min(self.global_batch_size, (self.batch_size - 1) * self.samples_per_prompt), distribution
        )


@dataclass(frozen=True)
class SlimeFacts(TrainingBackendFacts):
    """The facts of one Slime deployment: the model that it starts from and the LoRA that it trains, if any.

    Values: ``hf_checkpoint`` is the non-empty ``--hf-checkpoint``. ``lora_rank`` is ``--megatron-lora-rank``, an
    int that is at least 0; 0 trains all the weights. With a rank of 0, ``lora_alpha`` and ``lora_target_modules``
    are None. With a positive rank, ``lora_alpha`` is a positive int (``--megatron-lora-alpha``, the rank when the
    option is absent) and ``lora_target_modules`` are the Megatron Bridge targets of
    ``--megatron-lora-target-modules``, or None for the default targets.

    The members are those of ``TrainingBackendFacts``. The snapshots hold full weights, or with LoRA the PEFT
    adapter of the rank, whose target modules carry the engines' names (``hf_lora_target_modules``). Slime's workers
    load ``hf_checkpoint`` when they start and cannot load a snapshot, so ``check_root_start`` accepts only a root
    that came from the default start of the same ``default_start_inputs``, and ``check_import_start`` refuses every
    snapshot. ``default_start`` is a base root of those
    inputs (``reef.artifact.base_root``): the workers make the startup weights themselves, the model and with LoRA
    the adapter that Megatron initializes, so the root holds no weight file. A component holds one job's weights at
    a time, so neither scenarios nor the jobs of one scenario share it.

    ``rollout`` holds the rollout options that bound a task job's delayed samples (``delayed_sample_limit``). It takes
    no part in the equality of two values: Slime derives some of those options, so the training service compares the
    limits themselves when it checks the recipe's task jobs (``SlimeTrainingComponent.check_task_jobs``).

    Purpose: ``SlimeComponentFactory.facts`` makes one from the deployment's options (``from_arguments``);
    ``SlimeJobs`` makes one from its parsed arguments (``from_parsed``) and answers from it.
    """

    hf_checkpoint: str
    lora_rank: int = 0
    lora_alpha: int | None = None
    lora_target_modules: tuple[str, ...] | None = None
    rollout: SlimeRolloutOptions = field(default_factory=SlimeRolloutOptions, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.hf_checkpoint, str) or not self.hf_checkpoint:
            raise ValueError("SlimeFacts.hf_checkpoint must be a non-empty string")
        if isinstance(self.lora_rank, bool) or not isinstance(self.lora_rank, int) or self.lora_rank < 0:
            raise ValueError("SlimeFacts.lora_rank must be an int that is at least 0")
        if self.lora_rank == 0:
            if self.lora_alpha is not None or self.lora_target_modules is not None:
                raise ValueError("SlimeFacts.lora_alpha and lora_target_modules need a positive lora_rank")
            return
        alpha = self.lora_alpha
        if isinstance(alpha, bool) or not isinstance(alpha, int) or alpha <= 0:
            raise ValueError("SlimeFacts.lora_alpha must be a positive int when lora_rank is positive")
        targets = self.lora_target_modules
        if targets is not None and (
            not isinstance(targets, tuple) or not all(isinstance(target, str) and target for target in targets)
        ):
            raise ValueError("SlimeFacts.lora_target_modules must be a tuple of non-empty strings or None")

    @classmethod
    def from_arguments(cls, arguments: Sequence[str]) -> Self:
        """The facts of the Slime argument list ``arguments``, read as written, without Slime's parser.

        Each option is read by its full name (``option_values``). ``ValueError``: ``--hf-checkpoint`` is absent, a
        LoRA option is no value of ``SlimeFacts``, a rollout option is no number of its kind or is outside the domain
        of its ``SlimeRolloutOptions`` field, or ``--num-steps-per-rollout`` is below 1
        (``SlimeRolloutOptions.from_arguments``).
        """
        checkpoint = option_values(arguments, "--hf-checkpoint")
        if not checkpoint:
            raise ValueError("Slime's options name no --hf-checkpoint, the model that the workers start from")
        rank_values = option_values(arguments, "--megatron-lora-rank")
        alpha_values = option_values(arguments, "--megatron-lora-alpha")
        targets = option_values(arguments, "--megatron-lora-target-modules")
        try:
            rank = int(rank_values[0]) if rank_values else 0
            alpha = int(alpha_values[0]) if alpha_values else None
        except ValueError as exc:
            raise ValueError(f"--megatron-lora-rank and --megatron-lora-alpha are integers: {exc}") from exc
        if rank > 0 and alpha is None:
            alpha = rank
        rollout = SlimeRolloutOptions.from_arguments(arguments)
        return cls(checkpoint[0], rank, alpha, None if targets is None else tuple(targets), rollout)

    @classmethod
    def from_parsed(cls, args: SlimeArguments, *, lora: bool) -> Self:
        """The facts of Slime's parsed arguments ``args``; ``lora`` is whether the actor trains a LoRA adapter.

        ``ValueError``: a LoRA value is no value of ``SlimeFacts``, or a rollout option is outside the domain of its
        ``SlimeRolloutOptions`` field (``SlimeRolloutOptions.from_parsed``).
        """
        rollout = SlimeRolloutOptions.from_parsed(args)
        if not lora:
            return cls(args.hf_checkpoint, rollout=rollout)
        targets = args.megatron_lora_target_modules
        return cls(
            args.hf_checkpoint,
            int(args.megatron_lora_rank),
            args.megatron_lora_alpha,
            None if targets is None else tuple(targets),
            rollout,
        )

    @property
    def name(self) -> str:
        return SLIME_COMPONENT

    @property
    def artifact_format(self) -> ArtifactFormat:
        """Full weights (the rollout's Hugging Face export), or with LoRA the PEFT adapter of the workers' rank."""
        if self.lora_rank == 0:
            return ArtifactFormat("full")
        return ArtifactFormat("adapter", self.lora_rank, tuple(hf_lora_target_modules(self.lora_target_modules)))

    @property
    def shares_scenarios(self) -> bool:
        return False

    @property
    def concurrent_jobs(self) -> bool:
        return False

    @property
    def default_start_inputs(self) -> Mapping[str, str]:
        """The configuration that decides the startup weights: the model, and with LoRA the adapter's settings."""
        model = str(Path(self.hf_checkpoint).expanduser().resolve())
        if self.lora_rank == 0:
            return {"model": model}
        return {
            "model": model,
            "lora_rank": str(self.lora_rank),
            "lora_alpha": str(self.lora_alpha),
            "lora_target_modules": ",".join(sorted(hf_lora_target_modules(self.lora_target_modules))),
        }

    def default_start(self) -> DefaultStart:
        """A base root of ``default_start_inputs``: the startup weights are the model's, with no adapter."""
        inputs = self.default_start_inputs
        return DefaultStart(files={BASE_FILE: BaseRoot(inputs).to_text()}, inputs=inputs)

    def parameters_schema(self, kind: JobDataKind) -> dict[str, Any]:
        return slime_parameters_schema(kind)

    def delayed_sample_limit(self, parameters: Mapping[str, Any]) -> DelayedSampleLimit | None:
        """The limit of the rollout options (``SlimeRolloutOptions.delayed_sample_limit``) for a task job whose
        objective carries samples; None for a job of another objective."""
        objective = resolve_objective(SlimeJobParameters.from_parameters(parameters).objective)
        if not objective.carries_samples:
            return None
        return self.rollout.delayed_sample_limit()

    @property
    def start_files(self) -> tuple[str, ...]:
        """No file: a root is checked by its source, and a published snapshot is refused unread."""
        return ()

    def check_root_start(self, root: ScenarioRoot) -> None:
        configured = dict(self.default_start_inputs)
        if root.source.kind != "default" or dict(root.source.inputs) != configured:
            started = (
                f"the item {root.source.item!r}"
                if root.source.kind == "item"
                else f"the inputs {dict(root.source.inputs)!r}"
            )
            raise ValueError(
                f"Slime trains the weights that it starts with, from the inputs {configured!r}, but the scenario's root "
                f"snapshot came from {started}; create the scenario again with this configuration"
            )

    def check_import_start(self, artifact_dir: Path) -> None:
        raise ValueError(
            "Slime continues training from its own latest checkpoint; it cannot load a published snapshot "
            "into its workers"
        )


__all__ = [
    "DEFAULT_MEGATRON_TARGETS",
    "MEGATRON_TO_HF_TARGETS",
    "SLIME_COMPONENT",
    "SlimeFacts",
    "SlimeRolloutOptions",
    "hf_lora_target_modules",
    "option_number",
    "option_values",
]
