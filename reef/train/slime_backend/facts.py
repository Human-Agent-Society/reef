"""What the Slime backend states from its options alone, without Slime's parser or a component.

``SlimeFacts`` reads four of Slime's options: the model, and the LoRA rank, alpha and target modules. The
component answers the same members from the value of its parsed arguments (``SlimeFacts.from_parsed``), and it
refuses to start when the options as written give another value (``SlimeFacts.from_arguments``). The module imports
no Slime, Megatron, Ray or torch module, so control and inference read it in an environment without them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from reef.artifact.base_root import BASE_FILE, BaseRoot
from reef.artifact.repository import DefaultStart
from reef.core.weight_layout import ArtifactFormat
from reef.storage.job_data import JobDataKind
from reef.train.component import ScenarioRoot, TrainingBackendFacts
from reef.train.slime_backend.parameters import slime_parameters_schema
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

    Purpose: ``SlimeComponentFactory.facts`` makes one from the deployment's options (``from_arguments``);
    ``SlimeJobs`` makes one from its parsed arguments (``from_parsed``) and answers from it.
    """

    hf_checkpoint: str
    lora_rank: int = 0
    lora_alpha: int | None = None
    lora_target_modules: tuple[str, ...] | None = None

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

        Each option is read by its full name (``option_values``). ``ValueError``: ``--hf-checkpoint`` is absent, or
        a LoRA option is no value of ``SlimeFacts``.
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
        return cls(checkpoint[0], rank, alpha, None if targets is None else tuple(targets))

    @classmethod
    def from_parsed(cls, args: SlimeArguments, *, lora: bool) -> Self:
        """The facts of Slime's parsed arguments ``args``; ``lora`` is whether the actor trains a LoRA adapter."""
        if not lora:
            return cls(args.hf_checkpoint)
        targets = args.megatron_lora_target_modules
        return cls(
            args.hf_checkpoint,
            int(args.megatron_lora_rank),
            args.megatron_lora_alpha,
            None if targets is None else tuple(targets),
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
    "hf_lora_target_modules",
    "option_values",
]
