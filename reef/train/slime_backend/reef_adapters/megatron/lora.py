"""Megatron Bridge LoRA integration shared by model setup and smoke metrics."""

from __future__ import annotations

import logging
from argparse import Namespace
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import torch

logger = logging.getLogger(__name__)


_MEGATRON_TO_HF_TARGETS = {
    "linear_qkv": ("q_proj", "k_proj", "v_proj"),
    "linear_proj": ("o_proj",),
    "linear_fc1": ("gate_proj", "up_proj"),
    "linear_fc2": ("down_proj",),
}
_DEFAULT_MEGATRON_TARGETS = tuple(_MEGATRON_TO_HF_TARGETS)


def validate_megatron_lora_args(args: Namespace) -> None:
    """Normalize and validate Slime's optional Megatron LoRA configuration."""

    rank = args.megatron_lora_rank
    if rank < 0:
        raise ValueError("--megatron-lora-rank must be non-negative")
    if not 0 <= args.megatron_lora_dropout < 1:
        raise ValueError("--megatron-lora-dropout must be in [0, 1)")

    if rank == 0:
        if args.megatron_lora_alpha is not None:
            raise ValueError("--megatron-lora-alpha requires a positive --megatron-lora-rank")
        if args.megatron_lora_target_modules is not None:
            raise ValueError("--megatron-lora-target-modules requires a positive --megatron-lora-rank")
        return

    if args.megatron_to_hf_mode != "bridge":
        raise ValueError("--megatron-lora-rank requires --megatron-to-hf-mode=bridge")
    if args.megatron_lora_alpha is None:
        args.megatron_lora_alpha = rank
    elif args.megatron_lora_alpha <= 0:
        raise ValueError("--megatron-lora-alpha must be positive")
    if args.only_train_params_name_list or args.freeze_params_name_list:
        raise ValueError(
            "Megatron Bridge LoRA owns actor parameter freezing; do not combine it with "
            "--only-train-params-name-list or --freeze-params-name-list"
        )
    if getattr(args, "colocate", False) and getattr(args, "rollout_num_gpus_per_engine", 1) != 1:
        raise ValueError(
            "colocated Megatron LoRA sync currently requires --rollout-num-gpus-per-engine=1 "
            "so every adapter checksum is verified against its matching SGLang engine"
        )
    slots = getattr(args, "max_loaded_loras", None)
    if slots is not None and int(slots) < 1:
        raise ValueError("--max-loaded-loras must be at least 1")


def lora_engine_slots(args: Namespace) -> int:
    """Adapter slots the SGLang engine holds; the bridge's residency capacity.

    Every scenario keeps its current revision resident, and publishing a new
    revision needs one more slot while both exist, so size it to the number
    of scenarios plus one. With fewer slots the residency manager evicts the
    publishing scenario's own current revision first (generation is paused,
    so no request observes the gap); with exactly one slot that is the only
    option, which is fine for one scenario but leaves a second scenario's
    adapter unloaded whenever the other publishes.
    """

    value = getattr(args, "max_loaded_loras", None)
    return 1 if value is None else max(1, int(value))


def scenario_adapter_name(scenario: str, weight_version: str) -> str:
    """The engine adapter name for one scenario's publication at ``weight_version``.

    The same lossless ``adapter_name(scenario, version)`` the serving side
    uses for durable adapters, with the serving weight version as the
    version component: the training side knows it at publication time, and
    the live artifact Reef commits carries the same token. Every scenario's
    every publication therefore has its own engine name; nothing is ever
    overwritten in place.
    """

    from reef.surface.adapter import adapter_name

    return adapter_name(scenario, weight_version)


def megatron_lora_enabled(args: Namespace) -> bool:
    """Return whether the Reef/Slime Megatron actor uses LoRA."""

    return int(getattr(args, "megatron_lora_rank", 0) or 0) > 0


def megatron_lora_target_modules(args: Namespace) -> list[str]:
    """Return the effective Megatron target list used by Bridge LoRA."""

    configured = getattr(args, "megatron_lora_target_modules", None)
    return list(configured) if configured is not None else list(_DEFAULT_MEGATRON_TARGETS)


def sglang_lora_target_modules(args: Namespace) -> list[str]:
    """Convert Megatron Bridge target names to SGLang/HF leaf names."""

    converted: list[str] = []
    for target in megatron_lora_target_modules(args):
        leaf = target.rsplit(".", 1)[-1]
        hf_targets = _MEGATRON_TO_HF_TARGETS.get(leaf, (leaf,))
        for hf_target in hf_targets:
            if hf_target not in converted:
                converted.append(hf_target)
    return converted


def build_sglang_lora_config(args: Namespace) -> dict[str, Any]:
    """Build the PEFT-compatible config sent with an online adapter update."""

    return {
        "peft_type": "LORA",
        "r": int(args.megatron_lora_rank),
        "lora_alpha": int(args.megatron_lora_alpha),
        "target_modules": sglang_lora_target_modules(args),
        "lora_dropout": float(args.megatron_lora_dropout),
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


def is_lora_weight_name(name: str) -> bool:
    """Return whether an exported HF tensor belongs to the LoRA adapter."""

    return ".lora_A." in name or ".lora_B." in name


def lora_engine_identity(engine: Any) -> str:
    """Return a stable identity for one Ray rollout-engine incarnation."""

    actor_id = getattr(engine, "_actor_id", None)
    if actor_id is None:
        return f"object:{id(engine)}"
    as_hex = getattr(actor_id, "hex", None)
    value = as_hex() if callable(as_hex) else str(actor_id)
    return f"actor:{value}"


def is_megatron_lora_parameter(name: str) -> bool:
    """Return whether a Megatron Bridge parameter belongs to the adapter."""

    return name.startswith("adapter.") or ".adapter." in name


@torch.no_grad()
def zero_megatron_lora_adapters(model: Sequence[torch.nn.Module]) -> int:
    """Reset adapter tensors so a reference snapshot represents the frozen base."""

    zeroed = 0
    for model_chunk in model:
        for name, parameter in model_chunk.named_parameters():
            if is_megatron_lora_parameter(name):
                parameter.zero_()
                zeroed += 1
    if zeroed == 0:
        raise RuntimeError("Megatron LoRA is enabled but the model has no adapter parameters to reset")
    return zeroed


def validate_sglang_base_checksum_response(
    response: Any,
    *,
    engine_index: int,
    tie_word_embeddings: bool,
) -> str:
    """Validate a frozen-base checksum without rejecting configured tied heads."""

    if not isinstance(response, Mapping) or response.get("success") is not True:
        raise RuntimeError(f"SGLang engine {engine_index} base checksum failed: {response!r}")
    checksum = response.get("per_engine_checksum")
    if not isinstance(checksum, str) or not checksum:
        raise RuntimeError(f"SGLang engine {engine_index} returned no base checksum: {response!r}")

    ranks = response.get("ranks")
    rank_payloads = ranks if isinstance(ranks, list) else [ranks]
    tensor_checksums: dict[str, str] = {}
    for rank_payload in rank_payloads:
        if not isinstance(rank_payload, Mapping):
            continue
        checksums = rank_payload.get("checksums")
        if isinstance(checksums, Mapping):
            tensor_checksums.update((str(name), str(value)) for name, value in checksums.items())

    embed = {value for name, value in tensor_checksums.items() if name.endswith("embed_tokens.weight")}
    lm_head = {value for name, value in tensor_checksums.items() if name.endswith("lm_head.weight")}
    if not embed:
        raise RuntimeError(f"SGLang engine {engine_index} checksum omitted embed_tokens")
    if tie_word_embeddings:
        if lm_head and not (embed & lm_head):
            raise RuntimeError(
                f"SGLang engine {engine_index} is configured with tied embeddings but "
                "embed_tokens/lm_head checksums differ"
            )
    else:
        if not lm_head:
            raise RuntimeError(f"SGLang engine {engine_index} checksum omitted untied lm_head")
        if embed & lm_head:
            raise RuntimeError(
                f"SGLang engine {engine_index} has identical untied embed_tokens/lm_head checksums: "
                f"{sorted(embed & lm_head)}"
            )
    return checksum


def replace_adapter_task_weights_from_backup(adapter_tasks, backup_weights):
    """Point Bridge adapter merge tasks at Slime's resident CPU actor snapshot.

    A colocated actor is physically paused by ``torch_memory_saver`` while
    SGLang owns the GPU.  Slime already makes base-weight conversion tasks use
    the pinned CPU actor snapshot, but Bridge constructs LoRA adapter tasks
    internally.  Leaving those tasks attached to the paused model allocation
    causes an illegal CUDA access during the merged HF export.
    """

    def _replace_task(task):
        if task.param_weight is None:
            return task
        key = f"vp_stages.{task.vp_stage}.{task.param_name}"
        if key not in backup_weights:
            raise KeyError(f"LoRA adapter weight {key!r} is missing from the actor backup")
        return replace(task, param_weight=backup_weights[key].cuda())

    return [
        replace(
            adapter_task,
            linear_in_task=_replace_task(adapter_task.linear_in_task),
            linear_out_task=_replace_task(adapter_task.linear_out_task),
        )
        for adapter_task in adapter_tasks
    ]


def apply_megatron_lora(
    model: torch.nn.Module,
    args: Namespace,
    *,
    lora_cls: type | None = None,
) -> torch.nn.Module:
    """Apply Bridge LoRA and fail closed if any base parameter remains trainable."""

    rank = int(getattr(args, "megatron_lora_rank", 0) or 0)
    if rank == 0:
        return model

    if lora_cls is None:
        from megatron.bridge.peft.lora import LoRA

        lora_cls = LoRA

    lora_kwargs: dict[str, Any] = {
        "dim": rank,
        "alpha": int(args.megatron_lora_alpha),
        "dropout": float(args.megatron_lora_dropout),
    }
    target_modules = getattr(args, "megatron_lora_target_modules", None)
    if target_modules is not None:
        lora_kwargs["target_modules"] = list(target_modules)

    model = lora_cls(**lora_kwargs)(model, training=True)
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("Megatron Bridge LoRA produced no trainable parameters")
    unexpected = [name for name, _ in trainable if not is_megatron_lora_parameter(name)]
    if unexpected:
        raise RuntimeError(f"Megatron Bridge LoRA left non-adapter parameters trainable: {unexpected[:8]}")

    logger.info(
        "Enabled Megatron Bridge LoRA rank=%d alpha=%d on %d tensors (%d parameters)",
        rank,
        int(args.megatron_lora_alpha),
        len(trainable),
        sum(parameter.numel() for _, parameter in trainable),
    )
    return model


@torch.no_grad()
def collect_lora_train_metrics(model: Sequence[torch.nn.Module]) -> dict[str, float]:
    """Summarize adapter-only trainability and the post-step LoRA-B update."""

    adapter_parameters = 0
    base_trainable_parameters = 0
    lora_b_nonzero = 0
    lora_b_l1 = 0.0
    for model_chunk in model:
        for name, parameter in model_chunk.named_parameters():
            is_adapter = is_megatron_lora_parameter(name)
            if parameter.requires_grad:
                if is_adapter:
                    adapter_parameters += parameter.numel()
                else:
                    base_trainable_parameters += parameter.numel()
            if is_adapter and ".linear_out." in name:
                value = parameter.detach().float()
                lora_b_nonzero += int(torch.count_nonzero(value).item())
                lora_b_l1 += float(value.abs().sum().item())

    return {
        "train/lora_trainable_parameters": float(adapter_parameters),
        "train/lora_base_trainable_parameters": float(base_trainable_parameters),
        "train/lora_b_nonzero": float(lora_b_nonzero),
        "train/lora_b_l1": lora_b_l1,
    }
