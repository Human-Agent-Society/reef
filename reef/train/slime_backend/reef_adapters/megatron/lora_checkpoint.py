"""Persist Megatron Bridge LoRA weights as a loadable PEFT adapter."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from reef.artifact.peft import ADAPTER_CONFIG, ADAPTER_PROVENANCE, PROVENANCE_SCHEMA
from reef.train.slime_backend.reef_adapters.megatron.lora import build_sglang_lora_config, is_lora_weight_name

ADAPTER_WEIGHTS_FILE = "adapter_model.safetensors"
#: Files whose identity a tokenizer or chat template change would alter. The
#: base checkpoint owns them; the adapter only records which ones it was fit
#: against, so a later base swap is visible instead of silent.
TOKENIZER_FILES = ("tokenizer_config.json", "tokenizer.json", "chat_template.jinja")


def save_lora_adapter_to_path(
    args: Any,
    output_dir: str | Path,
    adapter_tensors,
    *,
    scenario: str | None = None,
    scenario_step: int | None = None,
) -> None:
    """Gather a replicated/sharded Bridge export and write one PEFT adapter.

    Bridge conversion runs on every Megatron rank because tensor-parallel
    exports may contain collectives. Duplicate tensors from data/tensor
    parallel replicas must agree exactly; pipeline-local tensors are merged.

    The directory is a portable Hugging Face PEFT adapter plus Reef's
    provenance sidecar, so it loads with plain ``transformers`` + ``peft`` and
    still records which training step produced it.
    """

    local_tensors: list[tuple[str, torch.Tensor]] = [(name, _cpu_tensor(tensor)) for name, tensor in adapter_tensors]
    _validate_adapter_tensors(local_tensors)

    group = None
    rank = 0
    gathered: list[list[tuple[str, torch.Tensor]] | None] | None = [local_tensors]
    if dist.is_available() and dist.is_initialized():
        from slime.utils.distributed_utils import get_gloo_group

        group = get_gloo_group()
        rank = dist.get_rank(group)
        gathered = [None] * dist.get_world_size(group) if rank == 0 else None
        dist.gather_object(local_tensors, gathered, dst=0, group=group)

    error = None
    if rank == 0:
        try:
            if gathered is None:
                raise RuntimeError("root rank did not allocate a LoRA tensor gather buffer")
            merged = _merge_adapter_tensors(gathered)
            _write_adapter_checkpoint(
                args,
                Path(output_dir),
                merged,
                scenario=scenario,
                scenario_step=scenario_step,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

    if group is not None:
        payload = [error]
        dist.broadcast_object_list(payload, src=0, group=group)
        error = payload[0]
    if error is not None:
        raise RuntimeError(f"failed to save LoRA adapter checkpoint: {error}")


def _merge_adapter_tensors(per_rank_tensors) -> dict[str, torch.Tensor]:
    merged: dict[str, torch.Tensor] = {}
    for tensors in per_rank_tensors:
        if tensors is None:
            continue
        for name, tensor in tensors:
            name = _peft_adapter_name(name)
            existing = merged.get(name)
            if existing is None:
                merged[name] = tensor
            elif existing.shape != tensor.shape or existing.dtype != tensor.dtype or not torch.equal(existing, tensor):
                raise ValueError(f"Megatron ranks exported conflicting LoRA tensor {name!r}")
    if not merged:
        raise ValueError("Megatron Bridge exported zero LoRA tensors")
    return merged


def _peft_adapter_name(name: str) -> str:
    if name.startswith("base_model."):
        return name
    return f"base_model.model.{name}"


def _write_adapter_checkpoint(
    args: Any,
    path: Path,
    tensors: dict[str, torch.Tensor],
    *,
    scenario: str | None = None,
    scenario_step: int | None = None,
) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    peft_config = build_sglang_lora_config(args)
    base_model = str(args.hf_checkpoint)
    config = {**peft_config, "base_model_name_or_path": base_model, "inference_mode": True}
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir()
    try:
        config_path = temporary / ADAPTER_CONFIG
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        weights_path = temporary / ADAPTER_WEIGHTS_FILE
        save_file(dict(sorted(tensors.items())), weights_path, metadata={"format": "pt"})
        # Checksums cover the two PEFT files, so the sidecar is written last
        # and never hashes itself.
        provenance_path = temporary / ADAPTER_PROVENANCE
        provenance = {
            "schema": PROVENANCE_SCHEMA,
            "base_model": {"name_or_path": base_model, "tokenizer": _tokenizer_identity(base_model)},
            "peft": dict(peft_config),
            # Reported from the tensors actually written, not from a training
            # flag: the artifact's dtype is whatever a loader will find in it.
            "dtype": _tensor_dtype(tensors),
            "source": {"scenario": scenario, "scenario_step": scenario_step},
            "files": {
                ADAPTER_CONFIG: _digest(config_path),
                ADAPTER_WEIGHTS_FILE: _digest(weights_path),
            },
        }
        provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for checkpoint_file in (config_path, weights_path, provenance_path):
            descriptor = os.open(checkpoint_file, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        _fsync_directory(temporary)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _tensor_dtype(tensors: dict[str, torch.Tensor]) -> str | None:
    """The one dtype every adapter tensor shares, or ``None`` when they differ.

    A mixed export is not rejected here — the tensors are already written and
    a loader can still read them — but it must not be recorded as a single
    dtype an auditor would then trust.
    """
    dtypes = {tensor.dtype for tensor in tensors.values()}
    if len(dtypes) != 1:
        return None
    return str(dtypes.pop()).removeprefix("torch.")


def _tokenizer_identity(base_model: str) -> dict[str, str]:
    """Digest whichever tokenizer and chat-template files the base checkpoint has.

    Recorded, never enforced here: the adapter cannot know which base a future
    deployment will pair it with, only which one it was fit against.
    """
    root = Path(base_model)
    identity: dict[str, str] = {}
    if not root.is_dir():
        return identity
    for name in TOKENIZER_FILES:
        candidate = root / name
        if candidate.is_file():
            identity[name] = _digest(candidate)
    return identity


def _validate_adapter_tensors(tensors) -> None:
    if not tensors:
        raise ValueError("Megatron Bridge exported zero LoRA tensors")
    unexpected = [name for name, _ in tensors if not is_lora_weight_name(name)]
    if unexpected:
        raise ValueError(f"LoRA checkpoint contains frozen/base tensors: {unexpected[:8]}")


def _cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach()
    if not value.is_contiguous():
        value = value.contiguous()
    return value.cpu()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = ["save_lora_adapter_to_path"]
