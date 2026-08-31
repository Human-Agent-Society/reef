"""Admission policy for artifacts that carry a Hugging Face PEFT adapter.

A scenario whose artifacts are LoRA/PEFT adapters (an offline SFT run, a
distillation output, or a checkpoint exported by Reef's own training) is
admitted only when the artifact is a servable adapter for the base model the
engine actually holds; serving an adapter fit to another base would apply it
to weights it never saw.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reef.artifact.artifact import Artifact
from reef.core.errors import ReefError

ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")


class AdapterArtifactError(ReefError):
    """An artifact does not carry a servable PEFT adapter."""


def read_peft_config(local_path: Path) -> Mapping[str, Any]:
    """Parse the Hugging Face PEFT ``adapter_config.json`` at an artifact root."""
    config_path = local_path / ADAPTER_CONFIG
    if not config_path.is_file():
        raise AdapterArtifactError(f"adapter artifact has no {ADAPTER_CONFIG}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdapterArtifactError(f"{ADAPTER_CONFIG} is not readable JSON: {exc}") from exc
    if not isinstance(config, Mapping):
        raise AdapterArtifactError(f"{ADAPTER_CONFIG} must contain a JSON object")
    return config


@dataclass(frozen=True)
class PEFTValidator:
    """Validate a PEFT artifact and its optional base-model binding."""

    base_model: str | None = None

    def validate(self, artifact: Artifact) -> None:
        local_path = artifact.materialize().local_path
        if local_path is None:
            raise AdapterArtifactError("adapter artifact has no local content to validate")
        root = Path(local_path)
        config = read_peft_config(root)

        peft_type = config.get("peft_type")
        if not isinstance(peft_type, str) or not peft_type:
            raise AdapterArtifactError(f"{ADAPTER_CONFIG} must declare a peft_type")

        # Check shape without reading tensors or adding a torch dependency to
        # the publication path.
        if not any((root / name).is_file() for name in ADAPTER_WEIGHTS):
            raise AdapterArtifactError(f"adapter artifact carries no adapter weights ({' or '.join(ADAPTER_WEIGHTS)})")

        rank = config.get("r")
        if rank is not None and (not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0):
            raise AdapterArtifactError(f"{ADAPTER_CONFIG} r must be a positive integer")

        base_model = config.get("base_model_name_or_path")
        if self.base_model is not None and base_model != self.base_model:
            raise AdapterArtifactError(
                f"adapter was fit to base model {base_model!r} but this scenario serves "
                f"{self.base_model!r}; serving it would apply the adapter to weights it never saw"
            )


__all__ = ["ADAPTER_CONFIG", "ADAPTER_WEIGHTS", "AdapterArtifactError", "PEFTValidator", "read_peft_config"]
