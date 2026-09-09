"""Admission policy for artifacts that carry a Hugging Face PEFT adapter.

A scenario whose artifacts are LoRA/PEFT adapters (an offline SFT run, a
distillation output, or a checkpoint exported by Reef's own training) is
admitted only when the artifact is a servable adapter for the base model the
engine actually holds; serving an adapter fit to another base would apply it
to weights it never saw.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reef.artifact.artifact import Artifact
from reef.core.errors import ReefError

ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")
#: Reef's provenance sidecar. PEFT loaders ignore files they do not know, so
#: an artifact carrying one still loads with plain ``transformers`` + ``peft``.
ADAPTER_PROVENANCE = "reef-adapter.json"
#: Bumped when a field changes meaning. A reader that does not know a schema
#: refuses the artifact rather than validating it against the wrong rules.
PROVENANCE_SCHEMA = 1
#: The PEFT settings a provenance document restates, so a disagreement between
#: the export and the config it claims to have written is caught at admission.
DECLARED_PEFT_KEYS = ("peft_type", "r", "lora_alpha", "lora_dropout", "target_modules")


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


def _digest(path: Path) -> str:
    """SHA-256 of one artifact file, read in chunks so a large adapter streams."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_provenance(local_path: Path) -> Mapping[str, Any] | None:
    """Parse Reef's provenance sidecar, or ``None`` when the artifact has none.

    Absence is not an error: an adapter from an offline SFT run or the Hub is
    a legitimate artifact, it just cannot be audited back to a training step.
    """
    provenance_path = local_path / ADAPTER_PROVENANCE
    if not provenance_path.is_file():
        return None
    try:
        document = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdapterArtifactError(f"{ADAPTER_PROVENANCE} is not readable JSON: {exc}") from exc
    if not isinstance(document, Mapping):
        raise AdapterArtifactError(f"{ADAPTER_PROVENANCE} must contain a JSON object")
    schema = document.get("schema")
    if schema != PROVENANCE_SCHEMA:
        raise AdapterArtifactError(
            f"{ADAPTER_PROVENANCE} declares schema {schema!r}, but this Reef understands {PROVENANCE_SCHEMA}"
        )
    return document


def _comparable(value: Any) -> Any:
    """Normalize a declared PEFT value so JSON round-tripping cannot fail a match.

    ``target_modules`` is a set in PEFT and a list on disk, and a dropout of
    ``0`` and ``0.0`` are the same setting written by different writers.
    """
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return sorted(_comparable(item) for item in value)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return value
    return float(value)


@dataclass(frozen=True)
class PEFTValidator:
    """Validate a PEFT artifact, its base-model binding, and its provenance.

    ``require_provenance`` is for scenarios served only by Reef's own exports:
    it refuses an adapter that cannot be audited back to the training step
    that produced it. Leave it off where hand-built or Hub adapters are
    legitimate — a sidecar that *is* present is always checked either way.
    """

    base_model: str | None = None
    require_provenance: bool = False

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

        provenance = read_provenance(root)
        if provenance is None:
            if self.require_provenance:
                raise AdapterArtifactError(
                    f"adapter artifact has no {ADAPTER_PROVENANCE}; this scenario serves only adapters "
                    "Reef exported, which carry the training step and checksums that make one auditable"
                )
            return
        self._validate_provenance(root, config, provenance)

    def _validate_provenance(self, root: Path, config: Mapping[str, Any], provenance: Mapping[str, Any]) -> None:
        """Check the sidecar against the bytes and the config it claims to describe."""
        declared_base = _mapping(provenance, "base_model").get("name_or_path")
        config_base = config.get("base_model_name_or_path")
        if declared_base != config_base:
            raise AdapterArtifactError(
                f"{ADAPTER_PROVENANCE} records base model {declared_base!r} but {ADAPTER_CONFIG} says "
                f"{config_base!r}; the export and the artifact disagree about what it was fit to"
            )

        declared_peft = _mapping(provenance, "peft")
        for key in DECLARED_PEFT_KEYS:
            if key not in declared_peft:
                continue
            if _comparable(declared_peft[key]) != _comparable(config.get(key)):
                raise AdapterArtifactError(
                    f"{ADAPTER_PROVENANCE} records {key}={declared_peft[key]!r} but {ADAPTER_CONFIG} says "
                    f"{config.get(key)!r}; the exported tensors match only one of them"
                )

        files = _mapping(provenance, "files")
        if not files:
            raise AdapterArtifactError(f"{ADAPTER_PROVENANCE} records no file checksums, so it audits nothing")
        for name, expected in sorted(files.items()):
            if not isinstance(expected, str) or not expected:
                raise AdapterArtifactError(f"{ADAPTER_PROVENANCE} checksum for {name!r} must be a non-empty string")
            target = root / name
            if not target.is_file():
                raise AdapterArtifactError(
                    f"{ADAPTER_PROVENANCE} covers {name!r}, which the artifact does not contain; "
                    "the adapter is incomplete"
                )
            if (actual := _digest(target)) != expected:
                raise AdapterArtifactError(
                    f"{name!r} hashes to {actual} but {ADAPTER_PROVENANCE} recorded {expected}; "
                    "the adapter was corrupted or modified after export"
                )


def _mapping(document: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = document.get(key, {})
    if not isinstance(value, Mapping):
        raise AdapterArtifactError(f"{ADAPTER_PROVENANCE} field {key!r} must be an object")
    return value


__all__ = [
    "ADAPTER_CONFIG",
    "ADAPTER_PROVENANCE",
    "ADAPTER_WEIGHTS",
    "DECLARED_PEFT_KEYS",
    "PROVENANCE_SCHEMA",
    "AdapterArtifactError",
    "PEFTValidator",
    "read_peft_config",
    "read_provenance",
]
