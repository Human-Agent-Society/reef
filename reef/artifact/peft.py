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
#: Training metadata Reef writes beside the PEFT files. Loaders ignore files
#: they do not know, so an artifact carrying it still loads with plain
#: ``transformers`` + ``peft``.
ADAPTER_METADATA = "reef-adapter.json"
#: Increase this when a field changes meaning. A reader that does not know
#: the schema refuses the artifact instead of checking it against wrong rules.
METADATA_SCHEMA = 1
#: PEFT settings the metadata repeats from ``adapter_config.json``, so an
#: export that disagrees with the config it wrote is caught before serving.
_DECLARED_PEFT_KEYS = ("peft_type", "r", "lora_alpha", "lora_dropout", "target_modules")


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


def read_adapter_metadata(local_path: Path) -> Mapping[str, Any] | None:
    """Parse the adapter's training metadata, or ``None`` when the file is absent.

    A missing file is not an error. An adapter from an offline SFT run or from
    the Hub is still valid; it just does not say which training step wrote it.
    """
    metadata_path = local_path / ADAPTER_METADATA
    if not metadata_path.is_file():
        return None
    try:
        document = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdapterArtifactError(f"{ADAPTER_METADATA} is not readable JSON: {exc}") from exc
    if not isinstance(document, Mapping):
        raise AdapterArtifactError(f"{ADAPTER_METADATA} must contain a JSON object")
    schema = document.get("schema")
    if schema != METADATA_SCHEMA:
        raise AdapterArtifactError(
            f"{ADAPTER_METADATA} declares schema {schema!r}, but this Reef understands {METADATA_SCHEMA}"
        )
    return document


def _comparable(value: Any) -> Any:
    """Normalize a PEFT value so a JSON round trip does not break a comparison.

    PEFT holds ``target_modules`` as a set but writes a list, so the order on
    disk is arbitrary, and ``0`` and ``0.0`` are the same dropout.
    """
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return sorted(_comparable(item) for item in value)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return value
    return float(value)


@dataclass(frozen=True)
class PEFTValidator:
    """Validate a PEFT artifact, the base model it names, and its metadata.

    Set ``require_metadata`` when a scenario serves only adapters Reef
    exported: it then refuses one that does not say which training step wrote
    it. Leave it off when hand-built or Hub adapters are expected. Metadata
    that is present is checked either way.
    """

    base_model: str | None = None
    require_metadata: bool = False

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

        metadata = read_adapter_metadata(root)
        if metadata is None:
            if self.require_metadata:
                raise AdapterArtifactError(
                    f"adapter artifact has no {ADAPTER_METADATA}; this scenario serves only adapters "
                    "Reef exported, which record the training step that wrote them and a checksum per file"
                )
            return
        self._validate_metadata(root, config, metadata)

    def _validate_metadata(self, root: Path, config: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
        """Check the metadata against the files and the config it describes."""
        declared_base = _mapping(metadata, "base_model").get("name_or_path")
        config_base = config.get("base_model_name_or_path")
        if declared_base != config_base:
            raise AdapterArtifactError(
                f"{ADAPTER_METADATA} records base model {declared_base!r} but {ADAPTER_CONFIG} says "
                f"{config_base!r}; the two files disagree about which base model this adapter is for"
            )

        declared_peft = _mapping(metadata, "peft")
        for key in _DECLARED_PEFT_KEYS:
            if key not in declared_peft:
                continue
            if _comparable(declared_peft[key]) != _comparable(config.get(key)):
                raise AdapterArtifactError(
                    f"{ADAPTER_METADATA} records {key}={declared_peft[key]!r} but {ADAPTER_CONFIG} says "
                    f"{config.get(key)!r}; the exported weights can only match one of them"
                )

        files = _mapping(metadata, "files")
        if not files:
            raise AdapterArtifactError(f"{ADAPTER_METADATA} records no file checksums, so it verifies nothing")
        for name, expected in sorted(files.items()):
            if not isinstance(expected, str) or not expected:
                raise AdapterArtifactError(f"{ADAPTER_METADATA} checksum for {name!r} must be a non-empty string")
            target = root / name
            if not target.is_file():
                raise AdapterArtifactError(
                    f"{ADAPTER_METADATA} lists {name!r}, which the artifact does not contain; "
                    "the adapter is incomplete"
                )
            if (actual := _digest(target)) != expected:
                raise AdapterArtifactError(
                    f"{name!r} hashes to {actual} but {ADAPTER_METADATA} recorded {expected}; "
                    "the adapter was corrupted or modified after export"
                )


def _mapping(document: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = document.get(key, {})
    if not isinstance(value, Mapping):
        raise AdapterArtifactError(f"{ADAPTER_METADATA} field {key!r} must be an object")
    return value


__all__ = [
    "ADAPTER_CONFIG",
    "ADAPTER_METADATA",
    "ADAPTER_WEIGHTS",
    "METADATA_SCHEMA",
    "AdapterArtifactError",
    "PEFTValidator",
    "read_adapter_metadata",
    "read_peft_config",
]
